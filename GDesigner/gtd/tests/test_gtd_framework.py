import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import sys

# Add GDesigner root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gdesigner_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if gdesigner_root not in sys.path:
    sys.path.insert(0, gdesigner_root)

from GDesigner.gtd.gtd_framework import GTDFramework
from GDesigner.gtd.proxy_reward_model import ProxyRewardModel

class TestGTDFramework(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Framework parameters
        self.num_nodes = 4
        self.node_feature_dim = 8
        self.condition_dim = 10
        self.time_embed_dim = 12
        self.gt_num_layers = 1
        self.gt_num_heads = 1
        self.diffusion_num_timesteps = 5 # Very few for quick tests

        # Proxy model parameters (for initializing guider)
        self.proxy_num_rewards = 3
        self.macp_weights = {'r1': 1.0, 'r2': -0.5, 'r3': 0.2} # Dummy names, count must match proxy_num_rewards

        self.dummy_proxy_model = ProxyRewardModel(
            node_feature_dim=self.node_feature_dim,
            condition_dim=self.condition_dim,
            gnn_hidden_dim=8, gnn_layers=1, mlp_hidden_dim=10,
            num_reward_components=self.proxy_num_rewards
        ).to(self.device)
        self.dummy_proxy_model.eval()

        # Initialize GTDFramework without guider (for diffusion training tests)
        self.gtd_system_no_guider = GTDFramework(
            node_feature_dim=self.node_feature_dim,
            condition_dim=self.condition_dim,
            time_embed_dim=self.time_embed_dim,
            gt_num_layers=self.gt_num_layers,
            gt_num_heads=self.gt_num_heads,
            diffusion_num_timesteps=self.diffusion_num_timesteps,
            device=self.device
        )

        # Initialize GTDFramework with guider (for guided generation tests)
        self.gtd_system_with_guider = GTDFramework(
            node_feature_dim=self.node_feature_dim,
            condition_dim=self.condition_dim,
            time_embed_dim=self.time_embed_dim,
            gt_num_layers=self.gt_num_layers,
            gt_num_heads=self.gt_num_heads,
            diffusion_num_timesteps=self.diffusion_num_timesteps,
            proxy_reward_model=self.dummy_proxy_model,
            macp_weights=self.macp_weights,
            num_candidates_per_step=3,
            device=self.device
        )

        # Dummy data for training/generation
        self.batch_size = 2
        self.A0_binary = (torch.rand(self.batch_size, self.num_nodes, self.num_nodes) > 0.5).float().to(self.device)
        self.node_features = torch.randn(self.batch_size, self.num_nodes, self.node_feature_dim).to(self.device)
        self.task_condition = torch.randn(self.batch_size, self.condition_dim).to(self.device)


    def test_framework_initialization(self):
        self.assertIsNotNone(self.gtd_system_no_guider.diffusion_model)
        self.assertIsNone(self.gtd_system_no_guider.guider)

        self.assertIsNotNone(self.gtd_system_with_guider.diffusion_model)
        self.assertIsNotNone(self.gtd_system_with_guider.guider)
        self.assertIsInstance(self.gtd_system_with_guider.guider.proxy_reward_model, ProxyRewardModel)

    def test_train_diffusion_model(self):
        # Create a dummy DataLoader
        dataset = TensorDataset(self.A0_binary, self.node_features, self.task_condition)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        # Test if training runs for one epoch without errors
        try:
            self.gtd_system_no_guider.train_diffusion_model(
                dataloader=dataloader,
                epochs=1,
                learning_rate=1e-4
            )
        except Exception as e:
            self.fail(f"train_diffusion_model raised an exception: {e}")

        # Check if model is in eval mode after training
        self.assertFalse(self.gtd_system_no_guider.diffusion_model.training)


    def test_generate_graphs_unguided(self):
        generated_probs = self.gtd_system_no_guider.generate_graphs(
            num_graphs=self.batch_size,
            num_nodes=self.num_nodes,
            node_features=self.node_features,
            task_condition=self.task_condition,
            use_guidance=False
        )
        self.assertEqual(generated_probs.shape,
                         (self.batch_size, self.num_nodes, self.num_nodes))
        self.assertTrue(torch.all(generated_probs >= 0) and torch.all(generated_probs <= 1))
        self.assertFalse(self.gtd_system_no_guider.diffusion_model.training) # Should be in eval

    def test_generate_graphs_guided(self):
        generated_probs = self.gtd_system_with_guider.generate_graphs(
            num_graphs=self.batch_size,
            num_nodes=self.num_nodes,
            node_features=self.node_features,
            task_condition=self.task_condition,
            use_guidance=True
        )
        self.assertEqual(generated_probs.shape,
                         (self.batch_size, self.num_nodes, self.num_nodes))
        self.assertTrue(torch.all(generated_probs >= 0) and torch.all(generated_probs <= 1))
        self.assertFalse(self.gtd_system_with_guider.diffusion_model.training) # Should be in eval
        self.assertFalse(self.gtd_system_with_guider.guider.proxy_reward_model.training) # Proxy should be eval


    def test_generate_graphs_guidance_requested_but_no_guider(self):
        with self.assertRaises(ValueError):
            self.gtd_system_no_guider.generate_graphs(
                num_graphs=self.batch_size,
                num_nodes=self.num_nodes,
                node_features=self.node_features,
                task_condition=self.task_condition,
                use_guidance=True # Request guidance
            )

    def test_generate_graphs_input_validation(self):
        # Wrong batch size for node_features
        wrong_node_features = torch.randn(self.batch_size + 1, self.num_nodes, self.node_feature_dim).to(self.device)
        with self.assertRaises(ValueError):
            self.gtd_system_no_guider.generate_graphs(
                num_graphs=self.batch_size, num_nodes=self.num_nodes,
                node_features=wrong_node_features, task_condition=self.task_condition, use_guidance=False
            )

        # Wrong num_nodes for node_features
        wrong_num_nodes_features = torch.randn(self.batch_size, self.num_nodes + 1, self.node_feature_dim).to(self.device)
        with self.assertRaises(ValueError):
            self.gtd_system_no_guider.generate_graphs(
                num_graphs=self.batch_size, num_nodes=self.num_nodes,
                node_features=wrong_num_nodes_features, task_condition=self.task_condition, use_guidance=False
            )

        # Wrong feature_dim for node_features
        wrong_feat_dim_features = torch.randn(self.batch_size, self.num_nodes, self.node_feature_dim + 1).to(self.device)
        with self.assertRaises(ValueError):
            self.gtd_system_no_guider.generate_graphs(
                num_graphs=self.batch_size, num_nodes=self.num_nodes,
                node_features=wrong_feat_dim_features, task_condition=self.task_condition, use_guidance=False
            )


if __name__ == '__main__':
    unittest.main()
```
