import unittest
import torch
import os
import sys

# Add GDesigner root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gdesigner_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if gdesigner_root not in sys.path:
    sys.path.insert(0, gdesigner_root)

from GDesigner.gtd.proxy_reward_model import ProxyRewardModel

class TestProxyRewardModel(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.batch_size = 2
        self.num_nodes = 6
        self.node_feature_dim = 10
        self.condition_dim = 12
        self.gnn_hidden_dim = 16
        self.gnn_layers = 2
        self.mlp_hidden_dim = 20
        self.num_reward_components = 3 # e.g., utility, cost, robustness

        self.proxy_model = ProxyRewardModel(
            node_feature_dim=self.node_feature_dim,
            condition_dim=self.condition_dim,
            gnn_hidden_dim=self.gnn_hidden_dim,
            gnn_layers=self.gnn_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
            num_reward_components=self.num_reward_components
        ).to(self.device)
        self.proxy_model.eval() # Set to eval mode for tests

        # Dummy inputs
        self.node_features = torch.randn(self.batch_size, self.num_nodes, self.node_feature_dim).to(self.device)
        # Adjacency matrix (binary)
        self.adj_matrix_binary = (torch.rand(self.batch_size, self.num_nodes, self.num_nodes) > 0.5).float().to(self.device)
        # Adjacency matrix (probabilities)
        self.adj_matrix_probs = torch.rand(self.batch_size, self.num_nodes, self.num_nodes).to(self.device)
        self.condition = torch.randn(self.batch_size, self.condition_dim).to(self.device)

    def test_forward_pass_shape_binary_adj(self):
        predicted_rewards = self.proxy_model(
            node_features=self.node_features,
            adj_matrix=self.adj_matrix_binary,
            condition=self.condition
        )
        self.assertEqual(predicted_rewards.shape, (self.batch_size, self.num_reward_components))

    def test_forward_pass_shape_prob_adj(self):
        # Proxy model should internally binarize probability adj matrix if needed
        predicted_rewards = self.proxy_model(
            node_features=self.node_features,
            adj_matrix=self.adj_matrix_probs, # Input probabilities
            condition=self.condition
        )
        self.assertEqual(predicted_rewards.shape, (self.batch_size, self.num_reward_components))

    def test_forward_pass_different_batch_size(self):
        bs = 1
        node_feat_s = torch.randn(bs, self.num_nodes, self.node_feature_dim).to(self.device)
        adj_mat_s = (torch.rand(bs, self.num_nodes, self.num_nodes) > 0.5).float().to(self.device)
        cond_s = torch.randn(bs, self.condition_dim).to(self.device)

        predicted_rewards = self.proxy_model(node_feat_s, adj_mat_s, cond_s)
        self.assertEqual(predicted_rewards.shape, (bs, self.num_reward_components))

    def test_gnn_layers_exist(self):
        self.assertTrue(len(self.proxy_model.gnn_layers) == self.gnn_layers)
        # Check if GATConv (or chosen GNN layer) is used
        from torch_geometric.nn import GATConv # Or GCNConv depending on ProxyRewardModel's default
        self.assertIsInstance(self.proxy_model.gnn_layers[0], GATConv)


if __name__ == '__main__':
    unittest.main()
```
