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
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_dense

class TestProxyRewardModelPyG(unittest.TestCase): # Renamed class

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.node_feature_dim = 10
        self.condition_dim = 12 # Graph-level condition
        self.gnn_hidden_dim = 16
        self.gnn_layers = 2
        self.mlp_hidden_dim = 20
        self.num_reward_components = 3

        self.proxy_model = ProxyRewardModel(
            node_feature_dim=self.node_feature_dim,
            condition_dim=self.condition_dim,
            gnn_hidden_dim=self.gnn_hidden_dim,
            gnn_layers=self.gnn_layers,
            mlp_hidden_dim=self.mlp_hidden_dim,
            num_reward_components=self.num_reward_components
        ).to(self.device)
        self.proxy_model.eval()

        # Dummy PyG Batch data
        self.batch_size = 2
        self.num_nodes_graph1 = 6
        self.num_nodes_graph2 = 4 # Different number of nodes for graph 2

        # Graph 1
        x1 = torch.randn(self.num_nodes_graph1, self.node_feature_dim)
        adj1 = (torch.rand(self.num_nodes_graph1, self.num_nodes_graph1) > 0.5).float()
        edge_index1, _ = from_dense(adj1)
        condition1 = torch.randn(1, self.condition_dim) # (1, condition_dim) for graph-level attr
        data1 = Data(x=x1, edge_index=edge_index1, condition=condition1)

        # Graph 2
        x2 = torch.randn(self.num_nodes_graph2, self.node_feature_dim)
        adj2 = (torch.rand(self.num_nodes_graph2, self.num_nodes_graph2) > 0.5).float()
        edge_index2, _ = from_dense(adj2)
        condition2 = torch.randn(1, self.condition_dim)
        data2 = Data(x=x2, edge_index=edge_index2, condition=condition2)

        self.pyg_batch = Batch.from_data_list([data1, data2]).to(self.device)


    def test_forward_pass_shape_with_pyg_batch(self):
        predicted_rewards = self.proxy_model(self.pyg_batch)
        self.assertEqual(predicted_rewards.shape, (self.batch_size, self.num_reward_components))

    def test_forward_pass_single_graph_in_batch(self):
        # Create a batch with a single graph
        x_single = torch.randn(self.num_nodes_graph1, self.node_feature_dim)
        adj_single = (torch.rand(self.num_nodes_graph1, self.num_nodes_graph1) > 0.5).float()
        edge_index_single, _ = from_dense(adj_single)
        condition_single = torch.randn(1, self.condition_dim)
        data_single = Data(x=x_single, edge_index=edge_index_single, condition=condition_single)

        pyg_batch_single = Batch.from_data_list([data_single]).to(self.device)

        predicted_rewards = self.proxy_model(pyg_batch_single)
        self.assertEqual(predicted_rewards.shape, (1, self.num_reward_components))

    def test_gnn_layers_exist(self): # This test remains largely the same
        self.assertTrue(len(self.proxy_model.gnn_layers) == self.gnn_layers)
        from torch_geometric.nn import GATConv
        self.assertIsInstance(self.proxy_model.gnn_layers[0], GATConv)

    def test_missing_condition_attribute(self):
        # Create a batch without the 'condition' attribute
        data_no_cond = Data(x=torch.randn(3, self.node_feature_dim),
                            edge_index=torch.tensor([[0,1],[1,2]], dtype=torch.long))
        batch_no_cond = Batch.from_data_list([data_no_cond]).to(self.device)

        with self.assertRaisesRegex(ValueError, "pyg_batch must have a 'condition' attribute"):
            self.proxy_model(batch_no_cond)

    def test_condition_shape_mismatch(self):
        # Create data where condition is not (1, cond_dim) before batching, leading to wrong shape after batching.
        # For example, if condition was (cond_dim) instead of (1, cond_dim), PyG Batch might handle it differently
        # or if it was (total_nodes, cond_dim) and not handled.
        # The model error check is `condition_tensor.shape[0] != pooled_embeddings.shape[0]`

        x1 = torch.randn(self.num_nodes_graph1, self.node_feature_dim)
        edge_index1, _ = from_dense((torch.rand(self.num_nodes_graph1, self.num_nodes_graph1) > 0.5).float())
        # Incorrectly shaped condition (e.g. if it was per-node and not handled)
        # Let's simulate if condition became (total_nodes, cond_dim) instead of (batch_size, cond_dim)
        # This is hard to directly simulate without deeper mock of Batch.from_data_list behavior for custom attrs.
        # The check is more about if `pyg_batch.condition` itself has shape (batch_size, cond_dim).

        # More direct test: create a batch then manually assign a wrongly shaped condition
        temp_batch = self.pyg_batch.clone()
        temp_batch.condition = torch.randn(self.pyg_batch.num_graphs + 1, self.condition_dim).to(self.device) # Wrong batch dim

        with self.assertRaisesRegex(ValueError, "Mismatch in batch size between pooled_embeddings"):
             self.proxy_model(temp_batch)


if __name__ == '__main__':
    unittest.main()
```
