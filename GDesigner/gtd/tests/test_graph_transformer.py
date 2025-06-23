import unittest
import torch
import os
import sys

# Add GDesigner root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gdesigner_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..')) # GDesigner/gtd/tests -> GDesigner
if gdesigner_root not in sys.path:
    sys.path.insert(0, gdesigner_root)

from GDesigner.gtd.graph_transformer import GraphTransformer, FiLMLayer

class TestGraphTransformer(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Common parameters for tests
        self.batch_size = 2
        self.num_nodes = 5
        self.node_feature_dim = 16
        self.condition_dim = 32
        self.time_embed_dim = 20 # Different from condition_dim to test projection
        self.num_layers = 2
        self.num_heads = 2
        self.output_dim = 1 # For GT's edge predictor (unused in current GT output layer directly)

        self.graph_transformer = GraphTransformer(
            node_feature_dim=self.node_feature_dim,
            condition_dim=self.condition_dim, # Task/agent condition
            time_embed_dim=self.time_embed_dim, # Time embedding
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            output_dim=self.output_dim
        ).to(self.device)

        # Inputs
        self.node_features = torch.randn(self.batch_size, self.num_nodes, self.node_feature_dim).to(self.device)
        self.adj_matrix = (torch.rand(self.batch_size, self.num_nodes, self.num_nodes) > 0.5).float().to(self.device)
        self.task_condition = torch.randn(self.batch_size, self.condition_dim).to(self.device)
        self.time_embedding = torch.randn(self.batch_size, self.time_embed_dim).to(self.device)

    def test_film_layer(self):
        film = FiLMLayer(self.condition_dim, self.node_feature_dim).to(self.device)
        # FiLMLayer expects condition (B, cond_dim) and features (B, N, feat_dim)
        # or (B, feat_dim) if applied directly.
        # In GTLayer, it's (B, N, feat_dim) and condition (B, combined_cond_dim)
        # Let's test with (B,N,feat_dim) for x
        x_film = torch.randn(self.batch_size, self.num_nodes, self.node_feature_dim).to(self.device)
        condition_film = torch.randn(self.batch_size, self.condition_dim).to(self.device)

        output = film(x_film, condition_film)
        self.assertEqual(output.shape, x_film.shape)

    def test_graph_transformer_forward_pass_shapes(self):
        output_probs = self.graph_transformer(
            node_features=self.node_features,
            adj_matrix=self.adj_matrix,
            task_condition=self.task_condition,
            time_embedding=self.time_embedding
        )

        # Expected output shape: (batch_size, num_nodes, num_nodes) for edge probabilities
        self.assertEqual(output_probs.shape, (self.batch_size, self.num_nodes, self.num_nodes))

        # Check probabilities are in [0, 1]
        self.assertTrue(torch.all(output_probs >= 0) and torch.all(output_probs <= 1))

    def test_graph_transformer_forward_pass_no_adj(self):
        # Test with no adjacency matrix (should use full attention)
        output_probs = self.graph_transformer(
            node_features=self.node_features,
            adj_matrix=None, # No explicit mask
            task_condition=self.task_condition,
            time_embedding=self.time_embedding
        )
        self.assertEqual(output_probs.shape, (self.batch_size, self.num_nodes, self.num_nodes))
        self.assertTrue(torch.all(output_probs >= 0) and torch.all(output_probs <= 1))

    def test_graph_transformer_internal_time_projection(self):
        # Time embedding (B, time_embed_dim) is projected to (B, condition_dim)
        # then concatenated with task_condition (B, condition_dim)
        # So, FiLM layer in GraphTransformerLayer receives condition of (B, condition_dim + condition_dim)

        # Accessing the first transformer layer to check its FiLM layer's input condition dim
        first_gt_layer = self.graph_transformer.transformer_layers[0]
        # The FiLMLayer's `condition_dim` should be `self.condition_dim + self.condition_dim`
        # because time_projection projects time_embed_dim to condition_dim.
        expected_film_condition_dim = self.condition_dim + self.condition_dim
        self.assertEqual(first_gt_layer.film.condition_dim, expected_film_condition_dim)

        # Test if time_projection exists and has correct output shape
        projected_time = self.graph_transformer.time_projection(self.time_embedding)
        self.assertEqual(projected_time.shape, (self.batch_size, self.condition_dim))


if __name__ == '__main__':
    unittest.main()
```
