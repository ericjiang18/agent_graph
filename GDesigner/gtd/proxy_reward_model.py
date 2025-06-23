import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv # Using GCNConv as a default, can be GATConv or other GNNs
from torch_geometric.utils import from_dense # Utility to convert dense adj to edge_index for PyG

class ProxyRewardModel(nn.Module):
    """
    A GNN-based model to predict proxy rewards for a given graph topology and condition.
    The model predicts multiple reward components (e.g., utility, cost, robustness).
    """
    def __init__(self,
                 node_feature_dim: int,
                 condition_dim: int,
                 gnn_hidden_dim: int,
                 gnn_layers: int,
                 mlp_hidden_dim: int,
                 num_reward_components: int, # e.g., 3 for utility, cost, robustness
                 dropout_rate=0.1):
        super(ProxyRewardModel, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.condition_dim = condition_dim
        self.num_reward_components = num_reward_components

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        input_dim = node_feature_dim
        for i in range(gnn_layers):
            # Example using GCNConv. Could be GATConv or other GNN layers.
            # self.gnn_layers.append(GCNConv(input_dim, gnn_hidden_dim))
            self.gnn_layers.append(GATConv(input_dim, gnn_hidden_dim, heads=4, concat=False, dropout=dropout_rate)) # Using GAT for potentially better performance
            input_dim = gnn_hidden_dim

        # MLP to process pooled graph embedding and condition
        # The graph embedding will be concatenated with the condition vector
        self.mlp = nn.Sequential(
            nn.Linear(gnn_hidden_dim + condition_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_reward_components)
        )

    def forward(self,
                node_features: torch.Tensor, # (batch_size, num_nodes, node_feature_dim)
                adj_matrix: torch.Tensor,    # (batch_size, num_nodes, num_nodes), binary or probabilities
                condition: torch.Tensor,     # (batch_size, condition_dim)
                batch_indices: torch.Tensor = None # (total_num_nodes_in_batch), for PyG global pooling
               ):
        """
        Forward pass of the proxy reward model.

        Args:
            node_features (torch.Tensor): Batch of node features.
            adj_matrix (torch.Tensor): Batch of adjacency matrices (can be binary or probabilities).
                                       If probabilities, thresholding might be needed for edge_index.
            condition (torch.Tensor): Batch of condition vectors.
            batch_indices (torch.Tensor, optional): Tensor indicating which graph each node belongs to.
                                                   Required if graphs in the batch have different sizes.
                                                   If None, assumes all graphs in batch have same num_nodes.

        Returns:
            torch.Tensor: Predicted reward components (batch_size, num_reward_components).
        """
        batch_size, num_nodes, _ = node_features.shape

        # Convert adjacency matrices to edge_index format for PyG layers
        # This part needs careful handling of batching if done manually.
        # PyG's DataLoader usually handles this by creating a single large graph.

        # Assuming node_features and adj_matrix are batched (B, N, D) and (B, N, N)
        # We need to process them graph by graph or create a PyG Batch object.

        # For simplicity, let's process each graph in the batch individually if batch_indices is not provided.
        # A more efficient way is to construct a PyG Batch object outside this model.

        # If adj_matrix contains probabilities, binarize it first
        if adj_matrix.dtype == torch.float32 and adj_matrix.max() <= 1.0 and adj_matrix.min() >=0.0:
             adj_binary = (adj_matrix > 0.5).float()
        else: # assume it's already binary or int
             adj_binary = adj_matrix.float()

        # Process graphs in batch (can be optimized by PyG Batch)
        graph_embeddings = []
        for i in range(batch_size):
            nf = node_features[i] # (num_nodes, node_feature_dim)
            am = adj_binary[i]    # (num_nodes, num_nodes)

            edge_index, _ = from_dense(am) # Get edge_index for PyG
            edge_index = edge_index.to(nf.device)

            h = nf
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index)
                h = F.relu(h)

            # Global pooling (e.g., mean pooling)
            # If batch_indices is provided, it's for a single large graph composed of the batch.
            # Here, we are iterating, so pool over the current graph's nodes.
            current_graph_embedding = torch.mean(h, dim=0) # (gnn_hidden_dim)
            graph_embeddings.append(current_graph_embedding)

        pooled_embeddings = torch.stack(graph_embeddings) # (batch_size, gnn_hidden_dim)

        # Concatenate pooled graph embedding with the condition vector
        combined_features = torch.cat((pooled_embeddings, condition), dim=1) # (batch_size, gnn_hidden_dim + condition_dim)

        # Pass through MLP to get reward components
        predicted_rewards = self.mlp(combined_features) # (batch_size, num_reward_components)

        return predicted_rewards

# Example Training (Conceptual)
# Assume you have a dataset: List[Tuple[node_features, adj_matrix, condition, true_rewards]]
# true_rewards is a tensor of shape (num_reward_components,)

def train_proxy_model(proxy_model, dataloader, optimizer, criterion, device):
    proxy_model.train()
    total_loss = 0
    for batch_idx, (node_feat_batch, adj_mat_batch, cond_batch, true_rewards_batch) in enumerate(dataloader):
        node_feat_batch = node_feat_batch.to(device)
        adj_mat_batch = adj_mat_batch.to(device)
        cond_batch = cond_batch.to(device)
        true_rewards_batch = true_rewards_batch.to(device) # (batch_size, num_reward_components)

        optimizer.zero_grad()

        pred_rewards_batch = proxy_model(node_feat_batch, adj_mat_batch, cond_batch)

        loss = criterion(pred_rewards_batch, true_rewards_batch) # MSE loss typically
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 50 == 0:
            print(f"ProxyTrain Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters for ProxyRewardModel
    num_nodes_example = 10
    node_feat_dim_proxy = 32
    cond_dim_proxy = 64
    gnn_hidden_proxy = 64
    gnn_layers_proxy = 2
    mlp_hidden_proxy = 128
    num_rewards = 3 # e.g., utility, cost, robustness

    proxy_model = ProxyRewardModel(
        node_feature_dim=node_feat_dim_proxy,
        condition_dim=cond_dim_proxy,
        gnn_hidden_dim=gnn_hidden_proxy,
        gnn_layers=gnn_layers_proxy,
        mlp_hidden_dim=mlp_hidden_proxy,
        num_reward_components=num_rewards
    ).to(device)

    # Dummy data for testing the forward pass
    batch_s_proxy = 4
    dummy_node_features_proxy = torch.randn(batch_s_proxy, num_nodes_example, node_feat_dim_proxy).to(device)
    # Adjacency matrix (binary)
    dummy_adj_matrix_proxy = (torch.rand(batch_s_proxy, num_nodes_example, num_nodes_example) > 0.5).float().to(device)
    dummy_condition_proxy = torch.randn(batch_s_proxy, cond_dim_proxy).to(device)

    print("Testing ProxyRewardModel forward pass...")
    predicted_proxy_rewards = proxy_model(dummy_node_features_proxy, dummy_adj_matrix_proxy, dummy_condition_proxy)
    print(f"Predicted proxy rewards shape: {predicted_proxy_rewards.shape}") # Expected: (batch_s_proxy, num_rewards)
    print("First predicted proxy reward vector:")
    print(predicted_proxy_rewards[0])

    # Conceptual Dataloader and Training Loop
    from torch.utils.data import DataLoader, TensorDataset

    # Create dummy dataset for training proxy model
    num_samples_proxy_train = 100
    train_node_feats = torch.randn(num_samples_proxy_train, num_nodes_example, node_feat_dim_proxy)
    train_adj_mats = (torch.rand(num_samples_proxy_train, num_nodes_example, num_nodes_example) > 0.5).float()
    train_conditions = torch.randn(num_samples_proxy_train, cond_dim_proxy)
    # True rewards: random values for utility (0-1), cost (0-N), robustness (0-1)
    train_true_rewards = torch.cat([
        torch.rand(num_samples_proxy_train, 1), # utility
        torch.rand(num_samples_proxy_train, 1) * num_nodes_example, # cost (e.g. number of edges, scaled)
        torch.rand(num_samples_proxy_train, 1)  # robustness
    ], dim=1)

    proxy_dataset = TensorDataset(train_node_feats, train_adj_mats, train_conditions, train_true_rewards)
    proxy_dataloader = DataLoader(proxy_dataset, batch_size=16, shuffle=True)

    optimizer_proxy = torch.optim.Adam(proxy_model.parameters(), lr=1e-3)
    criterion_proxy = nn.MSELoss() # Mean Squared Error for regression

    print("\nStarting conceptual training of ProxyRewardModel...")
    for epoch in range(3): # Few epochs for demonstration
        avg_loss = train_proxy_model(proxy_model, proxy_dataloader, optimizer_proxy, criterion_proxy, device)
        print(f"Epoch {epoch+1}, Average Proxy Model Training Loss: {avg_loss:.4f}")

    # After training, the proxy_model can be used to predict rewards for candidate graphs.
    print("\nTesting ProxyRewardModel prediction after conceptual training...")
    proxy_model.eval()
    with torch.no_grad():
        sample_pred_rewards = proxy_model(
            dummy_node_features_proxy,
            dummy_adj_matrix_proxy,
            dummy_condition_proxy
        )
        print(f"Sample predicted proxy rewards shape: {sample_pred_rewards.shape}")
        print("First sample predicted proxy reward vector (after 'training'):")
        print(sample_pred_rewards[0])

    # Note on PyTorch Geometric (PyG) integration:
    # For more complex GNNs or varying graph sizes within a batch,
    # it's standard to convert data into PyG's `Data` or `Batch` objects.
    # The current implementation processes graphs in a batch via a loop, which is fine for fixed-size graphs
    # but less efficient than PyG's batching for variable-sized graphs.
    # The `from_dense` utility is used here, which is a PyG tool.
    # If all graphs have the same number of nodes, the current batch processing in `forward` is acceptable.
    # If node_features are global (not per-node), the GNN part might need adjustment or removal.
    # The proposal implies node_features from agent configurations.
    # The GATConv layer was chosen as an example; GCNConv or other layers are also viable.
```
