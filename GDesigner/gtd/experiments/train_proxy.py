import torch
# from torch.utils.data import DataLoader, TensorDataset # Standard DataLoader
import torch.optim as optim
import torch.nn as nn
import os
import sys

# Add GDesigner root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gdesigner_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if gdesigner_root not in sys.path:
    sys.path.insert(0, gdesigner_root)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader # PyG DataLoader
from torch_geometric.utils import from_dense


from GDesigner.gtd.proxy_reward_model import ProxyRewardModel
# metrics_and_datasets.py create_proxy_training_data is too high-level for this direct use.
# We will construct PyG Data objects directly here.

def main_train_proxy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # Proxy Model parameters
    num_nodes_avg = 10  # Average number of nodes, can vary per graph
    node_feature_dim = 32
    condition_dim = 64
    gnn_hidden_dim = 64
    gnn_layers = 2
    mlp_hidden_dim = 128
    reward_component_names = ['utility', 'cost', 'vulnerability']
    num_reward_components = len(reward_component_names)

    # Training parameters
    num_train_samples = 512
    batch_size = 64 # PyG DataLoader batch_size
    epochs = 15
    learning_rate = 1e-3

    # --- Dataset Preparation (for PyG) ---
    print("Generating dummy proxy model training data (for PyG)...")

    data_list = []
    for i in range(num_train_samples):
        # Simulate variable number of nodes per graph
        current_num_nodes = num_nodes_avg - 2 + (i % 5) # e.g. 8 to 12 nodes

        # Node features (x)
        x = torch.randn(current_num_nodes, node_feature_dim)

        # Adjacency matrix and edge_index
        adj_matrix = (torch.rand(current_num_nodes, current_num_nodes) > (0.3 + torch.rand(1).item()*0.4)).float()
        edge_index, _ = from_dense(adj_matrix)

        # Condition vector (graph-level attribute)
        # Shape: (1, condition_dim) to be correctly handled by PyG Batch for stacking
        condition_vec = torch.randn(1, condition_dim)

        # True rewards (graph-level attribute, e.g., data.y or data.true_rewards)
        # Shape: (1, num_reward_components)
        # Dummy rewards for utility, cost, vulnerability
        utility = torch.rand(1).item()
        cost = adj_matrix.sum().item() / (current_num_nodes * (current_num_nodes -1) + 1e-5) # Normalized
        vulnerability = torch.rand(1).item() * 0.5
        true_rewards_vec = torch.tensor([[utility, cost, vulnerability]], dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, condition=condition_vec, true_rewards=true_rewards_vec)
        data_list.append(data)

    # Use PyG DataLoader
    pyg_dataloader = PyGDataLoader(data_list, batch_size=batch_size, shuffle=True)
    print(f"Created PyG DataLoader for proxy training with {len(pyg_dataloader)} batches.")


    # --- Initialize Proxy Reward Model ---
    print("Initializing ProxyRewardModel...")
    proxy_model = ProxyRewardModel(
        node_feature_dim=node_feature_dim,
        condition_dim=condition_dim,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        num_reward_components=num_reward_components,
        dropout_rate=0.1
    ).to(device)
    print("ProxyRewardModel initialized.")

    # --- Optimizer and Loss Function ---
    optimizer = optim.Adam(proxy_model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # --- Start Training ---
    print("Starting proxy model training with PyG data...")
    proxy_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, pyg_batch_data in enumerate(pyg_dataloader):
            pyg_batch_data = pyg_batch_data.to(device)

            # True rewards are on pyg_batch_data.true_rewards
            # It should be (batch_size, num_reward_components) after PyG batching
            true_rewards_b = pyg_batch_data.true_rewards
            if true_rewards_b.ndim == 3 and true_rewards_b.shape[1] == 1: # Squeeze if necessary
                true_rewards_b = true_rewards_b.squeeze(1)


            optimizer.zero_grad()
            # Proxy model now expects a PyG Batch object
            pred_rewards_b = proxy_model(pyg_batch_data)

            loss = criterion(pred_rewards_b, true_rewards_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(pyg_dataloader)}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / len(pyg_dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed. Average Proxy Training Loss: {avg_epoch_loss:.4f}")

    proxy_model.eval()
    print("Proxy model training finished.")

    # --- Save Model (Example) ---
    # torch.save(proxy_model.state_dict(), "trained_proxy_reward_model_pyg.pth")
    # print("Trained proxy model weights (conceptually) saved to trained_proxy_reward_model_pyg.pth")

if __name__ == "__main__":
    main_train_proxy()
```
