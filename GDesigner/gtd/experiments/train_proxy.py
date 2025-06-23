import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import os
import sys

# Add GDesigner root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gdesigner_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if gdesigner_root not in sys.path:
    sys.path.insert(0, gdesigner_root)

from GDesigner.gtd.proxy_reward_model import ProxyRewardModel
from GDesigner.gtd.metrics_and_datasets import create_proxy_training_data
# Using placeholder from metrics_and_datasets for data generation

def main_train_proxy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # Proxy Model parameters
    num_nodes = 10  # Max number of nodes in graphs
    node_feature_dim = 32 # Must match features used by proxy model
    condition_dim = 64    # Must match condition info used by proxy
    gnn_hidden_dim = 64
    gnn_layers = 2
    mlp_hidden_dim = 128
    # Define the order and number of reward components the proxy model will predict
    reward_component_names = ['utility', 'cost', 'vulnerability']
    num_reward_components = len(reward_component_names)

    # Training parameters
    num_train_samples = 512 # Number of dummy training samples for proxy model
    batch_size = 64
    epochs = 15 # Example epochs
    learning_rate = 1e-3

    # --- Dataset Preparation (Placeholder) ---
    # In a real scenario, load pre-generated
    # (task_cond_embed, node_feat_init, A_adj_matrix, true_reward_components_vector) tuples
    print("Generating dummy proxy model training data (using placeholder)...")

    # Dummy raw data and embedding functions for the placeholder
    dummy_raw_data_proxy = [{'task_text': f'task_{i}', 'agent_config': {'type': 'default'}} for i in range(num_train_samples)]

    def _dummy_embedding_func_proxy(task_text, agent_config):
        cond_embed = torch.randn(condition_dim)
        node_feats = torch.randn(num_nodes, node_feature_dim)
        return cond_embed, node_feats

    def _dummy_mas_reward_runner(raw_item, adj_matrix):
        # Returns a dict of reward components
        # Ensure this dict matches `reward_component_names` in order and content
        return {
            'utility': torch.rand(1).item(),
            'cost': adj_matrix.sum().item() / (num_nodes * (num_nodes -1) + 1e-5), # Normalized cost
            'vulnerability': torch.rand(1).item() * 0.5 # Lower vulnerability is better
        }

    def _dummy_topology_sampler(max_n):
        return (torch.rand(max_n, max_n) > (0.3 + torch.rand(1).item()*0.4) ).float() # Sample diverse densities

    # The placeholder `create_proxy_training_data` returns a list of tuples.
    # Each tuple: (task_cond_embed, node_feat_init, A_adj_matrix, true_rewards_vector)
    # For demo, generating data directly into tensors for TensorDataset.

    all_cond_embeds_proxy = []
    all_node_feats_proxy = []
    all_adj_matrices_proxy = []
    all_true_rewards_proxy = []

    for i in range(num_train_samples):
        raw_item = dummy_raw_data_proxy[i]
        cond_embed, node_feats = _dummy_embedding_func_proxy(raw_item['task_text'], raw_item['agent_config'])

        adj_matrix = _dummy_topology_sampler(num_nodes)
        rewards_dict = _dummy_mas_reward_runner(raw_item, adj_matrix)

        # Ensure rewards_vector is in the order defined by reward_component_names
        rewards_vector = torch.tensor([rewards_dict[name] for name in reward_component_names], dtype=torch.float32)

        all_cond_embeds_proxy.append(cond_embed)
        all_node_feats_proxy.append(node_feats)
        all_adj_matrices_proxy.append(adj_matrix)
        all_true_rewards_proxy.append(rewards_vector)

    stacked_cond_embeds = torch.stack(all_cond_embeds_proxy)
    stacked_node_feats = torch.stack(all_node_feats_proxy)
    stacked_adj_matrices = torch.stack(all_adj_matrices_proxy)
    stacked_true_rewards = torch.stack(all_true_rewards_proxy)

    print(f"Proxy Data Shapes: Cond: {stacked_cond_embeds.shape}, NodeFeats: {stacked_node_feats.shape}, Adj: {stacked_adj_matrices.shape}, Rewards: {stacked_true_rewards.shape}")

    # Order for TensorDataset: node_features, adj_matrix, condition, true_rewards
    # This order matches the example training loop in proxy_reward_model.py
    dataset = TensorDataset(stacked_node_feats, stacked_adj_matrices, stacked_cond_embeds, stacked_true_rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Created DataLoader for proxy training with {len(dataloader)} batches.")

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
    criterion = nn.MSELoss() # Mean Squared Error for reward regression

    # --- Start Training ---
    print("Starting proxy model training...")
    proxy_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (node_feat_b, adj_mat_b, cond_b, true_rewards_b) in enumerate(dataloader):
            node_feat_b = node_feat_b.to(device)
            adj_mat_b = adj_mat_b.to(device)
            cond_b = cond_b.to(device)
            true_rewards_b = true_rewards_b.to(device)

            optimizer.zero_grad()
            pred_rewards_b = proxy_model(node_feat_b, adj_mat_b, cond_b)
            loss = criterion(pred_rewards_b, true_rewards_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 20 == 0: # Log every 20 batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} completed. Average Proxy Training Loss: {avg_epoch_loss:.4f}")

    proxy_model.eval()
    print("Proxy model training finished.")

    # --- Save Model (Example) ---
    # torch.save(proxy_model.state_dict(), "trained_proxy_reward_model.pth")
    # print("Trained proxy model weights (conceptually) saved to trained_proxy_reward_model.pth")

if __name__ == "__main__":
    main_train_proxy()
```
