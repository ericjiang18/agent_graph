import torch
import os
import sys
import numpy as np # For example evaluation

# Add GDesigner root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gdesigner_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if gdesigner_root not in sys.path:
    sys.path.insert(0, gdesigner_root)

from GDesigner.gtd.gtd_framework import GTDFramework
from GDesigner.gtd.proxy_reward_model import ProxyRewardModel # Needed to load/init proxy
from GDesigner.gtd.metrics_and_datasets import (
    calculate_communication_cost,
    calculate_sparsity,
    # The following are placeholders and would need real implementations:
    # get_task_utility,
    # get_vulnerability_score,
    # calculate_macp_reward
)

def print_warning_message(message):
    import sys
    print(f"WARNING: {message}", file=sys.stderr)

# Dummy MAS execution functions (replace with actual logic for real evaluation)
def dummy_mas_utility_func(adj, cond, feat):
    # Higher is better. Let's make it slightly sensitive to number of edges.
    # More edges might mean more collaboration, up to a point.
    num_edges = adj.sum()
    density = num_edges / (adj.shape[0] * (adj.shape[0]-1) + 1e-6)
    # Simulate some utility based on density, e.g., peaks at moderate density
    utility = (-8 * (density - 0.4)**2 + 0.8).clamp(0,1) # peaks at density 0.4
    return utility.item() * torch.rand(1).item() * 0.3 + 0.5 # Add noise and base

def dummy_mas_vulnerability_func(adj, cond, feat):
    # Higher is worse (e.g. more vulnerable). Let's make it slightly sensitive to density.
    # Very sparse or very dense might be bad.
    density = calculate_sparsity(adj.unsqueeze(0)).item() # calculate_sparsity returns density here
    vulnerability = (density * (1-density)) * 4 # peaks at 0.5
    return vulnerability * torch.rand(1).item() * 0.2 + 0.1


def main_run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    # Framework parameters (should match trained models)
    num_nodes = 10
    node_feature_dim = 32
    condition_dim = 64
    time_embed_dim = 128
    gt_num_layers = 3
    gt_num_heads = 4
    diffusion_num_timesteps = 100 # Should match trained diffusion model

    # Proxy model parameters (if using guidance)
    proxy_node_feat_dim = node_feature_dim
    proxy_cond_dim = condition_dim
    proxy_gnn_hidden_dim = 64
    proxy_gnn_layers = 2
    proxy_mlp_hidden_dim = 128
    reward_component_names = ['utility', 'cost', 'vulnerability']
    proxy_num_rewards = len(reward_component_names)

    macp_weights = {'utility': 1.0, 'cost': -0.3, 'vulnerability': -0.2} # Cost/Vuln should be negative

    # Experiment parameters
    num_graphs_to_generate = 5 # Generate a few graphs for a sample task

    # --- Load Pre-trained Models (Conceptual) ---
    # In a real scenario, you would load weights from .pth files

    # Initialize GTD Framework (this will init diffusion model architecture)
    gtd_system = GTDFramework(
        node_feature_dim=node_feature_dim,
        condition_dim=condition_dim,
        time_embed_dim=time_embed_dim,
        gt_num_layers=gt_num_layers,
        gt_num_heads=gt_num_heads,
        diffusion_num_timesteps=diffusion_num_timesteps,
        # Proxy will be loaded/set separately if used
        device=device
    )
    print("GTDFramework initialized for experiment.")

    # Conceptually load diffusion model weights
    # gtd_system.diffusion_model.load_state_dict(torch.load("trained_diffusion_model.pth", map_location=device))
    # For this script, the diffusion model is freshly initialized (random weights).
    print_warning_message("Diffusion model is using fresh (random) weights. Load pre-trained weights for meaningful results.")
    gtd_system.diffusion_model.eval()

    # Initialize and load Proxy Model (if using guidance)
    proxy_model_for_guidance = ProxyRewardModel(
        node_feature_dim=proxy_node_feat_dim,
        condition_dim=proxy_cond_dim,
        gnn_hidden_dim=proxy_gnn_hidden_dim,
        gnn_layers=proxy_gnn_layers,
        mlp_hidden_dim=proxy_mlp_hidden_dim,
        num_reward_components=proxy_num_rewards
    ).to(device)
    # proxy_model_for_guidance.load_state_dict(torch.load("trained_proxy_reward_model.pth", map_location=device))
    print_warning_message("ProxyRewardModel is using fresh (random) weights. Load pre-trained weights for meaningful guidance.")
    proxy_model_for_guidance.eval()

    # Set up the guider in the GTD framework
    gtd_system.guider = gtd_system.guider = GTDFramework( # Re-init with proxy
         node_feature_dim=node_feature_dim,
         condition_dim=condition_dim,
         time_embed_dim=time_embed_dim,
         gt_num_layers=gt_num_layers,
         gt_num_heads=gt_num_heads,
         diffusion_num_timesteps=diffusion_num_timesteps,
         proxy_reward_model=proxy_model_for_guidance,
         macp_weights=macp_weights,
         num_candidates_per_step=10, # K for ZO
         device=device
    ).guider # Assign the guider from a new temp instance

    if gtd_system.guider:
        print("Guider has been set up in the GTD Framework.")
    else:
        print_warning_message("Failed to set up guider. Check proxy/weights.")


    # --- Prepare Input Data for Generation ---
    # This would typically come from a test dataset
    # (node_features for agents, task_condition for the specific task)
    print(f"\nPreparing dummy input data for generating {num_graphs_to_generate} graphs...")
    input_node_features = torch.randn(num_graphs_to_generate, num_nodes, node_feature_dim).to(device)
    input_task_condition = torch.randn(num_graphs_to_generate, condition_dim).to(device)
    print("Input data prepared.")

    # --- Run Generation ---
    # 1. Unguided Generation
    print("\n--- Running Unguided Generation ---")
    generated_A0_probs_unguided = gtd_system.generate_graphs(
        num_graphs=num_graphs_to_generate,
        num_nodes=num_nodes,
        node_features=input_node_features,
        task_condition=input_task_condition,
        use_guidance=False
    )
    # Binarize the generated probability matrices
    generated_adj_unguided = (generated_A0_probs_unguided > 0.5).float()
    print(f"Generated {generated_adj_unguided.shape[0]} unguided graphs.")

    # 2. Guided Generation
    print("\n--- Running Guided Generation ---")
    if gtd_system.guider is None:
        print_warning_message("Guider not available, skipping guided generation.")
        generated_adj_guided = torch.zeros_like(generated_adj_unguided) # Placeholder
    else:
        generated_A0_probs_guided = gtd_system.generate_graphs(
            num_graphs=num_graphs_to_generate,
            num_nodes=num_nodes,
            node_features=input_node_features,
            task_condition=input_task_condition,
            use_guidance=True
        )
        generated_adj_guided = (generated_A0_probs_guided > 0.5).float()
        print(f"Generated {generated_adj_guided.shape[0]} guided graphs.")

    # --- Evaluate Generated Graphs (Conceptual) ---
    # This section uses placeholder evaluation functions.
    print("\n--- Evaluating Generated Graphs (Conceptual) ---")

    results = {"unguided": [], "guided": []}

    for i in range(num_graphs_to_generate):
        adj_u = generated_adj_unguided[i]
        adj_g = generated_adj_guided[i]
        cond = input_task_condition[i] # Condition for this specific graph
        n_feat = input_node_features[i] # Node features for this specific graph

        eval_u = {}
        eval_u['cost'] = calculate_communication_cost(adj_u).item()
        eval_u['density'] = calculate_sparsity(adj_u).item()
        # For utility and vulnerability, we need to pass single graph data
        eval_u['utility'] = dummy_mas_utility_func(adj_u, cond, n_feat)
        eval_u['vulnerability'] = dummy_mas_vulnerability_func(adj_u, cond, n_feat)
        eval_u['macp_score'] = sum(w * eval_u[k] for k, w in macp_weights.items() if k in eval_u)
        results["unguided"].append(eval_u)

        eval_g = {}
        eval_g['cost'] = calculate_communication_cost(adj_g).item()
        eval_g['density'] = calculate_sparsity(adj_g).item()
        eval_g['utility'] = dummy_mas_utility_func(adj_g, cond, n_feat)
        eval_g['vulnerability'] = dummy_mas_vulnerability_func(adj_g, cond, n_feat)
        eval_g['macp_score'] = sum(w * eval_g[k] for k, w in macp_weights.items() if k in eval_g)
        results["guided"].append(eval_g)

    print("\nEvaluation Results (Example):")
    for i in range(num_graphs_to_generate):
        print(f"\nGraph {i+1}:")
        print(f"  Unguided: Cost={results['unguided'][i]['cost']:.1f}, Density={results['unguided'][i]['density']:.3f}, "
              f"Utility={results['unguided'][i]['utility']:.3f}, Vuln={results['unguided'][i]['vulnerability']:.3f}, "
              f"MACP={results['unguided'][i]['macp_score']:.3f}")
        print(f"  Guided:   Cost={results['guided'][i]['cost']:.1f}, Density={results['guided'][i]['density']:.3f}, "
              f"Utility={results['guided'][i]['utility']:.3f}, Vuln={results['guided'][i]['vulnerability']:.3f}, "
              f"MACP={results['guided'][i]['macp_score']:.3f}")

    # Aggregate results (example: average MACP score)
    avg_macp_unguided = np.mean([r['macp_score'] for r in results['unguided']])
    avg_macp_guided = np.mean([r['macp_score'] for r in results['guided']])
    print(f"\nAverage MACP Score (Unguided): {avg_macp_unguided:.3f}")
    print(f"Average MACP Score (Guided):   {avg_macp_guided:.3f}")

    print_warning_message("Evaluation uses DUMMY utility and vulnerability functions. Replace with real MAS execution for meaningful evaluation.")
    print("\nExperiment script finished.")

if __name__ == "__main__":
    main_run_experiment()
```
