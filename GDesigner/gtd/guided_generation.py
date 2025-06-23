import torch
import torch.nn.functional as F

from GDesigner.gtd.proxy_reward_model import ProxyRewardModel
from torch_geometric.data import Data, Batch
from torch_geometric.utils import from_dense


class GuidedGeneration:
    """
    Manages the MACP-guided generation process using Zeroth-Order Optimization.
    This class is designed to be used as a 'guider' by the diffusion model's sampling process.
    Uses PyG Batch for evaluating candidates with the ProxyRewardModel.
    """
    def __init__(self,
                 proxy_reward_model: ProxyRewardModel,
                 macp_weights: dict,
                 num_candidates_per_step: int = 10,
                 device='cpu'):
        self.proxy_reward_model = proxy_reward_model
        self.proxy_reward_model.eval()
        self.macp_weights = macp_weights
        self.num_candidates_per_step = num_candidates_per_step
        self.device = device

    def _calculate_composite_macp_reward(self, predicted_rewards: torch.Tensor):
        """
        Calculates the composite MACP reward from predicted components.
        Args:
            predicted_rewards (torch.Tensor): (batch_size, num_reward_components)
        Returns:
            torch.Tensor: (batch_size,) composite MACP scores.
        """
        if not hasattr(self, 'macp_weights_tensor_ordered'): # Changed attribute name for clarity
            # Ensure keys in macp_weights match the order of proxy_model's output components
            # This requires a defined convention for proxy_model's output columns.

            # Attempt to use 'reward_component_names' from proxy_model if it exists and is a list
            # This is a more robust way to ensure order.
            weights_list = []
            component_keys_source = "macp_weights.keys()" # Default source for keys

            if hasattr(self.proxy_reward_model, 'reward_component_names') and \
               isinstance(self.proxy_reward_model.reward_component_names, list):
                try:
                    weights_list = [self.macp_weights[key] for key in self.proxy_reward_model.reward_component_names]
                    component_keys_source = "proxy_model.reward_component_names"
                except KeyError as e:
                    raise ValueError(f"Key '{e}' from proxy_model.reward_component_names "
                                     "not found in provided macp_weights dictionary.")
            else: # Fallback to dict order if attribute not present or not a list
                 print("WARNING: ProxyRewardModel does not have 'reward_component_names' list attribute "
                       "or it's not a list. Relying on macp_weights dict order. "
                       "Ensure this matches proxy output column order.")
                 weights_list = [self.macp_weights[key] for key in self.macp_weights.keys()]

            weights_tensor = torch.tensor(weights_list, device=self.device, dtype=torch.float32)

            if self.proxy_reward_model.num_reward_components != predicted_rewards.shape[1]:
                 # This check is against the passed predicted_rewards, which should match proxy's output dim
                 pass # Already checked proxy_model.num_reward_components during its init.

            if weights_tensor.shape[0] != predicted_rewards.shape[1]:
                raise ValueError(
                    f"Mismatch between number of MACP weights ({weights_tensor.shape[0]}) derived from "
                    f"{component_keys_source} and number of "
                    f"predicted reward components from proxy model ({predicted_rewards.shape[1]}). Ensure consistency."
                )
            self.macp_weights_tensor_ordered = weights_tensor.unsqueeze(0) # Shape: (1, num_reward_components)

        composite_reward = torch.sum(predicted_rewards * self.macp_weights_tensor_ordered, dim=1)
        return composite_reward


    def guide(self,
              current_At_prob: torch.Tensor,
              timestep: torch.Tensor,
              unguided_A0_prediction: torch.Tensor,
              node_features: torch.Tensor, # (batch_size, num_nodes, node_feature_dim)
              task_condition: torch.Tensor  # (batch_size, condition_dim)
             ):
        """
        Performs one step of guided sampling.
        Args:
            node_features: Original node features for the batch of graphs being generated.
            task_condition: Original task condition for the batch.
        Returns:
            torch.Tensor: The A0_best_candidate (binary {0,1}) selected by the guider.
                          Shape: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = unguided_A0_prediction.shape
        A0_best_candidates_batch = torch.zeros_like(unguided_A0_prediction)

        for i in range(batch_size): # Process each graph in the generation batch individually
            unguided_A0_pred_item = unguided_A0_prediction[i] # (N, N) probabilities for current graph

            # 1. Candidate Generation: Sample K binary graphs from this item's probabilities
            candidate_A0s_binary_list = []
            for _ in range(self.num_candidates_per_step):
                # Bernoulli sampling from the (N,N) probability matrix
                binary_A0_sample = torch.bernoulli(unguided_A0_pred_item).float()
                candidate_A0s_binary_list.append(binary_A0_sample) # List of K tensors, each (N,N)

            if not candidate_A0s_binary_list:
                A0_best_candidates_batch[i] = unguided_A0_pred_item # Fallback if K=0 or error
                continue

            # 2. Proxy Evaluation using PyG Batch for these K candidates
            pyg_data_candidates = []
            # Node features for the current graph in the generation batch
            node_features_for_current_graph = node_features[i] # Shape: (N, node_feature_dim)
            # Task condition for the current graph (graph-level)
            # Needs to be (1, condition_dim) for each Data object, PyG Batch handles stacking later
            task_condition_for_current_graph = task_condition[i].unsqueeze(0) # Shape: (1, condition_dim)

            for binary_adj_candidate_matrix in candidate_A0s_binary_list: # binary_adj_candidate_matrix is (N,N)
                edge_index_candidate, _ = from_dense(binary_adj_candidate_matrix.to(self.device))

                # Each Data object gets node_features_for_current_graph and task_condition_for_current_graph
                data_candidate = Data(
                    x=node_features_for_current_graph.clone().to(self.device),
                    edge_index=edge_index_candidate, # Already on device
                    condition=task_condition_for_current_graph.clone().to(self.device) # Graph-level attr
                )
                pyg_data_candidates.append(data_candidate)

            # Create a PyG Batch from the list of K Data candidates
            # This batch will have K graphs in it.
            pyg_batch_for_proxy_eval = Batch.from_data_list(pyg_data_candidates).to(self.device)

            with torch.no_grad():
                # predicted_rewards_for_candidates will be (K, num_reward_components)
                predicted_rewards_for_candidates = self.proxy_reward_model(pyg_batch_for_proxy_eval)

            # 3. Selection (ZO Optimization part)
            # composite_macp_scores will be (K,)
            composite_macp_scores = self._calculate_composite_macp_reward(predicted_rewards_for_candidates)
            best_candidate_idx_in_k_batch = torch.argmax(composite_macp_scores)

            # The best candidate is one of the binary matrices from candidate_A0s_binary_list
            A0_best_candidate_for_item = candidate_A0s_binary_list[best_candidate_idx_in_k_batch] # (N, N), binary
            A0_best_candidates_batch[i] = A0_best_candidate_for_item # Store it for the i-th graph of original batch

        return A0_best_candidates_batch # (original_batch_size, N, N), binary {0,1}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes_example = 5
    node_feat_dim = 16
    cond_dim = 32 # This is graph-level condition for proxy

    proxy_node_feat_dim = node_feat_dim
    proxy_cond_dim = cond_dim
    proxy_gnn_hidden = 32
    proxy_gnn_layers = 1
    proxy_mlp_hidden = 20
    proxy_num_rewards = 3

    class DummyProxyWithNames(ProxyRewardModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reward_component_names = ['utility', 'cost', 'vulnerability']


    dummy_proxy_model_for_guide = DummyProxyWithNames(
        node_feature_dim=proxy_node_feat_dim,
        condition_dim=proxy_cond_dim,
        gnn_hidden_dim=proxy_gnn_hidden,
        gnn_layers=proxy_gnn_layers,
        mlp_hidden_dim=proxy_mlp_hidden,
        num_reward_components=proxy_num_rewards
    ).to(device).eval()

    macp_w = {'utility': 1.0, 'cost': -0.5, 'vulnerability': -0.2}

    guider = GuidedGeneration(
        proxy_reward_model=dummy_proxy_model_for_guide,
        macp_weights=macp_w,
        num_candidates_per_step=5, # K=5 candidates per graph in the batch
        device=device
    )

    # --- Test `guide` method ---
    # This is the batch size for the diffusion model's generation process
    generation_batch_size = 2

    # Inputs to guider.guide()
    dummy_current_At_prob = torch.rand(generation_batch_size, num_nodes_example, num_nodes_example, device=device)
    dummy_timestep = torch.randint(0, 100, (generation_batch_size,), device=device)
    dummy_unguided_A0_pred = torch.rand(generation_batch_size, num_nodes_example, num_nodes_example, device=device)

    # Node features and task conditions for the batch of graphs being generated by diffusion model
    dummy_node_features_for_diffusion_batch = torch.randn(generation_batch_size, num_nodes_example, node_feat_dim, device=device)
    dummy_task_condition_for_diffusion_batch = torch.randn(generation_batch_size, cond_dim, device=device)

    print("Testing GuidedGeneration's `guide` method (with PyG Batch for proxy)...")
    A0_best_from_guider = guider.guide(
        current_At_prob=dummy_current_At_prob,
        timestep=dummy_timestep,
        unguided_A0_prediction=dummy_unguided_A0_pred,
        node_features=dummy_node_features_for_diffusion_batch,
        task_condition=dummy_task_condition_for_diffusion_batch
    )

    print(f"Shape of A0_best_from_guider: {A0_best_from_guider.shape}")
    assert A0_best_from_guider.shape == (generation_batch_size, num_nodes_example, num_nodes_example)
    print("Content of the first guided A0 (binary {0,1}):")
    print(A0_best_from_guider[0])
    is_binary = torch.all((A0_best_from_guider == 0) | (A0_best_from_guider == 1))
    assert is_binary, "Output of guide method should be a binary matrix."
    print(f"Is the output binary? {is_binary.item()}")

    # --- Test _calculate_composite_macp_reward ---
    print("\nTesting _calculate_composite_macp_reward (with reward_component_names from proxy)...")
    # This test is for a batch of predicted rewards (e.g., K candidates for one graph, or B graphs)
    num_reward_vectors_to_test = 4
    dummy_predicted_rewards_vec = torch.rand(num_reward_vectors_to_test, proxy_num_rewards, device=device)

    # Force re-creation of macp_weights_tensor_ordered for this specific test call
    if hasattr(guider, 'macp_weights_tensor_ordered'):
        delattr(guider, 'macp_weights_tensor_ordered')

    composite_scores = guider._calculate_composite_macp_reward(dummy_predicted_rewards_vec)
    print(f"Calculated composite scores shape: {composite_scores.shape}") # Should be (num_reward_vectors_to_test,)
    assert composite_scores.shape == (num_reward_vectors_to_test,)
    print(f"Calculated composite scores example: {composite_scores[0].item()}")


    # Example with known values for _calculate_composite_macp_reward
    known_rewards = torch.tensor([[0.8, 0.2, 0.1], [0.3, 0.7, 0.6]], device=device)
    expected_scores = torch.tensor([0.68, -0.17], device=device) # From previous test calculation

    if hasattr(guider, 'macp_weights_tensor_ordered'):
        delattr(guider, 'macp_weights_tensor_ordered')
    calculated_known_scores = guider._calculate_composite_macp_reward(known_rewards)
    print(f"Calculated known scores: {calculated_known_scores}")
    print(f"Expected known scores: {expected_scores}")
    assert torch.allclose(calculated_known_scores, expected_scores, atol=1e-5)


    print("\nGuidedGeneration with PyG Batch for proxy evaluation test completed.")
```
