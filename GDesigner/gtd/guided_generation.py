import torch
import torch.nn.functional as F

from GDesigner.gtd.proxy_reward_model import ProxyRewardModel
# Assuming ConditionalDiscreteGraphDiffusion provides a way to get A0_pred from At
# and the main sampling loop is external or part of the diffusion model itself.
# This GuidedGeneration class will act as the 'guider' object passed to the diffusion model's sample method.

class GuidedGeneration:
    """
    Manages the MACP-guided generation process using Zeroth-Order Optimization.
    This class is designed to be used as a 'guider' by the diffusion model's sampling process.
    """
    def __init__(self,
                 proxy_reward_model: ProxyRewardModel,
                 macp_weights: dict, # e.g., {'utility': 1.0, 'cost': -0.1, 'robustness': 0.5}
                 num_candidates_per_step: int = 10, # K in "Best-of-N" or "ZO"
                 device='cpu'):
        self.proxy_reward_model = proxy_reward_model
        self.proxy_reward_model.eval() # Ensure proxy model is in evaluation mode
        self.macp_weights = macp_weights
        self.num_candidates_per_step = num_candidates_per_step
        self.device = device

    def _calculate_composite_macp_reward(self, predicted_rewards: torch.Tensor):
        """
        Calculates the composite MACP reward from predicted components.
        Args:
            predicted_rewards (torch.Tensor): (batch_size, num_reward_components)
                                              Order of components must match proxy model output.
        Returns:
            torch.Tensor: (batch_size,) composite MACP scores.
        """
        # Assuming proxy_model.num_reward_components aligns with macp_weights keys
        # This requires a fixed order of reward components from the proxy model.
        # For example, if proxy model outputs [utility, cost, robustness]
        # and macp_weights = {'utility': w_u, 'cost': w_c, 'robustness': w_r}
        # composite_reward = w_u * utility + w_c * cost + w_r * robustness

        # A more robust way would be if proxy_model could return a dict or if components are named.
        # For now, assume ordered components. Let's say: utility, cost, vulnerability
        # The proxy model's `num_reward_components` should match len(macp_weights).
        # We need a predefined order for the weights.

        # Let's assume macp_weights is a tensor of the same length as num_reward_components
        # Or, if macp_weights is a dict, ensure order. For simplicity, let's assume
        # proxy_model.num_reward_components corresponds to an ordered list of weights.
        # Example: if proxy output is [utility, cost_val, vulnerability_val]
        # and self.macp_weights_tensor = [w_u, w_c, w_v] (note: w_c is likely negative)

        if not hasattr(self, 'macp_weights_tensor'):
            # Create this tensor once, assuming a fixed order from the proxy model.
            # This is a simplification. Ideally, the proxy model or config would define this order.
            # Let's assume the order in the dict is the one we want, or we define it explicitly.
            # For this example, let's use a fixed order, e.g. from proxy_model.output_names if it had such attr.
            # For now, let's assume the dict keys are ordered:
            # This is not robust. It's better to have an explicit list of reward names.
            # For the example, let's say the proxy model is trained to output in order: utility, cost, robustness
            # And macp_weights provides these keys.

            # A simple fixed order for the example, assuming proxy model outputs them in this order:
            # This should align with how the ProxyRewardModel was trained and its output structure.
            # Let's assume ProxyRewardModel.num_reward_components = 3 and outputs are utility, cost, robustness.

            # If self.macp_weights is {'utility': w1, 'cost': w2, 'robustness': w3}
            # We need to ensure the order of multiplication.
            # Simplest: assume ProxyRewardModel has a known output order.
            # Let's say the ProxyRewardModel is trained to output:
            # component 0: utility (higher is better)
            # component 1: cost (lower is better, so weight should be negative)
            # component 2: vulnerability (lower is better, so weight should be negative)

            # Example structure for macp_weights:
            # self.macp_weights = {'utility': 1.0, 'cost': -0.2, 'vulnerability': -0.1}
            # And we need to map these to the columns of predicted_rewards.
            # This mapping should be defined clearly.

            # For the purpose of this class, let's assume macp_weights is already a tensor
            # or a list that matches the order of proxy_reward_model's output columns.
            # If it's a dict, we need a predefined order.

            # Let's assume the ProxyRewardModel has an attribute like `reward_component_names`
            # For now, we'll rely on the order in the macp_weights dict, Python 3.7+ preserves insertion order.
            # This is still not ideal. A list of weights would be more direct.

            weights_tensor = torch.tensor([self.macp_weights[key] for key in self.macp_weights.keys()],
                                          device=self.device, dtype=torch.float32)
            if weights_tensor.shape[0] != predicted_rewards.shape[1]:
                raise ValueError(f"Mismatch between number of MACP weights ({weights_tensor.shape[0]}) "
                                 f"and number of predicted reward components ({predicted_rewards.shape[1]})")
            self.macp_weights_tensor = weights_tensor.unsqueeze(0) # (1, num_reward_components)

        composite_reward = torch.sum(predicted_rewards * self.macp_weights_tensor, dim=1) # (batch_size,)
        return composite_reward


    def guide(self,
              current_At_prob: torch.Tensor, # Current noisy graph (probabilities)
              timestep: torch.Tensor,        # Current timestep t
              unguided_A0_prediction: torch.Tensor, # Model's prediction for A0 based on current_At_prob
              node_features: torch.Tensor,
              task_condition: torch.Tensor
             ):
        """
        Performs one step of guided sampling using ZO optimization with the proxy model.
        This method is called by the diffusion model's sampler at each step.

        Args:
            current_At_prob (torch.Tensor): Current noisy graph (batch_size, num_nodes, num_nodes)
            timestep (torch.Tensor): Current timestep t (batch_size,)
            unguided_A0_prediction (torch.Tensor): The diffusion model's raw prediction for A0_prob
                                                  (batch_size, num_nodes, num_nodes) based on current_At_prob.
            node_features (torch.Tensor): (batch_size, num_nodes, node_feature_dim)
            task_condition (torch.Tensor): (batch_size, condition_dim)

        Returns:
            torch.Tensor: The A0_best_candidate (probabilities) selected by the guider,
                          to be used by the diffusion model for the next reverse step.
                          Shape: (batch_size, num_nodes, num_nodes)
        """
        batch_size, num_nodes, _ = unguided_A0_prediction.shape

        # Store the best candidate A0 for each item in the batch
        A0_best_candidates_batch = torch.zeros_like(unguided_A0_prediction)

        # Process each item in the batch separately for candidate generation and evaluation
        for i in range(batch_size):
            # 1. Candidate Generation:
            # Generate K candidate A0 graphs based on the unguided_A0_prediction for this item.
            # These are perturbations or samples around the model's current prediction.
            # unguided_A0_pred_item is (num_nodes, num_nodes) probabilities
            unguided_A0_pred_item = unguided_A0_prediction[i].unsqueeze(0) # (1, N, N)

            # Generate K candidates. For probabilities, we can add small noise K times,
            # or sample K times if the prediction is a distribution.
            # Here, unguided_A0_prediction are probabilities. We can sample binary graphs from these.

            candidate_A0s_prob = [] # List to store K candidate A0s (probabilities)

            # Option 1: Perturb the probabilities slightly
            # for _ in range(self.num_candidates_per_step):
            #     noise = torch.randn_like(unguided_A0_pred_item) * 0.05 # Small noise
            #     perturbed_A0_prob = torch.clamp(unguided_A0_pred_item + noise, 0.0, 1.0)
            #     candidate_A0s_prob.append(perturbed_A0_prob)

            # Option 2: Sample K binary graphs from unguided_A0_pred_item probabilities
            # Then, these binary graphs are evaluated by the proxy model.
            # The "A0_best_candidate" returned should be probabilities if the diffusion model expects that.
            # If we sample binary graphs, the "best" one is binary. We might need to convert it back
            # to probabilities (e.g. one-hot like, or keep as binary if diffusion handles it).
            # The proposal says "generate K different candidate clean graphs ... through sampling".
            # This suggests sampling binary A0s.

            # Let's sample K binary graphs.
            candidate_A0s_binary = []
            for _ in range(self.num_candidates_per_step):
                # Sample a binary graph from the predicted probabilities
                # This represents a concrete graph structure.
                binary_A0_sample = torch.bernoulli(unguided_A0_pred_item).float() # (1, N, N)
                candidate_A0s_binary.append(binary_A0_sample)

            # If no candidates, use the original unguided prediction (as probabilities)
            if not candidate_A0s_binary:
                A0_best_candidates_batch[i] = unguided_A0_pred_item.squeeze(0)
                continue

            # Stack candidates for batch processing by proxy model
            # candidate_A0s_stacked_binary: (K, N, N)
            candidate_A0s_stacked_binary = torch.cat(candidate_A0s_binary, dim=0)

            # 2. Proxy Evaluation:
            # Evaluate these K candidates using the proxy reward model.
            # Proxy model needs node_features and task_condition, replicated K times.

            # node_features_item: (num_nodes, node_feature_dim)
            # task_condition_item: (condition_dim)
            node_features_item = node_features[i]
            task_condition_item = task_condition[i]

            # Replicate node_features and task_condition for the K candidates
            # node_features_proxy_input: (K, num_nodes, node_feature_dim)
            # task_condition_proxy_input: (K, condition_dim)
            node_features_proxy_input = node_features_item.unsqueeze(0).repeat(self.num_candidates_per_step, 1, 1)
            task_condition_proxy_input = task_condition_item.unsqueeze(0).repeat(self.num_candidates_per_step, 1)

            with torch.no_grad():
                # predicted_rewards_for_candidates: (K, num_reward_components)
                predicted_rewards_for_candidates = self.proxy_reward_model(
                    node_features=node_features_proxy_input,
                    adj_matrix=candidate_A0s_stacked_binary, # Proxy model takes binary adj
                    condition=task_condition_proxy_input
                )

            # 3. Selection (ZO Optimization part):
            # Calculate composite MACP reward for each candidate and select the best one.
            # composite_macp_scores: (K,)
            composite_macp_scores = self._calculate_composite_macp_reward(predicted_rewards_for_candidates)

            best_candidate_idx = torch.argmax(composite_macp_scores)

            # The "A0_best_candidate" should be what the diffusion model's p_sample expects.
            # p_sample in ConditionalDiscreteGraphDiffusion expects A0_pred_prob (probabilities).
            # The candidates we evaluated were binary.
            # So, the "best candidate" here is a binary graph.
            # We need to decide how to represent this for the diffusion model.
            # Option A: Return the binary graph. p_sample needs to handle this. (current diffusion expects probs)
            # Option B: Return the unguided_A0_prediction that *led* to this best binary sample (if K=1, or if perturbations were on probs)
            # Option C: Use the best binary graph as a "target" to modify unguided_A0_prediction.
            # Option D: The proposal says "generate K different candidate clean graphs {A_0,k}"
            #           and then "select the best candidate graph A_0,best".
            #           "compute the distribution of the next noisy graph A_t-1 based on the chosen optimal 'target' A_0,best"
            # This implies that A_0,best (which is binary here) is directly used in the reverse diffusion step.
            # The `p_sample` method in `ConditionalDiscreteGraphDiffusion` takes `guided_A0_override`.
            # If this is binary, `q_posterior_mean_variance` needs to handle it.
            # Currently, `A0_pred_prob` is expected. So, if we pass a binary matrix, it becomes {0,1} probs.

            # Let's assume the `guided_A0_override` in `p_sample` can be this selected binary graph.
            # The formulas in `q_posterior_mean_variance` take `A0_pred_prob`, scale to -1,1.
            # So, a binary {0,1} matrix will be scaled to {-1,1} which is fine.
            A0_best_candidate_for_item = candidate_A0s_stacked_binary[best_candidate_idx] # (N, N), binary

            A0_best_candidates_batch[i] = A0_best_candidate_for_item

            # Logging for debugging (optional)
            # print(f"  Guider (item {i}): Best candidate from {self.num_candidates_per_step} chosen with score {composite_macp_scores.max().item():.3f}")
            # print(f"    Unguided A0 prob sum: {unguided_A0_pred_item.sum().item():.1f}")
            # print(f"    Best binary A0 edge sum: {A0_best_candidate_for_item.sum().item():.1f}")


        return A0_best_candidates_batch # (batch_size, num_nodes, num_nodes), binary {0,1}

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters from previous examples
    num_nodes_example = 5
    node_feat_dim = 16
    cond_dim = 32

    # Proxy Model (dummy, not trained here, just for structure)
    proxy_node_feat_dim = node_feat_dim
    proxy_cond_dim = cond_dim
    proxy_gnn_hidden = 32
    proxy_gnn_layers = 2
    proxy_mlp_hidden = 64
    proxy_num_rewards = 3 # utility, cost, robustness

    dummy_proxy_model = ProxyRewardModel(
        node_feature_dim=proxy_node_feat_dim,
        condition_dim=proxy_cond_dim,
        gnn_hidden_dim=proxy_gnn_hidden,
        gnn_layers=proxy_gnn_layers,
        mlp_hidden_dim=proxy_mlp_hidden,
        num_reward_components=proxy_num_rewards
    ).to(device).eval()

    # MACP weights (example: utility is good, cost and vulnerability are bad)
    # Order: utility, cost, vulnerability (must match proxy model output order)
    macp_w = {'utility': 1.0, 'cost': -0.5, 'vulnerability': -0.2} # Example weights

    # GuidedGeneration instance
    guider = GuidedGeneration(
        proxy_reward_model=dummy_proxy_model,
        macp_weights=macp_w,
        num_candidates_per_step=5, # K=5
        device=device
    )

    # Dummy inputs for the guider's `guide` method
    batch_s = 2
    dummy_current_At_prob = torch.rand(batch_s, num_nodes_example, num_nodes_example, device=device)
    dummy_timestep = torch.randint(0, 100, (batch_s,), device=device) # Example timesteps
    # unguided_A0_prediction from the diffusion model (probabilities)
    dummy_unguided_A0_pred = torch.rand(batch_s, num_nodes_example, num_nodes_example, device=device)

    dummy_node_features = torch.randn(batch_s, num_nodes_example, node_feat_dim, device=device)
    dummy_task_condition = torch.randn(batch_s, cond_dim, device=device)

    print("Testing GuidedGeneration's `guide` method...")
    A0_best_from_guider = guider.guide(
        current_At_prob=dummy_current_At_prob,
        timestep=dummy_timestep,
        unguided_A0_prediction=dummy_unguided_A0_pred,
        node_features=dummy_node_features,
        task_condition=dummy_task_condition
    )

    print(f"Shape of A0_best_from_guider: {A0_best_from_guider.shape}") # Expected: (batch_s, num_nodes, num_nodes)
    print("Content of the first guided A0 (should be binary {0,1}):")
    print(A0_best_from_guider[0])
    # Verify it's binary
    is_binary = torch.all((A0_best_from_guider == 0) | (A0_best_from_guider == 1))
    print(f"Is the output binary? {is_binary.item()}")

    # Test _calculate_composite_macp_reward
    print("\nTesting _calculate_composite_macp_reward...")
    # Dummy predicted rewards: (batch_size, num_reward_components)
    # For batch_size=2, num_reward_components=3
    # Rewards: [utility, cost, vulnerability]
    # Item 1: [0.8 (high util), 10 (high cost), 0.1 (low vuln)]
    # Item 2: [0.5 (mid util),  2 (low cost), 0.6 (high vuln)]
    dummy_predicted_rewards_vec = torch.tensor([
        [0.8, 10.0, 0.1],
        [0.5, 2.0, 0.6]
    ], device=device)

    # MACP weights: utility=1.0, cost=-0.5, vulnerability=-0.2
    # Expected for item 1: 1.0*0.8 + (-0.5)*10.0 + (-0.2)*0.1 = 0.8 - 5.0 - 0.02 = -4.22
    # Expected for item 2: 1.0*0.5 + (-0.5)*2.0 + (-0.2)*0.6 = 0.5 - 1.0 - 0.12 = -0.62

    # Re-initialize guider to reset macp_weights_tensor if it was already set by a different shape
    guider_test_rewards = GuidedGeneration(dummy_proxy_model, macp_w, 5, device)
    composite_scores = guider_test_rewards._calculate_composite_macp_reward(dummy_predicted_rewards_vec)
    print(f"Calculated composite scores: {composite_scores}")
    # Check if matches expected
    expected_scores = torch.tensor([-4.22, -0.62], device=device)
    print(f"Expected composite scores: {expected_scores}")
    print(f"Match: {torch.allclose(composite_scores, expected_scores, atol=1e-5)}")

    # The design of `guide` method fits into the `ConditionalDiscreteGraphDiffusion.sample` loop,
    # where it can replace/guide the `A0_pred_prob` used in `p_sample`.
    # The diffusion model's `p_sample` was modified to accept `guided_A0_override`.
    # If `guided_A0_override` is binary {0,1}, `q_posterior_mean_variance` treats it as probabilities {0.0, 1.0},
    # which then get scaled to {-1,1}. This should be compatible.
```
