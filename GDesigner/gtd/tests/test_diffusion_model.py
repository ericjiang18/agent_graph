import unittest
import torch
import os
import sys

# Add GDesigner root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
gdesigner_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if gdesigner_root not in sys.path:
    sys.path.insert(0, gdesigner_root)

from GDesigner.gtd.graph_transformer import GraphTransformer
from GDesigner.gtd.diffusion_model import ConditionalDiscreteGraphDiffusion, get_timestep_embedding

class TestConditionalDiscreteGraphDiffusion(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters
        self.batch_size = 2
        self.num_nodes = 4
        self.node_feature_dim = 8
        self.condition_dim = 12 # For task/agent condition
        self.time_embed_dim = 16  # For time embedding input to GT
        self.gt_layers = 1
        self.gt_heads = 1

        self.num_timesteps = 10 # Small number for quick tests

        # Denoising network (GraphTransformer)
        self.denoising_net = GraphTransformer(
            node_feature_dim=self.node_feature_dim,
            condition_dim=self.condition_dim, # GT takes task_condition_dim
            time_embed_dim=self.time_embed_dim, # GT takes time_embed_dim
            num_layers=self.gt_layers,
            num_heads=self.gt_heads,
            output_dim=1
        ).to(self.device)

        # Diffusion Model
        self.diffusion_model = ConditionalDiscreteGraphDiffusion(
            denoising_network=self.denoising_net,
            num_timesteps=self.num_timesteps,
            device=self.device
        ).to(self.device)

        # Dummy inputs
        self.A0_binary = (torch.rand(self.batch_size, self.num_nodes, self.num_nodes) > 0.5).float().to(self.device)
        self.node_features = torch.randn(self.batch_size, self.num_nodes, self.node_feature_dim).to(self.device)
        self.task_condition = torch.randn(self.batch_size, self.condition_dim).to(self.device)
        self.t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device).long()

    def test_get_timestep_embedding(self):
        embedding_dim = 20
        timesteps = torch.tensor([0, 1, 5, self.num_timesteps - 1], device=self.device).long()
        time_embedding = get_timestep_embedding(timesteps, embedding_dim)
        self.assertEqual(time_embedding.shape, (timesteps.shape[0], embedding_dim))

    def test_q_sample(self):
        # Test forward noising process q(At | A0)
        At_prob = self.diffusion_model.q_sample(self.A0_binary, self.t)

        self.assertEqual(At_prob.shape, self.A0_binary.shape)
        self.assertTrue(torch.all(At_prob >= 0) and torch.all(At_prob <= 1))

        # For t=0, At should be very close to A0 (minimal noise)
        t_zero = torch.zeros_like(self.t)
        At_zero_noise_prob = self.diffusion_model.q_sample(self.A0_binary, t_zero)
        # With A0_scaled = 2*A0-1, noise = 0, noisy_A_scaled = sqrt_alpha_bar[0]*A0_scaled
        # sqrt_alpha_bar[0] is sqrt(1-beta[0]), close to 1.
        # So At_zero_noise_prob should be very close to A0_binary.float()
        self.assertTrue(torch.allclose(At_zero_noise_prob, self.A0_binary.float(), atol=0.1))
        # atol might need adjustment based on beta_start

        # For t=T-1 (large t), At should be close to random noise (probabilities around 0.5)
        t_max = torch.full_like(self.t, self.num_timesteps - 1)
        At_max_noise_prob = self.diffusion_model.q_sample(self.A0_binary, t_max)
        # sqrt_alphas_cumprod_t will be small, sqrt_one_minus_alphas_cumprod_t will be large
        # So At_scaled will be dominated by noise. At_prob should be around 0.5.
        # This is harder to check precisely without knowing noise values.
        # Check if values are not all 0 or 1.
        self.assertTrue(torch.any(At_max_noise_prob > 0.01) and torch.any(At_max_noise_prob < 0.99))


    def test_predict_A0_from_At(self):
        # Test the denoising network's prediction of A0 from At
        At_prob = self.diffusion_model.q_sample(self.A0_binary, self.t) # Get some noisy At

        A0_pred_prob = self.diffusion_model.predict_A0_from_At(
            At_prob, self.t, self.node_features, self.task_condition
        )
        self.assertEqual(A0_pred_prob.shape, self.A0_binary.shape)
        self.assertTrue(torch.all(A0_pred_prob >= 0) and torch.all(A0_pred_prob <= 1))

    def test_q_posterior_mean_variance(self):
        At_prob = self.diffusion_model.q_sample(self.A0_binary, self.t)
        A0_pred_prob = self.diffusion_model.predict_A0_from_At(At_prob, self.t, self.node_features, self.task_condition)

        mean_prob, log_var = self.diffusion_model.q_posterior_mean_variance(A0_pred_prob, At_prob, self.t)

        self.assertEqual(mean_prob.shape, self.A0_binary.shape)
        self.assertEqual(log_var.shape, self.A0_binary.shape) # log_var is broadcasted from (B,1,1) to (B,N,N) effectively
        self.assertTrue(torch.all(mean_prob >= 0) and torch.all(mean_prob <= 1))


    def test_p_sample(self):
        # Test one step of the reverse process p(A_{t-1} | At)
        At_prob = torch.rand_like(self.A0_binary).to(self.device) # Start with random At probabilities

        A_prev_prob = self.diffusion_model.p_sample(
            At_prob, self.t, self.node_features, self.task_condition
        )
        self.assertEqual(A_prev_prob.shape, self.A0_binary.shape)
        self.assertTrue(torch.all(A_prev_prob >= 0) and torch.all(A_prev_prob <= 1))

        # Test with t=0 (should involve less/no noise in p_sample)
        t_zero = torch.zeros_like(self.t)
        A_prev_at_t_zero = self.diffusion_model.p_sample(
             At_prob, t_zero, self.node_features, self.task_condition
        )
        # When t=0, nonzero_mask is 0, so no noise is added from posterior_log_variance.
        # The sample should be equal to the posterior_mean_prob.
        A0_pred_prob_t0 = self.diffusion_model.predict_A0_from_At(At_prob, t_zero, self.node_features, self.task_condition)
        mean_prob_t0, _ = self.diffusion_model.q_posterior_mean_variance(A0_pred_prob_t0, At_prob, t_zero)
        self.assertTrue(torch.allclose(A_prev_at_t_zero, mean_prob_t0, atol=1e-6))


    def test_sample_loop(self):
        # Test the full sampling loop (generation)
        generated_A0_probs = self.diffusion_model.sample(
            num_nodes=self.num_nodes,
            batch_size=self.batch_size,
            node_features=self.node_features,
            task_condition=self.task_condition
        )
        self.assertEqual(generated_A0_probs.shape, (self.batch_size, self.num_nodes, self.num_nodes))
        self.assertTrue(torch.all(generated_A0_probs >= 0) and torch.all(generated_A0_probs <= 1))

    def test_training_forward_pass(self):
        # Test the forward pass used during training (calculates loss)
        loss = self.diffusion_model(self.A0_binary, self.node_features, self.task_condition)
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.ndim == 0) # Scalar loss
        self.assertTrue(loss.item() >= 0) # Loss should be non-negative

    def test_p_sample_with_guidance(self):
        At_prob = torch.rand_like(self.A0_binary).to(self.device)
        # Dummy guided A0 (binary, as if from guider)
        guided_A0_override_binary = (torch.rand_like(self.A0_binary) > 0.5).float().to(self.device)

        A_prev_prob_guided = self.diffusion_model.p_sample(
            At_prob, self.t, self.node_features, self.task_condition,
            guided_A0_override=guided_A0_override_binary
        )
        self.assertEqual(A_prev_prob_guided.shape, self.A0_binary.shape)
        self.assertTrue(torch.all(A_prev_prob_guided >= 0) and torch.all(A_prev_prob_guided <= 1))

        # Check if the guided_A0_override was used:
        # The posterior mean should be based on guided_A0_override_binary
        mean_prob_if_guided, _ = self.diffusion_model.q_posterior_mean_variance(
            guided_A0_override_binary.float(), # Treat binary as {0,1} probs
            At_prob,
            self.t
        )
        # If t > 0, noise is added. If t = 0, A_prev_prob_guided should be mean_prob_if_guided.
        t_for_check = self.t.clone()
        # To make it more deterministic, let's test for t=0 where no sampling noise is added in p_sample
        t_zero = torch.zeros_like(self.t)
        A_prev_guided_t0 = self.diffusion_model.p_sample(
            At_prob, t_zero, self.node_features, self.task_condition,
            guided_A0_override=guided_A0_override_binary
        )
        mean_prob_if_guided_t0, _ = self.diffusion_model.q_posterior_mean_variance(
            guided_A0_override_binary.float(), At_prob, t_zero
        )
        self.assertTrue(torch.allclose(A_prev_guided_t0, mean_prob_if_guided_t0, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
```
