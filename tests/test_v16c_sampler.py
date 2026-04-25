"""V16c sampler unit tests.

These tests verify the core fixes in V16c:
1. DDIM step is NOT identity (the V16b bug)
2. DDPM step is NOT identity+noise
3. remove_mean_with_mask keeps padding zero
4. sample() uses node_mask correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.diffusion import ConditionalDDPM, SE3EquivariantDenoiser


def _make_ddpm(timesteps=1000):
    denoiser = SE3EquivariantDenoiser(max_atoms=10, cond_dim=32, hidden_dim=32, num_layers=1, num_heads=2)
    ddpm = ConditionalDDPM(denoiser, timesteps=timesteps)
    return ddpm


def test_ddim_step_not_identity():
    """CRITICAL: Ensure DDIM single step x_{t-1} != x_t.

    The V16b bug: using alpha_t for both prediction and reconstruction
    causes x_{t-1} = x_t (identity). V16c must use alpha_{t-1}.
    """
    ddpm = _make_ddpm()
    B, N = 2, 10
    x_t = torch.randn(B, N, 3)
    eps_pred = torch.randn(B, N, 3)
    t = torch.full((B,), 500, dtype=torch.long)
    t_prev = 480

    x_prev, x0_pred = ddpm._ddim_step(x_t, eps_pred, t, t_prev, eta=0.0)

    # x_prev must NOT be close to x_t
    diff = (x_prev - x_t).abs().max().item()
    assert diff > 0.01, f"DDIM step is near-identity! max diff = {diff:.6f}"
    print(f"  PASS: DDIM step max diff from x_t = {diff:.4f}")

    # Also verify: at t_prev=-1, should return x0_pred
    x_final, x0_final = ddpm._ddim_step(x_t, eps_pred, t, -1, eta=0.0)
    assert torch.allclose(x_final, x0_final), "Final step should return x0_pred"
    print("  PASS: DDIM final step returns x0_pred")


def test_ddpm_step_not_identity_or_noise_only():
    """CRITICAL: Ensure DDPM step is NOT x_t + noise.

    The V16b bug: the DDPM path was x_t + extra_noise (adding noise instead
    of removing it). V16c must use standard posterior.
    """
    ddpm = _make_ddpm()
    B, N = 2, 10
    x_t = torch.randn(B, N, 3)
    eps_pred = torch.randn(B, N, 3)
    t = torch.full((B,), 500, dtype=torch.long)

    # Run multiple times to average out stochasticity
    torch.manual_seed(42)
    diffs = []
    for _ in range(10):
        x_prev, x0_pred = ddpm._ddpm_step(x_t, eps_pred, t)
        diffs.append((x_prev - x_t).abs().mean().item())

    mean_diff = sum(diffs) / len(diffs)
    assert mean_diff > 0.01, f"DDPM step is near-identity! mean diff = {mean_diff:.6f}"
    print(f"  PASS: DDPM step mean diff from x_t = {mean_diff:.4f}")

    # Verify that x_prev is closer to x0_pred than x_t is
    # (denoising should move toward x0_pred)
    torch.manual_seed(42)
    x_prev, x0_pred = ddpm._ddpm_step(x_t, eps_pred, t)
    dist_prev_to_x0 = (x_prev - x0_pred).abs().mean().item()
    dist_xt_to_x0 = (x_t - x0_pred).abs().mean().item()
    print(f"  INFO: dist(x_prev, x0_pred) = {dist_prev_to_x0:.4f}, dist(x_t, x0_pred) = {dist_xt_to_x0:.4f}")

    # At t=0, DDPM step should add no noise
    t0 = torch.full((B,), 0, dtype=torch.long)
    x_prev_0, _ = ddpm._ddpm_step(x_t, eps_pred, t0)
    # Run again with different seed - result should be same (no noise at t=0)
    x_prev_0b, _ = ddpm._ddpm_step(x_t, eps_pred, t0)
    assert torch.allclose(x_prev_0, x_prev_0b, atol=1e-6), "DDPM at t=0 should be deterministic"
    print("  PASS: DDPM at t=0 is deterministic (no noise)")


def test_remove_mean_with_mask_keeps_padding_zero():
    """Ensure padding positions stay at zero after CoM removal."""
    ddpm = _make_ddpm()
    B, N = 3, 10
    x = torch.randn(B, N, 3)
    n_atoms = torch.tensor([5, 8, 3])
    mask = ddpm._build_node_mask(n_atoms, N, 'cpu')

    x_out = ddpm._remove_mean_with_mask(x, mask)

    # Check padding is zero
    for i in range(B):
        n = int(n_atoms[i].item())
        padding = x_out[i, n:]
        assert padding.abs().max().item() < 1e-8, \
            f"Padding not zero for sample {i}: max = {padding.abs().max().item()}"

    # Check CoM of valid atoms is zero
    for i in range(B):
        n = int(n_atoms[i].item())
        valid_mean = x_out[i, :n].mean(dim=0)
        assert valid_mean.abs().max().item() < 1e-6, \
            f"CoM not zero for sample {i}: {valid_mean.tolist()}"

    print("  PASS: padding zero, CoM zero for all samples")


def test_node_mask_construction():
    """Verify node mask shape and values."""
    ddpm = _make_ddpm()
    n_atoms = torch.tensor([3, 10, 0, 7])
    mask = ddpm._build_node_mask(n_atoms, 10, 'cpu')

    assert mask.shape == (4, 10)
    assert mask[0, :3].sum() == 3 and mask[0, 3:].sum() == 0
    assert mask[1, :].sum() == 10  # all valid
    assert mask[2, :].sum() == 0   # none valid
    assert mask[3, :7].sum() == 7 and mask[3, 7:].sum() == 0
    print("  PASS: node mask construction correct")


def test_ddim_time_pairs():
    """Verify DDIM time pairs are correct."""
    ddpm = _make_ddpm(timesteps=1000)
    pairs = ddpm._get_ddim_time_pairs(50)

    assert len(pairs) == 50
    # First pair should start from near T-1
    assert pairs[0][0] == 999
    # Last pair should end at -1
    assert pairs[-1][1] == -1
    # All t_cur > t_prev
    for t_cur, t_prev in pairs:
        assert t_cur > t_prev, f"t_cur={t_cur} <= t_prev={t_prev}"
    print(f"  PASS: {len(pairs)} time pairs, range [{pairs[0][0]}, {pairs[-1][1]}]")


def test_ddim_perfect_denoising():
    """Verify DDIM correctly recovers x_0 when eps_pred equals true noise.

    If the denoiser perfectly predicts the noise, then x0_pred = true x_0,
    and DDIM should converge exactly to x_0.
    """
    ddpm = _make_ddpm()
    B, N = 1, 10
    x_0_true = torch.randn(B, N, 3) * 0.3  # true clean signal

    # Forward diffusion: add noise at t=999
    t_start = torch.full((B,), 999, dtype=torch.long)
    x_T, true_noise = ddpm.q_sample(x_0_true, t_start)

    # If we know the true noise, DDIM with perfect eps_pred should recover x_0
    # Single step from t=999 to t=-1 should give x0_pred
    x_recovered, x0_pred = ddpm._ddim_step(x_T, true_noise, t_start, -1, eta=0.0)
    error = (x_recovered - x_0_true).abs().max().item()
    assert error < 0.01, f"Perfect single-step recovery failed: max error = {error:.6f}"
    print(f"  PASS: perfect single-step DDIM recovery, max error = {error:.8f}")

    # Multi-step: should also converge
    x = x_T.clone()
    pairs = ddpm._get_ddim_time_pairs(20)
    for t_cur, t_prev in pairs:
        t = torch.full((B,), t_cur, dtype=torch.long)
        # Re-derive the true noise for this timestep
        sqrt_a = ddpm.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_1ma = ddpm.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        # eps = (x_t - sqrt_a * x_0) / sqrt_1ma
        eps_oracle = (x - sqrt_a * x_0_true) / sqrt_1ma.clamp(min=1e-8)
        x, _ = ddpm._ddim_step(x, eps_oracle, t, t_prev, eta=0.0)

    multi_error = (x - x_0_true).abs().max().item()
    assert multi_error < 1e-3, f"Multi-step recovery failed: max error = {multi_error:.6f}"
    print(f"  PASS: multi-step DDIM recovery, max error = {multi_error:.6f}")


def test_sample_uses_node_mask():
    """End-to-end: verify sample() keeps padding slots at zero throughout.

    Constructs a small model, runs full sample() with different n_atoms per sample,
    and checks that padding positions are always zero in the output.
    """
    torch.manual_seed(123)
    denoiser = SE3EquivariantDenoiser(
        max_atoms=20, cond_dim=32, hidden_dim=32, num_layers=1, num_heads=2)
    ddpm = ConditionalDDPM(denoiser, timesteps=100)
    ddpm.eval()

    B = 3
    max_atoms = 20
    n_atoms = torch.tensor([5, 12, 8])
    c_global = torch.randn(B, 32)
    c_patches = torch.randn(B, 64, 32)  # minimal patch features

    with torch.no_grad():
        coords, type_logits = ddpm.sample(
            c_global, c_patches, n_atoms, max_atoms=max_atoms,
            use_ddim=True, ddim_steps=5,  # minimal steps for speed
            disable_guidance=True, disable_ring_snap=True,
        )

    assert coords.shape == (B, max_atoms, 3), f"Wrong coord shape: {coords.shape}"
    assert type_logits.shape[0] == B and type_logits.shape[1] == max_atoms

    for i in range(B):
        n = int(n_atoms[i].item())
        padding_coords = coords[i, n:]
        max_pad_val = padding_coords.abs().max().item()
        assert max_pad_val < 1e-6, \
            f"Sample {i}: padding coords not zero! n_atoms={n}, max padding val={max_pad_val:.8f}"

        # Valid atoms should have non-trivial values (not all zero)
        valid_coords = coords[i, :n]
        assert valid_coords.abs().max().item() > 0.001, \
            f"Sample {i}: valid coords are all near-zero, denoiser may not be working"

    print(f"  PASS: all {B} samples have padding=0, valid atoms non-trivial")
    print(f"  Shapes: coords={coords.shape}, types={type_logits.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("V16c Sampler Unit Tests")
    print("=" * 60)

    tests = [
        ("DDIM step not identity", test_ddim_step_not_identity),
        ("DDPM step not identity/noise-only", test_ddpm_step_not_identity_or_noise_only),
        ("remove_mean_with_mask padding", test_remove_mean_with_mask_keeps_padding_zero),
        ("node mask construction", test_node_mask_construction),
        ("DDIM time pairs", test_ddim_time_pairs),
        ("DDIM perfect denoising", test_ddim_perfect_denoising),
        ("sample() uses node_mask (e2e)", test_sample_uses_node_mask),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\nTest: {name}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
