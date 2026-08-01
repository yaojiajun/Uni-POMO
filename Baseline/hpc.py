"""
Purity order matrix computation, used by the AM decoder's purity-based
logit modulation (see rl4co/models/zoo/am/decoder.py).

Geometry structure:
         z1 ●
            /   \
           /     \
          /  上   \
         /  半圆   \
        /    H_R    \
       /             \
   i ●---------------● j  ← 直径
       \             /
        \   下      /
         \  半圆   /
          \  H_L  /
           \     /
            \   /
             z2 ●

- Split circle into upper/lower halves by the diameter line i-j
- K_p(i,j) = min(|H_R|, |H_L|) if both halves non-empty, else 0
"""

import torch
from torch import Tensor
from rl4co.utils.ops import batchify



def compute_purity_order_matrix(locs: Tensor) -> Tensor:
    """Compute purity order K_p for all pairs of nodes.

    Geometric definition:
    C(e_ij) = {v != i,j : (v_i - v).(v_j - v) < 0}  (in-circle test, diameter e_ij)
    H_R (upper) / H_L (lower) = the two halves of C(e_ij) split by diameter line i-j
    K_p(i,j) = 0 if either half is empty, else min(|H_R|, |H_L|)

    Args:
        locs: [B, N, 2] node coordinates in [0, 1]^2

    Returns:
        kp: [B, N, N] float purity order matrix
    """
    B, N, _ = locs.shape
    device = locs.device

    # Direction vector v_ij = v_j - v_i for all pairs (i,j)
    v_ij = locs.unsqueeze(1) - locs.unsqueeze(2)  # (B, N, N, 2)

    # Perpendicular vector to v_ij (rotate 90 degrees): (-dy, dx)
    # This represents the "up" direction relative to the diameter line i-j
    perp_ij = torch.stack([-v_ij[..., 1], v_ij[..., 0]], dim=-1)  # (B, N, N, 2)

    countL = torch.zeros(B, N, N, device=device, dtype=torch.float32)
    countR = torch.zeros(B, N, N, device=device, dtype=torch.float32)

    for z in range(N):
        z_xy = locs[:, z, :]  # (B, 2)

        # Compute diff for all nodes: v - z
        diff_all = z_xy[:, None, :] - locs  # (B, N, 2)

        # In-circle test: (v_i - z).(v_j - z) < 0
        dot = (diff_all.unsqueeze(2) * diff_all.unsqueeze(1)).sum(-1)  # (B, N, N)
        in_circle = dot < 0

        # Exclude i=z and j=z
        in_circle[:, z, :] = False
        in_circle[:, :, z] = False

        # Determine which side of the diameter line: (z - i) · perp_ij
        z_from_i = z_xy[:, None, None, :] - locs.unsqueeze(2)  # (B, N, N, 2) where dim2 is i
        side = (z_from_i * perp_ij).sum(-1)  # (B, N, N)

        # Count upper (side > 0) and lower (side <= 0) halves
        countR += (in_circle & (side > 0)).float()   # H_R: upper semicircle
        countL += (in_circle & (side <= 0)).float()  # H_L: lower semicircle

    # K_p = min(countL, countR) if both > 0, else 0
    valid = (countL > 0) & (countR > 0)
    kp = torch.where(valid, torch.minimum(countL, countR), torch.zeros_like(countL))

    # Zero out diagonal
    diag_mask = torch.eye(N, dtype=torch.bool, device=device)
    kp = kp.masked_fill(diag_mask.unsqueeze(0), 0.0)

    return kp  # [B, N, N]



def compute_purity_order_matrix_vectorized(locs: Tensor) -> Tensor:
    """Vectorized version of purity order computation for efficiency.

    This is equivalent to compute_purity_order_matrix but uses vectorized operations.
    """
    B, N, _ = locs.shape
    device = locs.device

    # Expand dimensions for broadcasting
    # v_i: (B, N, 1, 2), v_j: (B, 1, N, 2), v_k: (B, 1, 1, N, 2)
    v_i = locs.unsqueeze(2).unsqueeze(3)  # (B, N, 1, 1, 2)
    v_j = locs.unsqueeze(1).unsqueeze(3)  # (B, 1, N, 1, 2)
    v_k = locs.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N, 2)

    # Compute (v_j - v_k)^T · (v_i - v_k) for all combinations
    diff_j = v_j - v_k  # (B, 1, N, N, 2)
    diff_i = v_i - v_k  # (B, N, 1, N, 2)

    # Dot product: (B, N, N, N) where [b, i, j, k] = (v_j - v_k)^T · (v_i - v_k)
    dot_product = (diff_j * diff_i).sum(dim=-1)  # (B, N, N, N)

    # Check if each vertex k is in the covering set of edge (i, j)
    in_circle = (dot_product < 0).float()  # (B, N, N, N)

    # Mask out self-loops: k should not be i or j
    mask = torch.ones(N, N, N, device=device, dtype=torch.bool)
    for i in range(N):
        mask[i, :, i] = False  # k != i
        mask[i, i, :] = False  # j != i (no self-loops)
    for j in range(N):
        mask[:, j, j] = False  # k != j

    in_circle = in_circle * mask.unsqueeze(0).float()

    # Count vertices in covering set for each edge
    kp = in_circle.sum(dim=-1)  # (B, N, N)

    return kp


def compute_purity_weights(
    locs: Tensor,
    actions: Tensor,
    n_start: int = 1,
    gamma: float = 0.9,
) -> Tensor:
    """Compute PUPO purity weights W_t following NeurIPS 2025 paper.

    Definition 5.4 (Purity Weightings): The purity weighting W(U_t, τ_{t+1}) is:

        W(U_t, τ_{t+1}) = 1 + Σ_{j=t}^∞ δ^{j-t} C(U_j, τ_{j+1})

    where C(U_t, τ_{t+1}) is the purity cost (Def. 5.3):

        C(U_t, τ_{t+1}) = K_p(τ_t, τ_{t+1}) + φ(U_{t+1}) - φ(U_t)

    and φ(U_t) is the purity availability (Def. 5.1):

        φ(U_t) = Σ_{x_i ∈ U_t} min_{x_j ∈ U_t, j≠i} K_p(x_i, x_j) / |U_t|

    Args:
        locs: [B, N, 2] node coordinates (node 0 is depot)
        actions: [B*n_start, seq_len] chosen node indices per step
        n_start: number of POMO starts
        gamma: discount factor δ for future purity costs

    Returns:
        weights: [B*n_start, seq_len] normalized purity weights W_t
    """
    device = locs.device
    B_total, seq_len = actions.shape
    N = locs.shape[1]

    # Compute purity order matrix K_p
    kp = compute_purity_order_matrix_vectorized(locs)  # [B, N, N]

    if n_start and n_start > 1:
        kp = batchify(kp, n_start)  # [B*n_start, N, N] start-major layout

    # Reshape to [B, P, T]
    B = B_total // n_start if n_start > 1 else B_total
    P = n_start if n_start > 1 else 1
    solutions = actions.reshape(B, P, seq_len)

    N_cust = N - 1  # number of customers (exclude depot)

    # Extract customer-only K_p matrix (exclude depot row 0 and column 0)
    kp_cust = kp[:, 1:, 1:].clone()  # [B_total, N_cust, N_cust]
    kp_cust = kp_cust.reshape(B, P, N_cust, N_cust)  # [B, P, N_cust, N_cust]

    def compute_phi(unvisited):
        """Compute purity availability φ(U) = mean of min K_p per unvisited node.

        φ(U) = (1/|U|) * Σ_{x_i ∈ U} min_{x_j ∈ U, j≠i} K_p(x_i, x_j)
        """
        n_unvis = unvisited.sum(dim=2).float()  # [B, P]
        has_pair = (n_unvis > 1)

        if not has_pair.any():
            return torch.zeros(B, P, device=device)

        # Set large value for visited nodes to exclude them from min
        INF = 1e6
        kp_masked = kp_cust.clone()
        kp_masked = kp_masked + (~unvisited).unsqueeze(3).float() * INF  # mask columns

        # Mask diagonal
        eye_mask = torch.eye(N_cust, dtype=torch.bool, device=device)
        kp_masked = kp_masked.masked_fill(eye_mask[None, None], INF)

        # Compute min K_p for each unvisited node
        min_per_node, _ = kp_masked.min(dim=3)  # [B, P, N_cust]
        min_per_node = min_per_node.clamp(max=INF * 0.9)
        min_per_node[min_per_node >= INF * 0.9] = 0.0

        # Average over unvisited nodes
        phi = (min_per_node * unvisited.float()).sum(dim=2) / n_unvis.clamp(min=1)
        phi = phi * has_pair.float()

        return phi

    # Track unvisited customers
    unvisited = torch.ones(B, P, N_cust, dtype=torch.bool, device=device)
    phi_vals = torch.zeros(B, P, seq_len + 1, device=device)
    phi_vals[:, :, 0] = compute_phi(unvisited)
    kp_edge = torch.zeros(B, P, seq_len, device=device)

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, P)
    pomo_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
    prev_nodes = torch.zeros(B, P, dtype=torch.long, device=device)

    # Reconstruct full kp for edge lookups
    kp_full = kp.reshape(B, P, N, N)

    for t in range(seq_len):
        cur_nodes = solutions[:, :, t]

        # Get K_p for edge (prev_node, cur_node)
        kp_val = kp_full[batch_idx, pomo_idx, prev_nodes, cur_nodes]
        # Only count edges between customers (not depot)
        kp_val = kp_val * (prev_nodes > 0).float() * (cur_nodes > 0).float()
        kp_edge[:, :, t] = kp_val

        # Mark customer as visited
        is_cust = cur_nodes > 0
        cust_idx = (cur_nodes - 1).clamp(min=0)
        unvis_flat = unvisited.reshape(B * P, N_cust)
        bp = torch.arange(B * P, device=device)
        is_cust_flat = is_cust.reshape(B * P)
        cust_flat = cust_idx.reshape(B * P)
        unvis_flat[bp[is_cust_flat], cust_flat[is_cust_flat]] = False
        unvisited = unvis_flat.reshape(B, P, N_cust)

        phi_vals[:, :, t + 1] = compute_phi(unvisited)
        prev_nodes = cur_nodes

    # Compute purity cost: C_t = K_p(edge) + φ_{t+1} - φ_t
    C = kp_edge + phi_vals[:, :, 1:] - phi_vals[:, :, :seq_len]  # [B, P, T]

    # Compute purity weights: W_t = 1 + Σ_{j≥t} γ^{j-t} C_j
    W = torch.zeros(B, P, seq_len, device=device)
    running = torch.zeros(B, P, device=device)
    for t in range(seq_len - 1, -1, -1):
        running = C[:, :, t] + gamma * running
        W[:, :, t] = 1.0 + running

    # Normalize weights: W / mean(W) over time dimension
    W_mean = W.mean(dim=2, keepdim=True).clamp(min=1e-8)
    W = W / W_mean

    # Reshape back to [B*P, T]
    weights = W.reshape(B_total, seq_len)
    return weights
