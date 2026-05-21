import torch
import numpy as np

#=========================================================================================
#This code was inspired by Santiago Acevedo's work: https://github.com/acevedo-s/syn-sem
#=========================================================================================

def clip_hidden_torch(hidden, alphamin=0.05, alphamax=0.95):
    """
    Clip hidden states using PyTorch.

    Supports:
      - 4D: (B, C, H, W)
      - 3D: (L, T, E)
      - 2D: (B, E)
      - 1D: (E,)
    Returns: numpy.ndarray of dtype uint16 with bfloat16 bit patterns.
    """
    # Convert to torch tensor if needed
    if isinstance(hidden, np.ndarray):
        hidden = torch.from_numpy(hidden)

    hidden_float = hidden.float()
    ndim = hidden_float.dim()

    if ndim == 4:
        # (B, C, H, W): clip per across spatial dims, preserve channels
        B, C, H, W = hidden_float.shape
        hidden_flat = hidden_float.view(B, C, H * W)  # (B, C, HW)
        qmin = hidden_flat.quantile(alphamin, dim=2, keepdim=True)  # (B, C, 1)
        qmax = hidden_flat.quantile(alphamax, dim=2, keepdim=True)  # (B, C, 1)
        hidden_clipped = hidden_flat.clamp(min=qmin, max=qmax).view(B, C, H, W)

    elif ndim == 3:
        # (L, T, E)
        L, T, E = hidden_float.shape
        hidden_flat = hidden_float.view(L, T * E)

        qmin = hidden_flat.quantile(alphamin, dim=1, keepdim=True)
        qmax = hidden_flat.quantile(alphamax, dim=1, keepdim=True)

        hidden_clipped = hidden_flat.clamp(min=qmin, max=qmax).view(L, T, E)

    elif ndim == 2:
        # (B, E)
        B, E = hidden_float.shape
        hidden_flat = hidden_float  # (B, E)

        qmin = hidden_flat.quantile(alphamin, dim=1, keepdim=True)
        qmax = hidden_flat.quantile(alphamax, dim=1, keepdim=True)

        hidden_clipped = hidden_flat.clamp(min=qmin, max=qmax)  # (B, E)

    elif ndim == 1:
        # (E,)
        E = hidden_float.shape[0]
        hidden_flat = hidden_float.view(1, E)  # (1, E)

        qmin = hidden_flat.quantile(alphamin, dim=1, keepdim=True)  # (1, 1)
        qmax = hidden_flat.quantile(alphamax, dim=1, keepdim=True)  # (1, 1)

        hidden_clipped = hidden_flat.clamp(min=qmin, max=qmax).view(E)  # (E,)

    else:
        raise ValueError(f"Expected hidden to have 1, 2 or 3 dims, got shape {hidden.shape}")

    return hidden_clipped.cpu().numpy()
