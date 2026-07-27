import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    """
    Class comment
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """

        super().__init__()

        self.w = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
          Apply the linear transformation to the input.
        """

        y = einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")

        return y

