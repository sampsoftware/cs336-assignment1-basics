import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    """
    Class comment
    """

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Args:
            in_features: int  final dimension of the input
            out_features: int  final dimension of the output
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters

        Returns:
            Linear(Module) object with initialized weights tensor
        """

        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        nn.init.trunc_normal_(self.wweight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
          Apply the linear transformation to the input.
        """

        y = einsum(x, self.w, "... d_in, d_out d_in -> ... d_out")

        return y


class Embedding(nn.Module):
    """
    Class comment
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Args:
            num_embeddings: int  Size of the vocabulary
            embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters

        Returns:
            Embeddings(Module) with initialized embeddings tensor
        """

        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor: 
        """
        Lookup the embedding vectors for the given token IDs
        """

        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Args:
            d_model: int  Hidden dimension of the model
            eps: float = 1e-5  Epsilon value for numerical stability
            device: torch.device | None = None  Device to store the parameters on
            dtype: torch.dtype | None = None  Data type of the parameters
        
        Returns:
            RMSNorm(Module) with initialized tensor
        """

        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
            Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
            Note: Remember to upcast your input to torch.float32 before performing the normalization (and later downcast 
            to the original dtype)
        """

        in_dtype = x.dtype           # Save the incoming data type
        x = x.to(torch.float32)     # Upscale x to float32
        mean_squares = x.pow(2).mean(dim=-1, keepdim=True) # Get the mean of squares for the activation vector
        add_eps = mean_squares + self.eps   # add eps to the rms of each activation vector
        rms = add_eps.sqrt()                # get the square root of every activation vector
        normalized = x / rms        # Divide each activation scalar by it's vector's rms
        scaled = normalized * self.weight
        out = scaled.to(in_dtype)          # Scale back to whatever was sent in
        return out






