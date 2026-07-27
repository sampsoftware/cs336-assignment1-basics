import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int


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

        nn.init.trunc_normal_(self.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
          Apply the linear transformation to the input.
        """

        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

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


class SwiGLU(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_ff: int
    ):
        """
        Given the weights of a SwiGLU network, return the output of your implementation with these weights.

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
            w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
            w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
            w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
            in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

        Returns:
            Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
        """

        super().__init__()

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN(𝑥) = SwiGLU(𝑥, 𝑊1, 𝑊2, 𝑊3) 
        = 𝑊2 * (
            SiLU(𝑊1 * 𝑥) 
            *
            (𝑊3 * 𝑥)
        )

        = 𝑊2(
            (W1 x) * sigmoid(𝑊1 𝑥) 
            *
            (𝑊3 𝑥)
        )

    
        """

        W1x = self.w1(x)
        sigmoid_W1x = torch.sigmoid(W1x)
        silu = W1x * sigmoid_W1x
        W3x = self.w3(x)

        return self.w2(silu * W3x)

