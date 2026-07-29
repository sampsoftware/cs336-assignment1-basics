import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int
import math


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


class RotaryPositionalEmbedding(nn.Module):
    """
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """ 
        Construct the RoPE module and create buffers for sin and cos values. Create a position tensor rank == 1
        with initial values [0,1,...,len-1]. Then another tensor rank == 1 with values of the numerator of the
        RoPE formula. Use outer to create a rank == 2 tensor. So theta[x,y] position[x] for all x * inverse_frequencies[y]
        for all y. self.register_buffer uses the superclass's register function to make torch aware of the 
        tensor, and fill it with the cos or sin of the theta_ik.

        Args:
            theta: float  Θ value for the RoPE
            d_k: int  dimension of query and key vectors
            max_seq_len: int  Maximum sequence length that will be input
            device: torch.device | None = None  Device to store the buffer on

        Returns:
            None

        """
        super().__init__()

        position = torch.arange(max_seq_len)
        inverse_frequencies = 1 / (theta**(2*(torch.arange(d_k//2)/d_k)))
        theta_ik = torch.outer(position, inverse_frequencies)
        self.register_buffer("cos_table",torch.cos(theta_ik), persistent=False)
        self.register_buffer("sin_table",torch.sin(theta_ik), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note 
        that you should tolerate x with an arbitrary number of batch dimensions. You should assume 
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of 
        x along the sequence dimension. You should use the token positions to slice your (possibly precomputed) 
        cos and sin tensors along the sequence dimension.

        Args:
            x: Tensor to work on
            token_positions: The absolute position of the tokens represented in x -- they may not be simply
                1..n because the caller may be operating in the middle of a range or with scattered tokens for
                reasons of its own.

        Returns:
            A tensor like x but with the rotational position information encoded into it.

        """

        # Break the features dimension into two axes
        # Visualization: take the [batch seq feature] cube and make it into two cubes, cube 1 is 
        # [batch seq odd_indexed_features] and the second is [batch seq even_indexed_features].
        # An alternative vision that matches the code is d/2 new cubes each with even and odd indexed features.
        x_planes = rearrange(x,"... (a b) -> ... a b", b = 2)

        # Apply the rotation matrix R_ik = ([cos, -sin],[sin, cos])
        # new_k1 = old_k1*cos - old_k2*sin; new_k2 = old_k1*sin + old_k2*cos
        # rotated_plane_halves[0] has the new_k1s and rph[1] has new_k2s
        rotated_plane_halves = [
            x_planes[...,0] * self.cos_table[token_positions] - x_planes[...,1] * self.sin_table[token_positions],
            x_planes[...,0] * self.sin_table[token_positions] + x_planes[...,1] * self.cos_table[token_positions]
        ]

        # Merge the feature planes back into one axis. The first dimension of rotated_plane_halves goes to the list
        # created applying the rotation matrix 
        x_rotated_features = rearrange(rotated_plane_halves,"b ... a -> ... (a b)")

        return x_rotated_features
    


