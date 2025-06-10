import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) module.

    This module applies RMSNorm to the input tensor. It can operate in several modes:
    1. Standard RMSNorm over the last dimension.
    2. Standard RMSNorm with learnable gain parameters.
    3. Head-wise RMSNorm (for 3D+ inputs like multi-head attention outputs),
       where normalization is performed per head.
    4. Head-wise RMSNorm with learnable gain parameters (gain applied per head, per feature in head).
    """
    def __init__(self,
                 dim: int,
                 use_params: bool = True,
                 normalize_per_head: bool = False,
                 num_heads: int = None,
                 eps: float = 1e-5):
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): The dimension of the input tensor over which normalization is applied
                       (if not `normalize_per_head`) or the total dimension that will be
                       split into heads (if `normalize_per_head`).
            use_params (bool, optional): If True, learnable gain parameters are used.
                                         Defaults to True.
            normalize_per_head (bool, optional): If True, performs head-wise normalization.
                                                 Input `x` is expected to be 3D or higher
                                                 (e.g., batch, seq_len, dim).
                                                 `dim` will be reshaped into `num_heads * head_dim`.
                                                 Defaults to False.
            num_heads (int, optional): The number of heads if `normalize_per_head` is True.
                                       Must be specified if `normalize_per_head` is True.
                                       Defaults to None.
            eps (float, optional): A small value added to the denominator for numerical stability.
                                   Defaults to 1e-5.
        """
        super().__init__()
        self.dim = dim
        self.use_params = use_params
        self.normalize_per_head = normalize_per_head
        self.num_heads = num_heads
        self.eps = eps

        if self.normalize_per_head:
            if self.num_heads is None:
                raise ValueError("num_heads must be specified if normalize_per_head is True.")
            if self.dim % self.num_heads != 0:
                raise ValueError(f"dim ({self.dim}) must be divisible by num_heads ({self.num_heads}).")

        if self.use_params:
            # Gain parameter is always of shape (dim,)
            # For head-wise, this means it's (num_heads * head_dim).
            # It will be reshaped to (num_heads, head_dim) and then broadcast in forward if needed.
            self.g = nn.Parameter(torch.ones(dim))
        else:
            # If not using params, register 'g' as None.
            self.register_parameter('g', None)

    def _rms(self, x_for_rms: torch.Tensor, norm_dim: int) -> torch.Tensor:
        """
        Helper function to compute RMS value for normalization (actually 1/RMS).
        Args:
            x_for_rms (torch.Tensor): Tensor to compute RMS for.
            norm_dim (int): Dimension along which to compute mean square.
        Returns:
            torch.Tensor: 1 / RMS values (rsqrt of mean square + eps).
        """
        # Using x_for_rms.float() for stability before pow(2) and mean,
        # especially if input x might be float16.
        # rsqrt is 1 / sqrt(value).
        return (torch.mean(x_for_rms.float().pow(2), dim=norm_dim, keepdim=True) + self.eps).rsqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor. Expected to have its last dimension equal to self.dim.
                              Typically (batch_size, seq_len, dim) for sequence data or (batch_size, dim).

        Returns:
            torch.Tensor: Normalized (and optionally scaled) tensor, with the same shape as input x.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f"Input tensor's last dimension ({x.shape[-1]}) "
                             f"does not match module's dim ({self.dim}).")

        original_dtype = x.dtype # Remember original dtype

        if self.normalize_per_head:
            if x.ndim < 3: # e.g. (batch_size, seq_len, dim)
                 raise ValueError(f"Input tensor must be at least 3D for head-wise normalization, but got {x.ndim}D.")

            batch_dims = x.shape[:-2] # Dims before seq_len and dim, e.g. (batch_size,)
            seq_len = x.shape[-2] # seq_len dimension

            head_dim = self.dim // self.num_heads

            # Reshape for head-wise normalization: (*batch_dims, seq_len, num_heads, head_dim)
            x_reshaped_for_heads = x.reshape(*batch_dims, seq_len, self.num_heads, head_dim)

            rms_values = self._rms(x_reshaped_for_heads, norm_dim=-1) # Normalize over head_dim
            normalized_x_reshaped = x_reshaped_for_heads * rms_values.type(original_dtype) # Cast rms_values back

            # Reshape back to original input's structure but with self.dim at the end
            # This is equivalent to x.shape
            normalized_x = normalized_x_reshaped.reshape(*batch_dims, seq_len, self.dim)
        else:
            # Standard RMSNorm: Normalize over the last dimension (self.dim)
            # This handles inputs of various ndims (e.g., 2D, 3D, etc.)
            # as long as the last dimension is self.dim.
            rms_values = self._rms(x, norm_dim=-1) # Normalize over the last dimension
            normalized_x = x * rms_values.type(original_dtype) # Cast rms_values back

        if self.use_params and self.g is not None:
            # self.g is of shape (dim,).
            # If normalize_per_head, g is (total_dim) which is (num_heads * head_dim).
            # We want to apply g per feature, so for head-wise it should be reshaped to
            # effectively apply specific gains to specific heads' features.
            # The current self.g definition (torch.ones(dim)) is fine.
            # If normalize_per_head, g is (dim,). When multiplied with normalized_x (which is also (..., dim)),
            # it applies the gains correctly across the full dimension.
            # If one wanted per-head-per-feature distinct gains that are not tied across the original 'dim',
            # then g would need to be (num_heads, head_dim) and reshaped to (1,1,num_heads,head_dim) then applied to x_reshaped_for_heads.
            # However, the current setup with g=(dim,) means that if you split dim into heads,
            # the gains are applied as if it's still one flat dimension. This is consistent with common RMSNorm/LayerNorm usage.
            normalized_x = normalized_x * self.g

        return normalized_x.type(original_dtype) # Ensure final output matches original input dtype

if __name__ == "__main__":
    # Common parameters
    batch_size = 2
    seq_len = 3
    dim = 12 # Example dimension
    num_h = 4 # Example number of heads, so head_dim = 3
    if dim % num_h != 0:
        raise ValueError("Example dim must be divisible by num_h for headwise examples")

    print(f"--- RMSNorm PyTorch Module Examples ---")
    print(f"Input tensor shape: ({batch_size}, {seq_len}, {dim})")
    print(f"Number of heads for head-wise examples: {num_h}\n")

    # Create a sample input tensor
    sample_x = torch.randn(batch_size, seq_len, dim)
    print("Sample input tensor x (first batch, first token):\n", sample_x[0, 0, :])

    # 1. Standard RMSNorm without parameters
    print("\n--- 1. Standard RMSNorm (use_params=False, normalize_per_head=False) ---")
    rms_norm_std_no_params = RMSNorm(dim, use_params=False, normalize_per_head=False)
    output_std_no_params = rms_norm_std_no_params(sample_x.clone()) # Use clone for fresh input
    print("Output (first batch, first token):\n", output_std_no_params[0, 0, :])
    print("Output shape:", output_std_no_params.shape)
    if hasattr(rms_norm_std_no_params, 'g') and rms_norm_std_no_params.g is not None:
        print("Gain parameters exist: True (This is unexpected for use_params=False)")
    else:
        print("Gain parameters exist: False")


    # 2. Standard RMSNorm with parameters
    print("\n--- 2. Standard RMSNorm (use_params=True, normalize_per_head=False) ---")
    rms_norm_std_with_params = RMSNorm(dim, use_params=True, normalize_per_head=False)
    output_std_with_params = rms_norm_std_with_params(sample_x.clone())
    print("Output (first batch, first token):\n", output_std_with_params[0, 0, :])
    print("Output shape:", output_std_with_params.shape)
    if hasattr(rms_norm_std_with_params, 'g') and rms_norm_std_with_params.g is not None:
        print("Gain parameters exist: True")
        print("Gain g (first few values):", rms_norm_std_with_params.g[:5].data)
    else:
        print("Gain parameters exist: False (This is unexpected for use_params=True)")


    # 3. Head-wise RMSNorm without parameters
    print("\n--- 3. Head-wise RMSNorm (use_params=False, normalize_per_head=True) ---")
    rms_norm_headwise_no_params = RMSNorm(dim, use_params=False, normalize_per_head=True, num_heads=num_h)
    output_headwise_no_params = rms_norm_headwise_no_params(sample_x.clone())
    print("Output (first batch, first token):\n", output_headwise_no_params[0, 0, :])
    print("Output shape:", output_headwise_no_params.shape)
    if hasattr(rms_norm_headwise_no_params, 'g') and rms_norm_headwise_no_params.g is not None:
        print("Gain parameters exist: True (This is unexpected for use_params=False)")
    else:
        print("Gain parameters exist: False")
    # Verification: check variance per head for the first sample (optional)
    # output_reshaped = output_headwise_no_params[0].reshape(seq_len, num_h, dim // num_h)
    # print("Variance per head (sample 0, token 0):", torch.var(output_reshaped[0], dim=-1))


    # 4. Head-wise RMSNorm with parameters
    print("\n--- 4. Head-wise RMSNorm (use_params=True, normalize_per_head=True) ---")
    rms_norm_headwise_with_params = RMSNorm(dim, use_params=True, normalize_per_head=True, num_heads=num_h)
    output_headwise_with_params = rms_norm_headwise_with_params(sample_x.clone())
    print("Output (first batch, first token):\n", output_headwise_with_params[0, 0, :])
    print("Output shape:", output_headwise_with_params.shape)
    if hasattr(rms_norm_headwise_with_params, 'g') and rms_norm_headwise_with_params.g is not None:
        print("Gain parameters exist: True")
        # Gain g is (dim,), applied after reshaping back from heads
        print("Gain g (first few values):", rms_norm_headwise_with_params.g[:5].data)
    else:
        print("Gain parameters exist: False (This is unexpected for use_params=True)")

    # Example with a different input tensor shape (e.g., 2D) for standard normalization
    print("\n--- Example with 2D input for Standard RMSNorm ---")
    sample_x_2d = torch.randn(batch_size, dim) # Shape (batch_size, dim)
    print("Sample 2D input tensor x_2d (first sample):\n", sample_x_2d[0, :])
    rms_norm_std_2d = RMSNorm(dim, use_params=True, normalize_per_head=False)
    output_std_2d = rms_norm_std_2d(sample_x_2d.clone())
    print("Output 2D (first sample):\n", output_std_2d[0, :])
    print("Output 2D shape:", output_std_2d.shape)

    # Example demonstrating error for head-wise with 2D input
    print("\n--- Example: Error for Head-wise RMSNorm with 2D input ---")
    try:
        rms_norm_headwise_2d_error = RMSNorm(dim, use_params=False, normalize_per_head=True, num_heads=num_h)
        rms_norm_headwise_2d_error(sample_x_2d.clone())
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Example demonstrating error if num_heads is not provided for head-wise
    print("\n--- Example: Error if num_heads not provided for head-wise ---")
    try:
        rms_norm_no_num_heads = RMSNorm(dim, normalize_per_head=True) # num_heads is None by default
        rms_norm_no_num_heads(sample_x.clone())
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Example demonstrating error if dim is not divisible by num_heads for head-wise
    print("\n--- Example: Error if dim not divisible by num_heads for head-wise ---")
    try:
        rms_norm_bad_dim_div = RMSNorm(dim + 1, normalize_per_head=True, num_heads=num_h)
        # Create a tensor with dim+1 for this specific test
        sample_x_bad_dim = torch.randn(batch_size, seq_len, dim + 1)
        rms_norm_bad_dim_div(sample_x_bad_dim)
    except ValueError as e:
        print(f"Caught expected error: {e}")
