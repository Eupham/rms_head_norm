import numpy as np

def rms_norm_standard_no_params(x, eps=1e-5):
  """
  Performs Root Mean Square (RMS) normalization on the input array.

  Args:
    x: A multi-dimensional NumPy array. Normalization is done over the last dimension.
    eps: A small float value to prevent division by zero.

  Returns:
    A NumPy array with the same shape as x, but normalized.
  """
  # Calculate the square of the input x
  x_squared = np.square(x)

  # Calculate the mean of the squared values along the last dimension
  # Keepdims is important for broadcasting
  mean_squared = np.mean(x_squared, axis=-1, keepdims=True)

  # Add eps to this mean for numerical stability
  mean_squared_eps = mean_squared + eps

  # Take the square root of the result to get the RMS value
  rms_value = np.sqrt(mean_squared_eps)

  # Normalize x by dividing it by this RMS value
  normalized_x = x / rms_value

  return normalized_x

if __name__ == "__main__":
  # Create a sample NumPy array (e.g., 2x3x4)
  sample_array = np.random.rand(2, 3, 4)

  print("Original array:")
  print(sample_array)

  # Call rms_norm_standard_no_params with this array
  normalized_array = rms_norm_standard_no_params(sample_array)

  print("\nNormalized array:")
  print(normalized_array)


def rms_norm_standard_with_params(x, g, eps=1e-5):
  """
  Performs Root Mean Square (RMS) normalization with learnable gain parameters.

  Args:
    x: A multi-dimensional NumPy array. Normalization is done over the last dimension.
    g: A 1D NumPy array representing the learnable gain parameters.
       Its length should be equal to the size of the last dimension of x.
    eps: A small float value to prevent division by zero.

  Returns:
    A NumPy array with the same shape as x, normalized and scaled by g.
  """
  # Calculate the square of the input x
  x_squared = np.square(x)

  # Calculate the mean of the squared values along the last dimension
  # Keepdims is important for broadcasting
  mean_squared = np.mean(x_squared, axis=-1, keepdims=True)

  # Add eps to this mean for numerical stability
  mean_squared_eps = mean_squared + eps

  # Take the square root of the result to get the RMS value
  rms_value = np.sqrt(mean_squared_eps)

  # Normalize x by dividing it by this RMS value
  normalized_x = x / rms_value

  # Scale the normalized x by multiplying it with the gain parameter g
  # Ensure g is broadcastable to x's shape.
  # If x is [B, S, D], g is [D]. We want to reshape g to [1, 1, D] for broadcasting.
  # Or more generally, for x of shape (d1, d2, ..., dn), g of shape (dn)
  # g should be reshaped to (1, 1, ..., dn)
  g_reshaped = g
  if x.ndim > 1:
      new_shape = [1] * (x.ndim - 1) + [g.shape[0]]
      g_reshaped = g.reshape(new_shape)

  scaled_normalized_x = normalized_x * g_reshaped

  return scaled_normalized_x


if __name__ == "__main__":
  # --- Example for rms_norm_standard_no_params ---
  print("--- Example for rms_norm_standard_no_params ---")
  # Create a sample NumPy array (e.g., 2x3x4)
  sample_array_x1 = np.random.rand(2, 3, 4)

  print("Original array (x1):")
  print(sample_array_x1)

  # Call rms_norm_standard_no_params with this array
  normalized_array_x1 = rms_norm_standard_no_params(sample_array_x1)

  print("\nNormalized array (x1):")
  print(normalized_array_x1)

  # --- Example for rms_norm_standard_with_params ---
  print("\n\n--- Example for rms_norm_standard_with_params ---")
  # Create a sample NumPy array for x (e.g., 2x3x4)
  sample_array_x2 = np.random.rand(2, 3, 4)

  # Create a sample NumPy array for g (e.g., of size 4)
  # Initializing gain to ones means it initially doesn't change the scale.
  gain_params_g = np.ones(sample_array_x2.shape[-1])
  # Or random gains:
  # gain_params_g = np.random.rand(sample_array_x2.shape[-1]) * 2 # Random values typically between 0 and 2

  print("\nOriginal array (x2):")
  print(sample_array_x2)
  print("\nGain parameters (g):")
  print(gain_params_g)

  # Call rms_norm_standard_with_params with these arrays
  scaled_normalized_array_x2 = rms_norm_standard_with_params(sample_array_x2, gain_params_g)

  print("\nNormalized and scaled array (x2):")
  print(scaled_normalized_array_x2)


def rms_norm_headwise_no_params(x, num_heads, eps=1e-5):
  """
  Performs head-wise Root Mean Square (RMS) normalization on a 3D input array.
  The input is reshaped to 4D for head-wise normalization and then reshaped back.

  Args:
    x: A 3D NumPy array with shape (batch_size, seq_len, dim).
    num_heads: An integer, the number of attention heads.
    eps: A small float value to prevent division by zero.

  Returns:
    A 3D NumPy array with the same shape as x, (batch_size, seq_len, dim),
    normalized head-wise.
  """
  if x.ndim != 3:
    raise ValueError(f"Input array x must be 3-dimensional (batch_size, seq_len, dim), but got {x.ndim} dimensions.")

  batch_size, seq_len, dim = x.shape

  if not isinstance(num_heads, int) or num_heads <= 0:
    raise ValueError(f"num_heads must be a positive integer, but got {num_heads}.")
  if dim % num_heads != 0:
    raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

  head_dim = dim // num_heads

  # Reshape x to (batch_size, seq_len, num_heads, head_dim)
  x_reshaped = x.reshape(batch_size, seq_len, num_heads, head_dim)

  # Calculate the square of the reshaped x
  x_squared = np.square(x_reshaped)

  # Calculate the mean of the squared values along the last dimension (head_dim)
  # keepdims=True ensures the result has shape (batch_size, seq_len, num_heads, 1)
  mean_squared = np.mean(x_squared, axis=-1, keepdims=True)

  # Add eps to this mean for numerical stability
  mean_squared_eps = mean_squared + eps

  # Take the square root of the result to get the RMS value
  rms_value = np.sqrt(mean_squared_eps) # Shape: (batch_size, seq_len, num_heads, 1)

  # Normalize the reshaped x by dividing it by this RMS value
  normalized_x_reshaped = x_reshaped / rms_value

  # Reshape the normalized tensor back to (batch_size, seq_len, dim)
  normalized_x = normalized_x_reshaped.reshape(batch_size, seq_len, dim)

  return normalized_x


if __name__ == "__main__":
  # --- Example for rms_norm_standard_no_params ---
  print("--- Example for rms_norm_standard_no_params ---")
  # Create a sample NumPy array (e.g., 2x3x4)
  sample_array_x1 = np.random.rand(2, 3, 4)

  print("Original array (x1):")
  print(sample_array_x1)

  # Call rms_norm_standard_no_params with this array
  normalized_array_x1 = rms_norm_standard_no_params(sample_array_x1)

  print("\nNormalized array (x1):")
  print(normalized_array_x1)

  # --- Example for rms_norm_standard_with_params ---
  print("\n\n--- Example for rms_norm_standard_with_params ---")
  # Create a sample NumPy array for x (e.g., 2x3x4)
  sample_array_x2 = np.random.rand(2, 3, 4)

  # Create a sample NumPy array for g (e.g., of size 4)
  # Initializing gain to ones means it initially doesn't change the scale.
  gain_params_g = np.ones(sample_array_x2.shape[-1])
  # Or random gains:
  # gain_params_g = np.random.rand(sample_array_x2.shape[-1]) * 2 # Random values typically between 0 and 2

  print("\nOriginal array (x2):")
  print(sample_array_x2)
  print("\nGain parameters (g):")
  print(gain_params_g)

  # Call rms_norm_standard_with_params with these arrays
  scaled_normalized_array_x2 = rms_norm_standard_with_params(sample_array_x2, gain_params_g)

  print("\nNormalized and scaled array (x2):")
  print(scaled_normalized_array_x2)

  # --- Example for rms_norm_headwise_no_params ---
  print("\n\n--- Example for rms_norm_headwise_no_params ---")
  # Create a sample NumPy array for x with a 4D shape
  # (batch_size=1, seq_len=2, num_heads=3, head_dim=4)
  sample_array_x3 = np.random.rand(1, 2, 3, 4)

  print("\nOriginal 4D array (x3):")
  print(sample_array_x3)

  # Call rms_norm_headwise_no_params with this array
  headwise_normalized_array_x3 = rms_norm_headwise_no_params(sample_array_x3)

  print("\nHead-wise normalized array (x3):")
  print(headwise_normalized_array_x3)

  # Example of a head's RMS value (optional, for verification)
  # For the first head of the first batch and first sequence element:
  # head_0_0_0 = sample_array_x3[0, 0, 0, :]
  # rms_0_0_0 = np.sqrt(np.mean(np.square(head_0_0_0)) + 1e-5)
  # print(f"\nRMS for head (0,0,0): {rms_0_0_0}")
  # print(f"Normalized head (0,0,0) by calculation: {head_0_0_0 / rms_0_0_0}")
  # print(f"Normalized head (0,0,0) from function: {normalized_array_x3_3d.reshape(batch_s, seq_l, num_h, -1)[0,0,0,:]}") # Reshape to check head


def rms_norm_headwise_with_params(x, g, num_heads, eps=1e-5):
  """
  Performs head-wise RMS normalization with learnable gain parameters on a 3D input.
  The input x is reshaped to 4D for head-wise normalization and scaling,
  then reshaped back to 3D.

  Args:
    x: A 3D NumPy array with shape (batch_size, seq_len, dim).
    g: A 2D NumPy array representing the learnable gain parameters,
       with shape (num_heads, head_dim), where head_dim is dim / num_heads.
    num_heads: An integer, the number of attention heads.
    eps: A small float value to prevent division by zero.

  Returns:
    A 3D NumPy array with the same shape as x, (batch_size, seq_len, dim),
    normalized head-wise and scaled by g.
  """
  if x.ndim != 3:
    raise ValueError(f"Input array x must be 3-dimensional (batch_size, seq_len, dim), but got {x.ndim} dimensions.")

  batch_size, seq_len, dim = x.shape

  if not isinstance(num_heads, int) or num_heads <= 0:
    raise ValueError(f"num_heads must be a positive integer, but got {num_heads}.")
  if dim % num_heads != 0:
    raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")

  head_dim = dim // num_heads

  if g.ndim != 2:
    raise ValueError(f"Gain array g must be 2-dimensional (num_heads, head_dim), but got {g.ndim} dimensions.")
  if g.shape[0] != num_heads:
    raise ValueError(f"Dimension mismatch: g.shape[0] ({g.shape[0]}) should be equal to num_heads ({num_heads}).")
  if g.shape[1] != head_dim:
    raise ValueError(f"Dimension mismatch: g.shape[1] ({g.shape[1]}) should be equal to head_dim ({head_dim}).")

  # Reshape x to (batch_size, seq_len, num_heads, head_dim)
  x_reshaped = x.reshape(batch_size, seq_len, num_heads, head_dim)

  # Calculate the square of the reshaped x
  x_squared = np.square(x_reshaped)

  # Calculate the mean of the squared values along the last dimension (head_dim)
  # keepdims=True ensures the result has shape (batch_size, seq_len, num_heads, 1)
  mean_squared = np.mean(x_squared, axis=-1, keepdims=True)

  # Add eps to this mean for numerical stability
  mean_squared_eps = mean_squared + eps

  # Take the square root of the result to get the RMS value
  rms_value = np.sqrt(mean_squared_eps) # Shape: (batch_size, seq_len, num_heads, 1)

  # Normalize the reshaped x by dividing it by this RMS value
  normalized_x_reshaped = x_reshaped / rms_value

  # Scale the normalized x by multiplying it with the gain parameter g
  # g has shape (num_heads, head_dim). Reshape g to (1, 1, num_heads, head_dim) for broadcasting.
  g_broadcastable = g.reshape(1, 1, num_heads, head_dim)

  scaled_normalized_x_reshaped = normalized_x_reshaped * g_broadcastable

  # Reshape the result back to (batch_size, seq_len, dim)
  scaled_normalized_x = scaled_normalized_x_reshaped.reshape(batch_size, seq_len, dim)

  return scaled_normalized_x


if __name__ == "__main__":
  # --- Example for rms_norm_standard_no_params ---
  print("--- Example for rms_norm_standard_no_params ---")
  # Create a sample NumPy array (e.g., 2x3x4)
  sample_array_x1 = np.random.rand(2, 3, 4)

  print("Original array (x1):")
  print(sample_array_x1)

  # Call rms_norm_standard_no_params with this array
  normalized_array_x1 = rms_norm_standard_no_params(sample_array_x1)

  print("\nNormalized array (x1):")
  print(normalized_array_x1)

  # --- Example for rms_norm_standard_with_params ---
  print("\n\n--- Example for rms_norm_standard_with_params ---")
  # Create a sample NumPy array for x (e.g., 2x3x4)
  sample_array_x2 = np.random.rand(2, 3, 4)

  # Create a sample NumPy array for g (e.g., of size 4)
  # Initializing gain to ones means it initially doesn't change the scale.
  gain_params_g1 = np.ones(sample_array_x2.shape[-1])
  # Or random gains:
  # gain_params_g = np.random.rand(sample_array_x2.shape[-1]) * 2 # Random values typically between 0 and 2

  print("\nOriginal array (x2):")
  print(sample_array_x2)
  print("\nGain parameters (g1):")
  print(gain_params_g1)

  # Call rms_norm_standard_with_params with these arrays
  scaled_normalized_array_x2 = rms_norm_standard_with_params(sample_array_x2, gain_params_g1)

  print("\nNormalized and scaled array (x2):")
  print(scaled_normalized_array_x2)

  # --- Example for rms_norm_headwise_no_params (3D input) ---
  print("\n\n--- Example for rms_norm_headwise_no_params (3D input) ---")
  # Create a sample 3D NumPy array for x (e.g., 1x2x12 for batch_size=1, seq_len=2, dim=12)
  batch_s_hnp, seq_l_hnp, d_hnp = 1, 2, 12
  sample_array_x3_3d = np.random.rand(batch_s_hnp, seq_l_hnp, d_hnp)
  num_h_hnp = 3 # num_heads (e.g., 3, so head_dim would be 4)

  print(f"\nOriginal 3D array (x3_3d) shape: {sample_array_x3_3d.shape}")
  print(sample_array_x3_3d)
  print(f"Number of heads: {num_h_hnp}")

  # Call rms_norm_headwise_no_params with this 3D array and num_heads
  normalized_array_x3_3d = rms_norm_headwise_no_params(sample_array_x3_3d, num_h_hnp)

  print(f"\nHead-wise normalized 3D array (x3_3d) shape: {normalized_array_x3_3d.shape}")
  print(normalized_array_x3_3d)

  # --- Example for rms_norm_headwise_with_params (3D input) ---
  print("\n\n--- Example for rms_norm_headwise_with_params (3D input) ---")
  # Create a sample 3D NumPy array for x (e.g., 1x2x12)
  batch_s_hwp, seq_l_hwp, d_hwp = 1, 2, 12
  sample_array_x4_3d = np.random.rand(batch_s_hwp, seq_l_hwp, d_hwp)
  num_h_hwp = 3 # num_heads (e.g., 3, so head_dim is 4)
  head_d_hwp = d_hwp // num_h_hwp

  # Create a sample 2D NumPy array for g with shape (num_heads, head_dim)
  gain_params_g2_2d = np.ones((num_h_hwp, head_d_hwp))
  # Or random gains:
  # gain_params_g2_2d = np.random.rand(num_h_hwp, head_d_hwp) * 2

  print(f"\nOriginal 3D array (x4_3d) shape: {sample_array_x4_3d.shape}")
  print(sample_array_x4_3d)
  print(f"Number of heads: {num_h_hwp}, Head dimension: {head_d_hwp}")
  print(f"Gain parameters 2D (g2_2d) shape: {gain_params_g2_2d.shape}")
  print(gain_params_g2_2d)

  # Call rms_norm_headwise_with_params with these arrays
  scaled_normalized_array_x4_3d = rms_norm_headwise_with_params(sample_array_x4_3d, gain_params_g2_2d, num_h_hwp)

  print(f"\nHead-wise normalized and scaled 3D array (x4_3d) shape: {scaled_normalized_array_x4_3d.shape}")
  print(scaled_normalized_array_x4_3d)

  # Test shape validation (optional)
  # try:
  #   wrong_g_shape = np.ones((num_h_hwp + 1, head_d_hwp))
  #   rms_norm_headwise_with_params(sample_array_x4_3d, wrong_g_shape, num_h_hwp)
  # except ValueError as e:
  #   print(f"\nSuccessfully caught error for wrong g shape: {e}")

  # try:
  #   wrong_x_shape = np.random.rand(1,2,3,4) # 4D instead of 3D
  #   rms_norm_headwise_with_params(wrong_x_shape, gain_params_g2_2d, num_h_hwp)
  # except ValueError as e:
  #   print(f"\nSuccessfully caught error for wrong x shape: {e}")
