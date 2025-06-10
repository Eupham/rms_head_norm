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


def rms_norm_headwise_no_params(x, eps=1e-5):
  """
  Performs head-wise Root Mean Square (RMS) normalization on a 4D input array.

  Args:
    x: A 4D NumPy array with shape (batch_size, seq_len, num_heads, head_dim).
       Normalization is performed over the last dimension (head_dim).
    eps: A small float value to prevent division by zero.

  Returns:
    A NumPy array with the same shape as x, but with each head's activations
    normalized by their own RMS.
  """
  if x.ndim != 4:
    raise ValueError(f"Input array x must be 4-dimensional (batch_size, seq_len, num_heads, head_dim), but got {x.ndim} dimensions.")

  # Calculate the square of the input x
  x_squared = np.square(x)

  # Calculate the mean of the squared values along the last dimension (head_dim)
  # keepdims=True ensures the result has shape (batch_size, seq_len, num_heads, 1)
  mean_squared = np.mean(x_squared, axis=-1, keepdims=True)

  # Add eps to this mean for numerical stability
  mean_squared_eps = mean_squared + eps

  # Take the square root of the result to get the RMS value
  rms_value = np.sqrt(mean_squared_eps)

  # Normalize x by dividing it by this RMS value
  # Broadcasting will handle element-wise division for each head
  normalized_x = x / rms_value

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
  # print(f"Normalized head (0,0,0) from function: {headwise_normalized_array_x3[0, 0, 0, :]}")


def rms_norm_headwise_with_params(x, g, eps=1e-5):
  """
  Performs head-wise RMS normalization with learnable gain parameters on a 4D input.

  Args:
    x: A 4D NumPy array with shape (batch_size, seq_len, num_heads, head_dim).
       Normalization is performed over the last dimension (head_dim).
    g: A 2D NumPy array representing the learnable gain parameters,
       with shape (num_heads, head_dim).
    eps: A small float value to prevent division by zero.

  Returns:
    A NumPy array with the same shape as x, with each head's activations
    normalized by their own RMS and scaled by corresponding gains in g.
  """
  if x.ndim != 4:
    raise ValueError(f"Input array x must be 4-dimensional (batch_size, seq_len, num_heads, head_dim), but got {x.ndim} dimensions.")
  if g.ndim != 2:
    raise ValueError(f"Gain array g must be 2-dimensional (num_heads, head_dim), but got {g.ndim} dimensions.")
  if x.shape[2] != g.shape[0]:
    raise ValueError(f"Dimension mismatch: x.shape[2] (num_heads in x) should be equal to g.shape[0] (num_heads in g). Got {x.shape[2]} and {g.shape[0]}.")
  if x.shape[3] != g.shape[1]:
    raise ValueError(f"Dimension mismatch: x.shape[3] (head_dim in x) should be equal to g.shape[1] (head_dim in g). Got {x.shape[3]} and {g.shape[1]}.")

  # Calculate the square of the input x
  x_squared = np.square(x)

  # Calculate the mean of the squared values along the last dimension (head_dim)
  # keepdims=True ensures the result has shape (batch_size, seq_len, num_heads, 1)
  mean_squared = np.mean(x_squared, axis=-1, keepdims=True)

  # Add eps to this mean for numerical stability
  mean_squared_eps = mean_squared + eps

  # Take the square root of the result to get the RMS value
  rms_value = np.sqrt(mean_squared_eps)

  # Normalize x by dividing it by this RMS value
  normalized_x = x / rms_value

  # Scale the normalized x by multiplying it with the gain parameter g
  # g has shape (num_heads, head_dim). It needs to be reshaped for broadcasting with x.
  # Reshape g to (1, 1, num_heads, head_dim) to align with x's dimensions.
  g_reshaped = g.reshape(1, 1, g.shape[0], g.shape[1])

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
  # print(f"Normalized head (0,0,0) from function: {headwise_normalized_array_x3[0, 0, 0, :]}")

  # --- Example for rms_norm_headwise_with_params ---
  print("\n\n--- Example for rms_norm_headwise_with_params ---")
  # Create a sample NumPy array for x with a 4D shape
  # (batch_size=1, seq_len=2, num_heads=3, head_dim=4)
  sample_array_x4 = np.random.rand(1, 2, 3, 4)

  # Create a sample NumPy array for g with shape (num_heads, head_dim)
  # For this example: (3, 4)
  gain_params_g2 = np.ones((sample_array_x4.shape[2], sample_array_x4.shape[3]))
  # Or random gains:
  # gain_params_g2 = np.random.rand(sample_array_x4.shape[2], sample_array_x4.shape[3]) * 2

  print("\nOriginal 4D array (x4):")
  print(sample_array_x4)
  print("\nGain parameters 2D (g2) shape {}:".format(gain_params_g2.shape))
  print(gain_params_g2)

  # Call rms_norm_headwise_with_params with these arrays
  scaled_headwise_normalized_array_x4 = rms_norm_headwise_with_params(sample_array_x4, gain_params_g2)

  print("\nHead-wise normalized and scaled array (x4):")
  print(scaled_headwise_normalized_array_x4)

  # Test shape validation (optional)
  # try:
  #   wrong_g_shape = np.ones((sample_array_x4.shape[2] + 1, sample_array_x4.shape[3]))
  #   rms_norm_headwise_with_params(sample_array_x4, wrong_g_shape)
  # except ValueError as e:
  #   print(f"\nSuccessfully caught error for wrong g shape: {e}")

  # try:
  #   wrong_x_shape = np.random.rand(1,2,3) # 3D instead of 4D
  #   rms_norm_headwise_with_params(wrong_x_shape, gain_params_g2)
  # except ValueError as e:
  #   print(f"\nSuccessfully caught error for wrong x shape: {e}")
