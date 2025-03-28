import torch
import numpy as np

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention as described in the attention implementation guide.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k)
        value: Value tensor of shape (batch_size, seq_len, d_v)
        mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len)
        
    Returns:
        tuple: (output, attention_weights)
            - output: Attention output of shape (batch_size, seq_len, d_v)
            - attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
    """
    # Get dimensions
    d_k = query.size(-1)
    
    # Calculate dot products
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)
    
    # Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Calculate weighted sum
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

def multi_head_attention(query, key, value, num_heads, mask=None):
    """
    Compute multi-head attention as used in transformer architectures.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, d_model)
        key: Key tensor of shape (batch_size, seq_len, d_model)
        value: Value tensor of shape (batch_size, seq_len, d_model)
        num_heads: Number of attention heads
        mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len)
        
    Returns:
        tuple: (output, attention_weights)
            - output: Attention output of shape (batch_size, seq_len, d_model)
            - attention_weights: List of attention weights, one per head
    """
    batch_size, seq_len, d_model = query.size()
    
    # Head dimension
    d_k = d_model // num_heads
    
    # Split heads
    def split_heads(x):
        # Reshape from (batch_size, seq_len, d_model) to (batch_size, num_heads, seq_len, d_k)
        x = x.view(batch_size, seq_len, num_heads, d_k)
        return x.permute(0, 2, 1, 3)
    
    # Split Q, K, V into heads
    q_heads = split_heads(query)
    k_heads = split_heads(key)
    v_heads = split_heads(value)
    
    # Calculate attention for each head
    attn_output_heads = []
    attn_weights_heads = []
    
    for i in range(num_heads):
        head_output, head_weights = scaled_dot_product_attention(
            q_heads[:, i:i+1], 
            k_heads[:, i:i+1], 
            v_heads[:, i:i+1], 
            mask
        )
        attn_output_heads.append(head_output)
        attn_weights_heads.append(head_weights)
    
    # Concatenate heads
    attn_output = torch.cat(attn_output_heads, dim=1)
    
    # Reshape back to original shape
    attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, d_model)
    
    return attn_output, attn_weights_heads


# Fixed-point approximation utilities for hardware implementation reference
def fixed_point_softmax(x, scale_factor=256, bits=8):
    """
    Approximate softmax using fixed-point arithmetic
    
    Args:
        x: Input tensor
        scale_factor: Fixed-point scaling factor
        bits: Bit width
        
    Returns:
        Fixed-point softmax approximation
    """
    # Find maximum for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    x_norm = x - x_max
    
    # Approximate exp using lookup table or piece-wise linear approximation
    # This is a simple piece-wise linear approximation
    def approx_exp(x, scale=scale_factor):
        # Clamp values to prevent overflow
        x_clamped = np.clip(x, -8.0, 8.0)
        
        # Basic piece-wise linear approximation of exp
        if x_clamped <= 0:
            return scale * (1.0 + 0.5 * x_clamped)
        else:
            return scale * (1.0 + x_clamped + 0.5 * x_clamped**2)
    
    exp_x = np.vectorize(approx_exp)(x_norm)
    
    # Calculate sum and normalize
    sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
    softmax_result = exp_x / sum_exp
    
    # Quantize to specified bit width
    max_val = (1 << bits) - 1
    quantized = np.round(softmax_result * max_val) / max_val
    
    return quantized


def generate_test_vectors(batch_size=1, seq_len=16, d_model=64, num_heads=4):
    """
    Generate test vectors for attention implementation
    
    Returns:
        Dictionary of test vectors and expected outputs
    """
    # Create random input tensors
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Calculate expected outputs (float precision)
    sdpa_output, sdpa_weights = scaled_dot_product_attention(query, key, value)
    mha_output, mha_weights = multi_head_attention(query, key, value, num_heads)
    
    # Convert to numpy arrays for fixed-point conversion
    query_np = query.numpy()
    key_np = key.numpy()
    value_np = value.numpy()
    
    # Expected quantized outputs (for hardware comparison)
    # This would be compared against RTL simulation results
    qk_scores = np.matmul(query_np, np.transpose(key_np, (0, 2, 1))) / np.sqrt(d_model)
    softmax_approx = fixed_point_softmax(qk_scores)
    output_approx = np.matmul(softmax_approx, value_np)
    
    return {
        "inputs": {
            "query": query,
            "key": key,
            "value": value,
            "seq_len": seq_len,
            "d_model": d_model,
            "num_heads": num_heads
        },
        "expected_outputs": {
            "sdpa_output": sdpa_output,
            "sdpa_weights": sdpa_weights,
            "mha_output": mha_output,
            "mha_weights": mha_weights
        },
        "quantized_expected": {
            "qk_scores": qk_scores,
            "softmax_approx": softmax_approx,
            "output_approx": output_approx
        }
    }


if __name__ == "__main__":
    # Generate test vectors and print shapes
    test_data = generate_test_vectors(batch_size=2, seq_len=16, d_model=64, num_heads=4)
    
    print("Test Vectors Generated:")
    print(f"Query shape: {test_data['inputs']['query'].shape}")
    print(f"Key shape: {test_data['inputs']['key'].shape}")
    print(f"Value shape: {test_data['inputs']['value'].shape}")
    print(f"SDPA output shape: {test_data['expected_outputs']['sdpa_output'].shape}")
    print(f"MHA output shape: {test_data['expected_outputs']['mha_output'].shape}")
    
    # Calculate error between float and fixed-point implementations
    float_output = test_data['expected_outputs']['sdpa_output'].numpy()
    fixed_output = test_data['quantized_expected']['output_approx']
    
    print(f"\nMean squared error between float and fixed-point: {np.mean((float_output - fixed_output)**2)}")