def attamba_mask(chunk_size: int, window_size: int):
    """Create an attention mask function for attamba-style masking.
    
    Args:
        chunk_size: Size of chunks for boundary checking
        window_size: Size of window for lead tokens
    
    Returns:
        Callable that takes q_idx and kv_idx tensors and returns a boolean mask tensor
    """
    def _attamba_mask(q_idx, kv_idx):
        # Check if kv position is a chunk boundary
        is_chunk_boundary = ((kv_idx + 1) % chunk_size) == 0
        
        # Check if within window size (lead tokens)
        is_lead_tokens = (q_idx - kv_idx) < window_size
        
        # Causal masking
        causal_mask = q_idx >= kv_idx
        
        # Combine all conditions
        return (is_chunk_boundary | is_lead_tokens) & causal_mask
        
    return _attamba_mask