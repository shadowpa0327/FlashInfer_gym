def causal_mask(q_idx, kv_idx):
    return q_idx >= kv_idx