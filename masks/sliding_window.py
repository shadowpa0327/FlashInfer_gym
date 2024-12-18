def sliding_window(window_size):
    def _sliding_window(q_idx, kv_idx):
        is_causal = q_idx >= kv_idx
        is_in_window = q_idx - kv_idx <= window_size
        return is_causal & is_in_window
    return _sliding_window