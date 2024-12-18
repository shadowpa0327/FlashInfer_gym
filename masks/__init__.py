from masks.causal import causal_mask
from masks.sliding_window import sliding_window
from masks.attamba import attamba_mask


from enum import Enum
from typing import Callable, Optional, Union

_supported_masks = ['causal', 'sliding_window', 'attamba']

class MaskType(Enum):
    """Supported mask types"""
    CAUSAL = "causal"
    SLIDING_WINDOW = "sliding_window"
    ATTAMBA = "attamba"
    

def get_mask_fn(
    mask_type: Union[str, MaskType],
    window_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
):
    """Get attention mask function based on mask type.
    
    Args:
        mask_type: Type of mask to use. Can be either a string or MaskType enum.
                  Supported values: "causal", "sliding_window", "attamba", "full"
        window_size: Size of attention window for sliding_window and attamba masks
        chunk_size: Size of chunks for attamba mask
    
    Returns:
        mask_fn: A function that takes q_idx and kv_idx and returns a boolean mask
    
    Raises:
        ValueError: If mask_type is not supported or required parameters are missing
    """
    if isinstance(mask_type, str):
        try:
            mask_type = MaskType(mask_type.lower())
        except ValueError:
            raise ValueError(
                f"Unsupported mask type: {mask_type}. "
                f"Supported types are: {[m.value for m in MaskType]}"
            )

    if mask_type == MaskType.CAUSAL:
        return causal_mask
    
    elif mask_type == MaskType.SLIDING_WINDOW:
        if window_size is None:
            raise ValueError("window_size must be provided for sliding window mask")
        return sliding_window(window_size=window_size)

    elif mask_type == MaskType.ATTAMBA:
        if window_size is None or chunk_size is None:
            raise ValueError("Both window_size and chunk_size must be provided for attamba mask")
        return attamba_mask(chunk_size=chunk_size, window_size=window_size)
    
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")
