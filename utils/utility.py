import re
import numpy as np
from scipy.ndimage import label
def rsshow(I, scale=0.005):
    low, high = np.quantile(I, [scale, 1 - scale])
    I[I > high] = high
    I[I < low] = low
    I = (I - low) / (high - low)
    return I


def keep_largest_region(mask):
    """
    Retain only the largest connected region in a binary mask.

    Parameters:
        mask (np.ndarray): Binary mask (values 0 and 1)

    Returns:
        np.ndarray: Binary mask with only the largest connected region retained
    """
    # 1. Label connected regions in the mask
    labeled_mask, num_features = label(mask)

    if num_features == 0:  # If there are no connected regions
        return np.zeros_like(mask, dtype=np.uint8)

    # 2. Compute the size of each connected region
    region_sizes = np.bincount(labeled_mask.ravel())  # Count pixels for each region
    region_sizes[0] = 0  # Ignore the background (label=0)

    # 3. Find the label of the largest region
    largest_region_label = region_sizes.argmax()

    # 4. Generate a mask containing only the largest region
    largest_region_mask = (labeled_mask == largest_region_label).astype(np.uint8)

    return largest_region_mask

