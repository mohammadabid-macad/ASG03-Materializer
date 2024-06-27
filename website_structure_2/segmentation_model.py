import numpy as np
from PIL import Image

def run_segmentation_model(image_path):
    # Dummy implementation
    # Replace with actual model inference code
    brick_percentage = 0.45779499411582947
    ceramic_percentage = 0.058589570224285126
    glass_percentage = 0.03563307225704193
    metal_percentage = 0.000011294204341538716
    painted_percentage = 0.015030953101813793
    stone_percentage = 0.07172920554876328
    tile_percentage = 0.07241785526275635
    wood_percentage = 0.20852695405483246

    # For simplicity, returning a black image as a placeholder
    image = Image.open(image_path)
    segmented_image = np.zeros_like(np.array(image))

    return segmented_image, {
        "brick_percentage": brick_percentage,
        "ceramic_percentage": ceramic_percentage,
        "glass_percentage": glass_percentage,
        "metal_percentage": metal_percentage,
        "painted_percentage": painted_percentage,
        "stone_percentage": stone_percentage,
        "tile_percentage": tile_percentage,
        "wood_percentage": wood_percentage,
    }
