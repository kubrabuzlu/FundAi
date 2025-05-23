import cv2
import os
from tqdm import tqdm
from typing import Tuple
import numpy as np
import argparse


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to each color channel of the image.

    Args:
        image (np.ndarray): Input color image in BGR format.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.

    Returns:
        np.ndarray: Enhanced image with CLAHE applied to each channel.
    """
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    enhanced_b = clahe.apply(b)
    enhanced_g = clahe.apply(g)
    enhanced_r = clahe.apply(r)

    return cv2.merge([enhanced_b, enhanced_g, enhanced_r])


def process_and_save_images(
    input_dir: str,
    output_dir: str,
    image_size: Tuple[int, int],
    clahe_clip_limit: float,
    clahe_tile_grid_size: Tuple[int, int]
) -> None:
    """
    Processes all images in a directory using CLAHE and saves resized enhanced images to a new directory.

    Args:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to save the processed images.
        image_size (Tuple[int, int]): Desired image size (width, height).
        clahe_clip_limit (float): CLAHE contrast limiting parameter.
        clahe_tile_grid_size (Tuple[int, int]): CLAHE tile grid size.
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_name in tqdm(os.listdir(input_dir), desc="Processing images"):
        input_path = os.path.join(input_dir, img_name)
        image = cv2.imread(input_path)
        if image is None:
            continue

        enhanced_image = apply_clahe(image, clahe_clip_limit, clahe_tile_grid_size)
        resized_image = cv2.resize(enhanced_image, image_size)
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, resized_image)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Apply CLAHE to images in a directory and save the results.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input image directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the enhanced images.")
    parser.add_argument("--width", type=int, default=224, help="Width of the output image.")
    parser.add_argument("--height", type=int, default=224, help="Height of the output image.")
    parser.add_argument("--clip_limit", type=float, default=2.0, help="CLAHE clip limit.")
    parser.add_argument("--tile_grid_size", type=int, nargs=2, default=[8, 8], help="CLAHE tile grid size (2 integers).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_and_save_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=(args.width, args.height),
        clahe_clip_limit=args.clip_limit,
        clahe_tile_grid_size=tuple(args.tile_grid_size)
    )
