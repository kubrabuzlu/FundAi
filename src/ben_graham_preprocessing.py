import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from typing import Optional


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crop black areas from the image.

    Args:
        img (np.ndarray): Input image.
        tol (int): Tolerance for grayscale threshold.

    Returns:
        np.ndarray: Cropped image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
            return img  # If image is too dark, return original
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img


def load_ben_color(path: str, sigmaX: int = 10, image_size: int = 224, tol: int = 7) -> np.ndarray:
    """
    Apply Ben Graham preprocessing: crop, resize, and Gaussian blur enhancement.

    Args:
        path (str): Path to image file.
        sigmaX (int): Standard deviation for Gaussian blur.
        image_size (int): Target image size.
        tol (int): Tolerance for cropping.

    Returns:
        np.ndarray: Processed image.
    """
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image, tol=tol)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def process_folder(input_dir: str, output_dir: str, sigmaX: int, image_size: int, tol: int) -> None:
    """
    Process all images in a folder using Ben preprocessing and save results.

    Args:
        input_dir (str): Path to folder containing input images.
        output_dir (str): Path to folder where output images will be saved.
        sigmaX (int): Sigma value for Gaussian blur.
        image_size (int): Resize size.
        tol (int): Tolerance for cropping.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                processed_img = load_ben_color(input_path, sigmaX, image_size, tol)
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, processed_img)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Ben preprocessing to images in a folder.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output image folder")
    parser.add_argument("--sigmaX", type=int, default=10, help="Sigma for Gaussian blur")
    parser.add_argument("--image_size", type=int, default=224, help="Target image size")
    parser.add_argument("--tol", type=int, default=7, help="Tolerance for cropping black areas")

    args = parser.parse_args()

    process_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sigmaX=args.sigmaX,
        image_size=args.image_size,
        tol=args.tol
    )
