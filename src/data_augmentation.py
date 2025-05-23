import argparse
import os
import cv2
import numpy as np
import albumentations as A
import math
from typing import List
from tqdm import tqdm


def get_class_folders(base_path: str) -> List[str]:
    """
    Get the list of subdirectories (class folders) in the given directory.

    Args:
        base_path (str): The path to the directory containing class folders.

    Returns:
        List[str]: A list of class folder names found in the directory.
    """
    return [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]


def augment_image(image: np.ndarray, augmenter: A.Compose) -> np.ndarray:
    """
    Apply augmentation transformations to a single image.

    Args:
        image (np.ndarray): The input image array.
        augmenter (A.Compose): Albumentations Compose object defining augmentations.

    Returns:
        np.ndarray: The augmented image.
    """
    augmented = augmenter(image=image)
    return augmented["image"]


def augment_class_images(
        class_path: str,
        desired_samples: int,
        augmenter: A.Compose,
        image_size: int = 512
) -> None:
    """
    Perform data augmentation on images within a class folder until the desired number of samples is reached.

    Args:
        class_path (str): Path to the class folder containing images.
        desired_samples (int): Target number of samples for this class after augmentation.
        augmenter (A.Compose): Albumentations Compose object defining augmentations.
        image_size (int, optional): Size to which images will be resized. Defaults to 512.

    Returns:
        None
    """
    file_list = os.listdir(class_path)
    for file_name in file_list:
        try:
            img_path = os.path.join(class_path, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (image_size, image_size))

            for i in range(math.ceil(desired_samples / len(file_list)) - 1):
                augmented_img = augment_image(img, augmenter)
                aug_name = f"{os.path.splitext(file_name)[0]}_augmented_{i}.jpg"
                aug_path = os.path.join(class_path, aug_name)
                cv2.imwrite(aug_path, augmented_img)

                if len(os.listdir(class_path)) >= desired_samples:
                    break
            if len(os.listdir(class_path)) >= desired_samples:
                break
        except Exception:
            continue


def augment_dataset(
        dataset_path: str,
        fold: str,
        desired_samples: int,
        augmenter: A.Compose
) -> None:
    """
    Augment all image classes within a specific fold of the dataset to balance the number of samples per class.

    Args:
        dataset_path (str): Base path to the dataset.
        fold (str): Specific fold directory to process (e.g., "fold1").
        desired_samples (int): Target number of samples per class after augmentation.
        augmenter (A.Compose): Albumentations Compose object defining augmentations.

    Returns:
        None
    """
    train_path = os.path.join(dataset_path, fold, "train")
    class_folders = get_class_folders(train_path)

    for class_name in class_folders:
        print(f"Processing class: {class_name}")
        class_path = os.path.join(train_path, class_name)
        augment_class_images(class_path, desired_samples, augmenter)

        print(f"Final image count for {class_name}: {len(os.listdir(class_path))}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Augment dataset using Albumentations")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory (e.g., 'ODIR').")
    parser.add_argument("--fold", type=str, default="fold1", help="Fold name under dataset (e.g., 'fold1').")
    parser.add_argument("--desired_samples", type=int, default=10000, help="Target number of images per class.")
    parser.add_argument("--image_size", type=int, default=512, help="Size to resize images before augmentation.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    augmenter = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    ])

    augment_dataset(
        dataset_path=args.dataset_path,
        fold=args.fold,
        desired_samples=args.desired_samples,
        augmenter=augmenter,
        image_size=args.image_size
    )


