"""
Brain Tumor MRI Preprocessing Pipeline
======================================

Pipeline Steps:
1. Color Standardization: force RGB 3 channels (handle RGBA, grayscale, etc.)
2. Content-Based Deduplication (32x32): removes duplicates even if color differs.
3. Orientation Correction (only for filenames starting with "brain_").
4. ROI Extraction.
5. Classic Split (70/15/15).
"""

import os
import shutil
import hashlib
import logging
import argparse
import random
from pathlib import Path
from typing import Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class ImageUtils:

    # Convert image to RGB 3 channels
    @staticmethod
    def to_rgb_3ch(img: np.ndarray) -> np.ndarray:
        
        if img is None:
            return None

        # grayscale -> RGB
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # RGBA -> RGB
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # RGB -> if effectively gray, force gray
        if img.shape[2] == 3:
            if np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2]):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return img

    # Computes a hash based on the visual content of the image (32x32)
    @staticmethod
    def compute_content_hash(img_array: np.ndarray, size: int = 32) -> str:

        if img_array is None:
            return None

        # convert to grayscale for hashing
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        thumb = cv2.resize(img_array, (size, size), interpolation=cv2.INTER_AREA)
        return hashlib.md5(thumb.tobytes()).hexdigest()

    # Detect if image should be rotated
    @staticmethod
    def correct_orientation(img: np.ndarray) -> Tuple[np.ndarray, bool]:

        gray = cv2.GaussianBlur(img, (7, 7), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img, False

        c = max(contours, key=cv2.contourArea)
        if len(c) < 5:
            return img, False

        x, y, w, h = cv2.boundingRect(c)
        if h > 0 and (w / float(h)) > 1.2:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True

        return img, False

    # Crop the brain from the image
    @staticmethod
    def crop_brain_roi(img: np.ndarray) -> np.ndarray:

        gray = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        c = max(contours, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        return img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]


class PreprocessingPipeline:

    def __init__(self, input_dir: str, output_dir: str, img_size: int = 224):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.temp_dir = self.output_dir / "temp_processing"

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def run(self):
        logger.info("Starting Preprocessing Pipeline...")
        self._process_images()
        self._split_dataset()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        logger.info("Pipeline completed successfully.")

    def _process_images(self):
        logger.info("Phase 1: Loading, Color Standardization, Deduplication, Processing...")

        seen_hashes: Set[str] = set()
        duplicates_count = 0

        classes = [d.name for d in self.input_dir.iterdir() if d.is_dir()]

        for class_name in classes:
            src_class_dir = self.input_dir / class_name
            dst_class_dir = self.temp_dir / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            files = list(src_class_dir.glob("*.[jJ][pP][gG]")) + \
                    list(src_class_dir.glob("*.[pP][nN][gG]"))

            for filepath in tqdm(files, desc=f"Processing {class_name}"):

                # 1. LOAD IMAGE
                img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue

                # 2. Convert to RGB 3 channels
                img = ImageUtils.to_rgb_3ch(img)
                if img is None:
                    continue

                # 3. Deduplication based on content hash (32x32 grayscale)
                content_hash = ImageUtils.compute_content_hash(img, size=32)
                if content_hash in seen_hashes:
                    duplicates_count += 1
                    continue
                seen_hashes.add(content_hash)

                # 4. Orientation correction if filename starts with "brain_"
                if filepath.name.startswith("brain_"):
                    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img_gray, _ = ImageUtils.correct_orientation(img_gray)
                    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

                # 5. ROI crop
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_roi = ImageUtils.crop_brain_roi(img_gray)

                # 6. Resize + convert back to RGB
                img_roi = cv2.resize(img_roi, (self.img_size, self.img_size))
                img_roi = cv2.cvtColor(img_roi, cv2.COLOR_GRAY2RGB)

                # 7. Save
                cv2.imwrite(str(dst_class_dir / filepath.name), img_roi)

        logger.info(f"Removed {duplicates_count} visual duplicates (32x32 grayscale).")

    def _split_dataset(self):
        logger.info("Phase 2: Splitting dataset (classic 70/15/15)...")

        classes = [d.name for d in self.temp_dir.iterdir() if d.is_dir()]

        for class_name in classes:
            class_dir = self.temp_dir / class_name
            images = list(class_dir.glob("*"))

            random.seed(42)
            random.shuffle(images)

            n_total = len(images)
            n_train = int(n_total * 0.70)
            n_val = int(n_total * 0.15)

            splits = {
                'train': images[:n_train],
                'val': images[n_train:n_train + n_val],
                'test': images[n_train + n_val:]
            }

            for split_name, img_list in splits.items():
                save_dir = self.output_dir / split_name / class_name
                save_dir.mkdir(parents=True, exist_ok=True)
                for img_path in img_list:
                    shutil.copy2(img_path, save_dir / img_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/processed/dataset_final")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()

    if os.path.exists(args.input):
        pipeline = PreprocessingPipeline(args.input, args.output, args.size)
        pipeline.run()
