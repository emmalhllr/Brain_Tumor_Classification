"""
Brain Tumor MRI Preprocessing Pipeline
======================================

This script performs a complete preprocessing pipeline for Brain Tumor MRI datasets.
It is designed to prepare data for Deep Learning models (VGG16, ResNet50).

Pipeline Steps:
1. Deduplication: Removes exact duplicates using MD5 hashing.
2. Orientation Correction: Detects and rotates brains that are not vertically aligned.
3. ROI Extraction: Crops the brain region, removing black background and artifacts.
4. Patient-Aware Splitting: Uses Perceptual Hashing (pHash) to group similar slices 
   (simulating patient IDs) and performs a leak-free Train/Val/Test split.

"""

# Importation
import os
import shutil
import hashlib
import logging
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Set

import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ImageUtils:

    @staticmethod
    def compute_md5(filepath: str) -> str:
        hasher = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {filepath}: {e}")
            return None

    @staticmethod
    def compute_phash(filepath: str) -> imagehash.ImageHash:
        try:
            img = Image.open(filepath)
            return imagehash.phash(img)
        except Exception as e:
            logger.error(f"Failed to compute pHash for {filepath}: {e}")
            return None

    @staticmethod
    def correct_orientation(img: np.ndarray) -> Tuple[np.ndarray, bool]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img, False

        c = max(contours, key=cv2.contourArea)
        if len(c) < 5:
            return img, False

        (x, y), (MA, ma), angle = cv2.fitEllipse(c)

        if 45 < angle < 135:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return rotated, True
        
        return img, False

    @staticmethod
    def crop_brain_roi(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
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
        
        new_img = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        return new_img


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

        logger.info("Phase 1: Deduplication, Orientation, and Cropping...")
        
        seen_hashes: Set[str] = set()
        classes = [d.name for d in self.input_dir.iterdir() if d.is_dir()]
        
        for class_name in classes:
            src_class_dir = self.input_dir / class_name
            dst_class_dir = self.temp_dir / class_name
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            
            files = list(src_class_dir.glob("*.[jJ][pP][gG]")) + \
                    list(src_class_dir.glob("*.[pP][nN][gG]"))
            
            for filepath in tqdm(files, desc=f"Processing {class_name}"):
                # 1. Deduplication
                file_hash = ImageUtils.compute_md5(str(filepath))
                if file_hash in seen_hashes or file_hash is None:
                    continue
                seen_hashes.add(file_hash)
                
                img = cv2.imread(str(filepath))
                if img is None:
                    continue
                
                # 2. Orientation Correction
                img, _ = ImageUtils.correct_orientation(img)
                
                # 3. ROI Cropping
                img = ImageUtils.crop_brain_roi(img)
                
                # 4. Resize
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                cv2.imwrite(str(dst_class_dir / filepath.name), img)

    def _split_dataset(self):

        logger.info("Phase 2: Visual Clustering and Splitting...")
        
        classes = [d.name for d in self.temp_dir.iterdir() if d.is_dir()]
        threshold = 5  
        
        for class_name in classes:
            class_dir = self.temp_dir / class_name
            images = list(class_dir.glob("*"))
            
            clusters = self._cluster_images(images, threshold)
            logger.info(f"Class '{class_name}': Grouped {len(images)} images into {len(clusters)} unique clusters.")
            
            random.seed(42)
            random.shuffle(clusters)
            
            n_total = len(clusters)
            n_train = int(n_total * 0.70)
            n_val = int(n_total * 0.15)
            
            splits = {
                'train': clusters[:n_train],
                'val': clusters[n_train:n_train + n_val],
                'test': clusters[n_train + n_val:]
            }
            
            for split_name, cluster_list in splits.items():
                save_dir = self.output_dir / split_name / class_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                for cluster in cluster_list:
                    for img_path in cluster:
                        shutil.copy2(img_path, save_dir / img_path.name)

    def _cluster_images(self, image_paths: List[Path], threshold: int) -> List[List[Path]]:

        clusters = []
        ref_hashes = []
        
        for img_path in tqdm(image_paths, desc="Clustering", leave=False):
            phash = ImageUtils.compute_phash(str(img_path))
            if phash is None: continue
            
            found = False
            for i, ref_h in enumerate(ref_hashes):
                if phash - ref_h < threshold:
                    clusters[i].append(img_path)
                    found = True
                    break
            
            if not found:
                clusters.append([img_path])
                ref_hashes.append(phash)
                
        return clusters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Dataset Preprocessing Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset folder")
    parser.add_argument("--output", type=str, default="dataset_ready", help="Path to output folder")
    parser.add_argument("--size", type=int, default=224, help="Target image size (default: 224)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"Input directory '{args.input}' not found.")
        exit(1)
        
    pipeline = PreprocessingPipeline(args.input, args.output, args.size)
    pipeline.run()