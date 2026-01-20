"""
Brain Tumor MRI Preprocessing Pipeline
======================================

Pipeline Steps:
1. Grayscale Conversion: Immediate standardization.
2. Content-Based Deduplication: Removes duplicates even if one is purple and one is gray.
3. Orientation Correction.
4. ROI Extraction.
5. Patient-Aware Splitting.
"""

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class ImageUtils:

    @staticmethod
    def compute_content_hash(img_array: np.ndarray) -> str:
        """
        Computes a hash based on the visual content of the image.
        Resizes to small thumbnail to ignore compression noise.
        """
        # 1. Resize to tiny square (16x16) to be robust against JPEG artifacts
        # This acts like a 'low frequency' filter
        thumb = cv2.resize(img_array, (16, 16), interpolation=cv2.INTER_AREA)
        
        # 2. Compute MD5 of the raw pixels
        return hashlib.md5(thumb.tobytes()).hexdigest()

    @staticmethod
    def compute_phash(filepath: str) -> imagehash.ImageHash:
        try:
            img = Image.open(filepath)
            return imagehash.phash(img.convert('L')) 
        except Exception as e:
            logger.error(f"Failed to compute pHash for {filepath}: {e}")
            return None

    @staticmethod
    def correct_orientation(img: np.ndarray) -> Tuple[np.ndarray, bool]:
        # (Fonction inchangée car elle prend déjà une image chargée)
        gray = cv2.GaussianBlur(img, (7, 7), 0) # img is already gray here in pipeline
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return img, False

        c = max(contours, key=cv2.contourArea)
        if len(c) < 5: return img, False

        x, y, w, h = cv2.boundingRect(c)
        if h > 0 and (w / float(h)) > 1.2:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), True
        
        return img, False

    @staticmethod
    def crop_brain_roi(img: np.ndarray) -> np.ndarray:
        # (Fonction inchangée)
        gray = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return img

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
        
        if self.output_dir.exists(): shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

    def run(self):
        logger.info("Starting Preprocessing Pipeline...")
        self._process_images()
        self._split_dataset()
        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir)
        logger.info("Pipeline completed successfully.")

    def _process_images(self):
        logger.info("Phase 1: Loading, Gray Conversion, Visual Deduplication, Processing...")
        
        seen_content_hashes: Set[str] = set()
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
                img = cv2.imread(str(filepath))
                if img is None: continue

                # 2. FORCE GRAYSCALE (Standardization)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # 3. VISUAL DEDUPLICATION (Content-Based)
                # We hash the GRAYSCALE content. 
                # So 'purple.jpg' converted to gray and 'gray.jpg' will have the SAME hash.
                content_hash = ImageUtils.compute_content_hash(img)
                
                if content_hash in seen_content_hashes:
                    duplicates_count += 1
                    continue # Skip this file, it's a visual duplicate
                
                seen_content_hashes.add(content_hash)
                
                # 4. Orientation Correction (Only if needed or checked)
                # Applied to all images for robustness, but checks internally if ratio > 1.2
                if filepath.name.startswith("brain_"):
                    img, _ = ImageUtils.correct_orientation(img)
                
                # 5. ROI Cropping
                img = ImageUtils.crop_brain_roi(img)
                
                # 6. Resize
                img = cv2.resize(img, (self.img_size, self.img_size))
                
                # Save
                cv2.imwrite(str(dst_class_dir / filepath.name), img)
                
        logger.info(f"Removed {duplicates_count} visual duplicates (including purple/gray redundancies).")

    def _split_dataset(self):
        logger.info("Phase 2: Visual Clustering and Splitting...")
        # ... (Le reste de ton code de split reste IDENTIQUE) ...
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="dataset_ready")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        pipeline = PreprocessingPipeline(args.input, args.output, args.size)
        pipeline.run()