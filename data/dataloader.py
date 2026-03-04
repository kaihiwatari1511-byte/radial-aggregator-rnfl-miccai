"""
FairFormer-DPT DataLoader - FIXED VERSION V3 (GEOMETRY CORRECTED)
"""

import torch
import numpy as np
import webdataset as wds
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional, Dict
import cv2
import warnings
import io
from torch.utils.data import Dataset
from PIL import Image
import os

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================
class DataConfig:
    """Configuration optimized for computing cluster"""
    
    # Paths updated to match standard repository structure
    MASTER_CSV = "./data/FairFedMed/Universal_Master_Dataset_Split_Final.csv"
    TAR_PATH = "./data/FairFedMed/fairfedmed_dataset.tar"
    
    # Model parameters
    IMG_SIZE = 224
    NUM_OUTPUTS = 360
    
    # A100 GPU settings
    BATCH_SIZE = 16
    NUM_WORKERS = 16
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Normalization
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    SEED = 42


# =============================================================================
# GEOMETRIC FIX: DISC CROP (Aligned with RETFound)
# =============================================================================
class DiscCrop:
    @staticmethod
    def apply(img, center_x, center_y, output_size=224):
        """
        Crops a square region around the optic disc.
        This keeps the anatomy CIRCULAR, so RadialAggregator works correctly.
        """
        H, W = img.shape[:2]
        
        # Crop radius: approx 25% of image dim (covers disc + peripapillary RNFL)
        crop_radius = int(min(H, W) * 0.25) 
        crop_radius = max(crop_radius, 150) # Minimum safe size
        
        # Ensure bounds
        x1 = max(0, center_x - crop_radius)
        y1 = max(0, center_y - crop_radius)
        x2 = min(W, center_x + crop_radius)
        y2 = min(H, center_y + crop_radius)
        
        # Crop
        crop = img[y1:y2, x1:x2]
        
        # Handle edge case (empty crop)
        if crop.size == 0: crop = img
        
        # Resize to 224x224 (LANCZOS4 for best vessel detail)
        resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
        
        return resized


# =============================================================================
# STREAM DECODER
# =============================================================================
def stream_decoder(sample, meta_dict, transform):
    """Decode .tar stream sample into model input"""
    
    key = sample.get("__key__", "")
    key = key.replace("./", "").replace(".npz", "")
    
    if key not in meta_dict:
        return None
    
    row = meta_dict[key]
    
    # Load NPZ
    img = None
    try:
        if "npz" in sample:
            file_bytes = sample["npz"]
        else:
            for v in sample.values():
                if isinstance(v, bytes) and len(v) > 100:
                    file_bytes = v
                    break
            else:
                return None
        
        with np.load(io.BytesIO(file_bytes), allow_pickle=True) as data:
            for slo_key in ['slo_fundus', 'slo', 'fundus', 'slo_image']:
                if slo_key in data:
                    img = data[slo_key]
                    break
            
            if img is None:
                return None
            
            if img.ndim == 3:
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
    
    except Exception:
        return None
    
    # Read centers from CSV
    center_x = int(row.get('Center_X', img.shape[1]//2))
    center_y = int(row.get('Center_Y', img.shape[0]//2))
    
    # === GEOMETRIC FIX APPLIED HERE ===
    # Using DiscCrop instead of PolarTransform
    img = DiscCrop.apply(img, center_x, center_y, DataConfig.IMG_SIZE)
    # ==================================
    
    # Convert to 3-channel
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    
    # Extract target
    target = np.array([float(row.get(str(i), 0.0)) for i in range(360)], dtype=np.float32)
    
    # Apply augmentations
    if transform:
        try:
            transformed = transform(image=img)
            img_tensor = transformed['image']
        except Exception:
            img = cv2.resize(img, (DataConfig.IMG_SIZE, DataConfig.IMG_SIZE))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mean = torch.tensor(DataConfig.MEAN).view(3, 1, 1)
            std = torch.tensor(DataConfig.STD).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
    else:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    
    # Parse metadata
    glaucoma_map = {'yes': 1, 'no': 0, '1': 1, '0': 0}
    gender_map = {'female': 1, 'male': 0, 'f': 1, 'm': 0}
    
    glaucoma = glaucoma_map.get(str(row.get('glaucoma', '')).lower().strip(), 0)
    gender = gender_map.get(str(row.get('gender', '')).lower().strip(), 0)
    
    r_str = str(row.get('race', '')).lower()
    if 'black' in r_str:
        race = 1
    elif 'asian' in r_str:
        race = 2
    elif 'hispanic' in r_str:
        race = 3
    else:
        race = 0
    
    # Normalize gender string for Trainer (M/F)
    g_raw = str(row.get('gender', '')).lower().strip()
    if g_raw in ['m', 'male', 'man']:
        gender_str = 'M'
    elif g_raw in ['f', 'female', 'woman']:
        gender_str = 'F'
    else:
        gender_str = 'U'

    return {
        'image': img_tensor,
        'target': torch.tensor(target, dtype=torch.float32),
        'key': key,
        'age': torch.tensor(float(row.get('age', 0.0)), dtype=torch.float32),
        'gender_str': gender_str,
        'race': torch.tensor(race, dtype=torch.long),
        'metadata': {
            'glaucoma': glaucoma,
            'age': float(row.get('age', 0.0)),
            'race': race,
            'gender': gender
        }
    }


# =============================================================================
# AUGMENTATION
# =============================================================================
def get_transforms():
    """Augmentation pipeline (Unchanged)"""
    
    train_transform = A.Compose([
        A.Resize(DataConfig.IMG_SIZE, DataConfig.IMG_SIZE, 
                 interpolation=cv2.INTER_LANCZOS4),
        
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        
        A.Normalize(mean=DataConfig.MEAN, std=DataConfig.STD, max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(DataConfig.IMG_SIZE, DataConfig.IMG_SIZE, 
                 interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=DataConfig.MEAN, std=DataConfig.STD, max_pixel_value=255.0),
        ToTensorV2()
    ])
    
    return train_transform, val_transform


# =============================================================================
# DATALOADER FACTORY
# =============================================================================
def create_dataloaders(
    csv_path=DataConfig.MASTER_CSV,
    tar_path=DataConfig.TAR_PATH,
    batch_size=DataConfig.BATCH_SIZE,
    num_workers=DataConfig.NUM_WORKERS,
    img_size=DataConfig.IMG_SIZE
):
    print("\n" + "="*60)
    print("CREATING DATALOADERS (FIXED VERSION V3 - DISC CROP)")
    print("="*60 + "\n")
    
    # Load CSV
    print("Loading CSV metadata...")
    df = pd.read_csv(csv_path)
    
    meta_dict = {}
    for _, row in df.iterrows():
        key = str(row['key']).replace('.npz', '').replace('./', '')
        meta_dict[key] = row
    
    print(f" Loaded {len(meta_dict)} samples\n")
    
    split_counts = df['split'].value_counts()
    train_tf, val_tf = get_transforms()
    
    def make_loader(split_name, transform, shuffle=False):
        num_samples = int(split_counts.get(split_name, 0))
        num_batches = int(np.ceil(num_samples / batch_size))
        
        def filter_split(sample):
            key = sample.get("__key__", "").replace('./', '').replace('.npz', '')
            if key not in meta_dict:
                return False
            return meta_dict[key]['split'] == split_name
        
        def process(sample):
            return stream_decoder(sample, meta_dict, transform)
        
        dataset = (
            wds.WebDataset(tar_path, resampled=True, handler=wds.warn_and_continue)
            .select(filter_split)
            .shuffle(1000 if shuffle else 0)
            .map(process)
            .select(lambda x: x is not None)
        )
        
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=DataConfig.PIN_MEMORY,
            persistent_workers=DataConfig.PERSISTENT_WORKERS
        )
        
        loader = loader.batched(batch_size, partial=not (split_name == 'train'))
        loader = loader.with_epoch(num_batches)
        
        return loader
    
    train_loader = make_loader('train', train_tf, shuffle=True)
    val_loader = make_loader('val', val_tf, shuffle=False)
    test_loader = make_loader('test', val_tf, shuffle=False)
    
    print("="*60)
    print("DATALOADER SUMMARY")
    print("="*60)
    print(f"Train: {split_counts['train']:5d} samples")
    print(f"Val:   {split_counts['val']:5d} samples")
    print(f"Test:  {split_counts['test']:5d} samples")
    print(f"Batch size: {batch_size}")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader

# =============================================================================
# VERIFICATION
# =============================================================================
if __name__ == "__main__":
    print("\n TESTING FIXED DATALOADER V3...\n")
    try:
        train_loader, val_loader, test_loader = create_dataloaders()
        print("Loading test batch...")
        batch = next(iter(train_loader))
        print("\n BATCH VERIFICATION:")
        print(f"   Image shape:  {batch['image'].shape}")
        if 'age' in batch:
            print(f"   Age shape:    {batch['age'].shape}")
            print("   ✓ Demographic data present!")
        else:
            print("   ✗ Demographic data MISSING!")
    except Exception as e:
        print(f"\n ERROR: {e}")

class GrapeDataset(Dataset):
    """
    Dataloader for the GRAPE Dataset (CFP/SLO Cross-Modality).
    Dynamically applies the same geometric DiscCrop used in the FairFedMed pipeline.
    """
    def __init__(self, excel_path, image_dir, transform=None):
        full_df = pd.read_excel(excel_path)
        self.df = full_df.iloc[1:].copy()

        # Drop missing images
        img_col_name = full_df.columns[16]
        self.df = self.df.dropna(subset=[img_col_name])

        # Force SNIT columns to numeric & Drop rows with '/'
        target_cols = self.df.columns[12:16]
        self.df[target_cols] = self.df[target_cols].apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna(subset=target_cols)

        self.image_dir = image_dir
        self.transform = transform
        self.targets = self.df[target_cols].values.astype(np.float32)
        self.filenames = self.df[img_col_name].values

        print(f"Loaded {len(self.df)} clean rows for Sector Analysis.")

    def find_disc_center(self, img_np):
        """Finds optic disc by locating the brightest blurred region (NumPy instead of PIL)"""
        cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(cv_img, (25, 25), 0)
        _, _, _, maxLoc = cv2.minMaxLoc(blurred)
        return maxLoc

    def robust_disc_crop(self, img_np, output_size=224):
        """Applies geometric crop matching FairFedMed logic, using OpenCV for speed/consistency"""
        cx, cy = self.find_disc_center(img_np)
        H, W = img_np.shape[:2]
        
        crop_radius = int(min(H, W) * 0.25)
        crop_radius = max(crop_radius, 150)
        
        # Ensure bounds
        x1 = max(0, cx - crop_radius)
        y1 = max(0, cy - crop_radius)
        x2 = min(W, cx + crop_radius)
        y2 = min(H, cy + crop_radius)
        
        crop = img_np[y1:y2, x1:x2]
        
        if crop.size == 0: 
            crop = img_np
            
        # Resize using Lanczos4 to match FairFedMed exactly
        resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)
        return resized

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        base_name = str(self.filenames[idx]).strip()
        img_path = os.path.join(self.image_dir, base_name)
        
        if not os.path.exists(img_path): 
            img_path += ".jpg"

        try:
            # 1. Load with OpenCV (Matches FairFedMed webdataset behavior)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                raise FileNotFoundError
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # 2. Apply Geometric Crop
            image_np = self.robust_disc_crop(image_rgb)
        except Exception as e:
            # Fallback to black image on failure to prevent dataloader crashing
            image_np = np.zeros((224, 224, 3), dtype=np.uint8)

        # 3. Apply Albumentations Transform
        if self.transform:
            transformed = self.transform(image=image_np)
            image_tensor = transformed['image']
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return {
            'image': image_tensor,
            'target': torch.tensor(self.targets[idx], dtype=torch.float32),  # 4-dim SNIT vector
            'key': base_name
        }
