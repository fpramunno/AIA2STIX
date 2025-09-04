import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from pathlib import Path
from torchvision.transforms import Compose, Normalize
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from typing import Iterator
from datetime import datetime
from collections.abc import Sized

from src.utils import get_alpha, get_default_transforms

class RandomSamplerSeed(Sampler[int]):
    """Overwrite the RandomSampler to allow for a seed for each epoch.
    Effectively going over the same data at same epochs."""

    def __init__(
        self,
        dataset: Sized, 
        num_samples: int | None = None,
        generator=None, 
        epoch: int = 0
    ):
        self.dataset = dataset
        self._num_samples = num_samples
        self.generator = generator
        self.epoch = epoch

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.dataset)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if self.generator is None:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=g).item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        for _ in range(self.num_samples // n):
            yield from torch.randperm(n, generator=generator).tolist()
        yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return len(self.dataset)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

def extract_timestamp(filename):
            base = Path(filename).stem
            parts = base.split('_')
            try:
                dt = datetime(
                    int(parts[-6]), int(parts[-5]), int(parts[-4]),
                    int(parts[-3]), int(parts[-2]), int(parts[-1])
                )
            except Exception:
                dt = None
            return dt

def extract_timestamp_from_aia_processed(filename):
    """
    Extract timestamp from aia_processed files.
    Expected format: aia_processed_YYMMDDHHMMSS_rowindex.npy
    """
    base = Path(filename).stem  # Remove .npy extension
    parts = base.split('_')
    
    if len(parts) >= 3 and parts[0] == 'aia' and parts[1] == 'processed':
        flare_id = parts[2]  # e.g., '2105221710'
        
        if len(flare_id) == 10:  # YYMMDDHHMMSS format
            try:
                year = 2000 + int(flare_id[:2])    # 21 -> 2021
                month = int(flare_id[2:4])         # 05
                day = int(flare_id[4:6])           # 22
                hour = int(flare_id[6:8])          # 17
                minute = int(flare_id[8:10])       # 10
                
                return datetime(year, month, day, hour, minute)
            except (ValueError, IndexError):
                return None
    
    return None


def parse_complex_visibility(complex_str):
    """
    Parse complex number string like '(0.62395840883255-0.14794765412807465j)' 
    into [real, imag] array.
    """
    import re
    
    # Remove parentheses and extract the complex number
    clean_str = complex_str.strip('()')
    
    try:
        # Convert to complex number
        complex_num = complex(clean_str)
        return [complex_num.real, complex_num.imag]
    except (ValueError, TypeError):
        # Return zeros if parsing fails
        return [0.0, 0.0]


class AIA2STIXDataset(Dataset):  # type: ignore
    def __init__(
        self,
        data_path=None,
        vis_path=None,
        train_val_test=[0.8, 0.10, 0.10],
        split="train",
        transforms=True,
        seed=42,
        transform_aia=None
    ):
        # path to dataset
        base_path = data_path
        self.base_path = Path(base_path)

        # path to visibilities
        vis_base_path = vis_path
        self.vis_base_path = Path(vis_base_path)

        # Transformations
        self.are_transform = transforms
        self.transform_aia = transform_aia if transform_aia is not None else None

        self.train_perc = train_val_test[0]
        self.valid_perc = train_val_test[1]
        self.test_perc = train_val_test[2]
        self.seed = seed
        self.split = split
        
        # Load visibility data and create timestamp mapping
        self.visibility_data = self._load_visibility_data()
        
        # Get file paths and filter only those with matching visibility data
        self.files_paths = self.get_filespaths()

    def _load_visibility_data(self):
        """Load visibility CSV and create timestamp-based lookup."""
        df = pd.read_csv(self.vis_base_path)
        
        # Create a dictionary mapping flare_id to visibility data
        vis_dict = {}
        
        for _, row in df.iterrows():
            flare_id = str(row['flare_id'])
            
            # Extract visibility columns (vis_01 to vis_24)
            vis_cols = [f'vis_{i:02d}' for i in range(1, 25)]
            
            # Parse complex numbers and convert to (24, 2) array
            visibilities = []
            for col in vis_cols:
                if col in row and pd.notna(row[col]):
                    real_imag = parse_complex_visibility(str(row[col]))
                    visibilities.append(real_imag)
                else:
                    # Fill missing with zeros
                    visibilities.append([0.0, 0.0])
            
            vis_dict[flare_id] = np.array(visibilities, dtype=np.float32)  # Shape: (24, 2)
        
        return vis_dict

    def get_filespaths(self):
        # Get all files in the base path
        files = list(self.base_path.glob("aia_processed_*.npy"))
        
        # Filter files that have corresponding visibility data
        valid_files = []
        for file_path in files:
            timestamp = extract_timestamp_from_aia_processed(file_path.name)
            if timestamp is not None:
                # Extract flare_id from filename
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    flare_id = parts[2]
                    if flare_id in self.visibility_data:
                        valid_files.append(file_path)
        
        # Sort the files
        valid_files.sort()
        
        # Shuffle deterministically
        rng = random.Random(self.seed)
        rng.shuffle(valid_files)

        # Compute sizes
        total_size = len(valid_files)
        train_size = int(self.train_perc * total_size)
        val_size = int(self.valid_perc * total_size)
        test_size = total_size - train_size - val_size  # remainder

        # Split the list
        train_files = valid_files[:train_size]
        val_files = valid_files[train_size:train_size + val_size]
        test_files = valid_files[train_size + val_size:]

        file_paths = {
            'train': train_files,
            'valid': val_files,
            'test': test_files,
        }[self.split]

        return file_paths
    
    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        # Load the data
        file_path = self.files_paths[idx]
        data = np.load(file_path, allow_pickle=True)
        
        # Extract flare_id from filename
        parts = file_path.stem.split('_')
        flare_id = parts[2]
        
        # Get corresponding visibility data
        visibility_matrix = self.visibility_data[flare_id]  # Shape: (24, 2)

        alpha_vis = get_alpha(visibility_matrix)  # Compute alpha value
        visibility_norm = visibility_matrix / alpha_vis if alpha_vis > 0 else visibility_matrix
        # Convert to tensors
        data_tensor = torch.from_numpy(data).float()
        if data_tensor.ndim == 2:  # Add channel dimension if needed
            data_tensor = data_tensor.unsqueeze(0)
        
        # Apply transformations if any
        if self.are_transform:
            data_tensor = self.transform_aia(data_tensor) if self.transform_aia else data_tensor
        
        visibility_tensor = torch.from_numpy(visibility_norm).float()  # Shape: (24, 2)
        
        return data_tensor, visibility_tensor

# Create combined dataset
class CombinedAIA2STIXDataset(Dataset):
    def __init__(self, base_dataset, encoded_data):
        self.base_dataset = base_dataset
        self.encoded_data = encoded_data
        assert len(base_dataset) == len(encoded_data), f"Dataset length mismatch: {len(base_dataset)} vs {len(encoded_data)}"
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        aia_data, vis_data = self.base_dataset[idx]
        enc_data = torch.from_numpy(self.encoded_data[idx]).float()
        return aia_data, vis_data, enc_data

def get_aia2stix_data_objects(
    batch_size,
    distributed,
    num_data_workers,
    train_val_test=[0.8, 0.10, 0.10],
    rank=None,
    world_size=None,
    split="train",
    data_path=None,
    vis_path=None,
    transforms=True,
    seed=42,
    enc_data_path=None
):
    """
    Create data loader for AIA2STIX dataset.
    
    Args:
        data_path: Path to directory containing aia_processed_*.npy files
        vis_path: Path to visibility CSV file
        enc_data_path: Path to encoded data directory. If None, returns standard dataset/dataloader.
                      If provided, appends 'train' or 'valid' based on split and returns original_data, visibilities, enc_data
        Other args: Same as other data loader functions
    """
    
    # If enc_data_path is None, behave as before
    if enc_data_path is None:
        transform_1600 = get_default_transforms(target_size=256, channel="1600A")
        dataset = AIA2STIXDataset(
            data_path=data_path,
            vis_path=vis_path,
            transforms=transforms,
            split=split,
            seed=seed,
            train_val_test=train_val_test,
            transform_aia=transform_1600
        )
        
        # sampler
        if distributed:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, seed=0
            )
        else:
            sampler = RandomSamplerSeed(dataset)
        
        # dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_data_workers,
            shuffle=False,  # shuffle determined by the sampler
            sampler=sampler,
            drop_last=False,
            pin_memory=torch.cuda.is_available()
        )

        return dataset, sampler, dataloader
    
    else:
        # Create a combined dataset that returns [original_data, visibilities, encoded_data]
        from pathlib import Path
        
        # Load original dataset for AIA data and visibilities
        transform_1600 = get_default_transforms(target_size=256, channel="1600A")
        base_dataset = AIA2STIXDataset(
            data_path=data_path,
            vis_path=vis_path,
            transforms=transforms,
            split=split,
            seed=seed,
            train_val_test=train_val_test,
            transform_aia=transform_1600
        )
        
        # Determine encoded data path based on split
        if split == "train":
            actual_enc_path = Path(enc_data_path + "train/" + "train")
        elif split == "valid":
            actual_enc_path = Path(enc_data_path + "valid/" + "valid")  
        else:
            print(f'There is no enc data for split={split}') # fallback for other splits
            return None, None, None
        
        # Look for encoded files
        enc_files = sorted(list(actual_enc_path.glob("*.npy")))
        excluded_files = ["valid_encoded_features.npy", "train_encoded_features.npy"]
        enc_files = [f for f in enc_files if f.name not in excluded_files]
        if len(enc_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {actual_enc_path}")
        
        # Load encoded data
        enc_data_list = []
        for file_path in enc_files:
            data = np.load(file_path).reshape(24, 2)
            enc_data_list.append(data)
        
        dataset = CombinedAIA2STIXDataset(base_dataset, enc_data_list)
        
        # sampler
        if distributed:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, seed=0
            )
        else:
            sampler = RandomSamplerSeed(dataset)
        
        # dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_data_workers,
            shuffle=False,  # shuffle determined by the sampler
            sampler=sampler,
            drop_last=True,
            pin_memory=torch.cuda.is_available()
        )

        return dataset, sampler, dataloader

