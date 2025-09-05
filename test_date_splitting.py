#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for date-based splitting functionality with command-line arguments.

This script demonstrates how to use date ranges to control train/valid/test splits
based on flare timestamps instead of random percentage splits.

Usage:
    python test_date_splitting.py --train-dates 210520 210630 --valid-dates 210701 210730 --test-dates 210801 210831
    python test_date_splitting.py --train-dates 20210520 20210630 --valid-dates 20210701 20210730 --test-dates 20210801 20210831

@author: francesco
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.data.dataset import get_aia2stix_data_objects
import torch


def parse_arguments():
    """Parse command-line arguments for date ranges."""
    parser = argparse.ArgumentParser(
        description='Test date-based splitting for AIA2STIX dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using YYMMDD format:
  python %(prog)s --train-dates 210520 210630 --valid-dates 210701 210730 --test-dates 210801 210831
  
  # Using YYYYMMDD format:
  python %(prog)s --train-dates 20210520 20210630 --valid-dates 20210701 20210730 --test-dates 20210801 20210831
  
  # Test only specific splits:
  python %(prog)s --train-dates 210520 210630 --test-only
        """
    )
    
    # Date range arguments
    parser.add_argument('--train-dates', type=str, nargs=2, metavar=('START', 'END'),
                       help='Train split date range (format: YYMMDD or YYYYMMDD)')
    parser.add_argument('--valid-dates', type=str, nargs=2, metavar=('START', 'END'),
                       help='Validation split date range (format: YYMMDD or YYYYMMDD)')
    parser.add_argument('--test-dates', type=str, nargs=2, metavar=('START', 'END'),
                       help='Test split date range (format: YYMMDD or YYYYMMDD)')
    
    # Data paths
    parser.add_argument('--vis-path', type=str, 
                       default="/mnt/nas05/data01/francesco/AIA2STIX/Flarelist_visibilites.csv",
                       help='Path to visibility CSV file')
    parser.add_argument('--data-path', type=str,
                       default="/mnt/nas05/astrodata01/aia_2_stix/prepro_data_20250731_210359/processed_images",
                       help='Path to processed images directory')
    parser.add_argument('--augmented-path', type=str,
                       default="./test_augmented_dataset",
                       help='Path to augmented dataset directory')
    
    # Test options
    parser.add_argument('--test-augmented', action='store_true',
                       help='Test augmented dataset splitting')
    parser.add_argument('--test-comparison', action='store_true',
                       help='Compare percentage vs date-based splitting')
    parser.add_argument('--train-only', action='store_true',
                       help='Test only train split')
    parser.add_argument('--valid-only', action='store_true',
                       help='Test only validation split')
    parser.add_argument('--test-only', action='store_true',
                       help='Test only test split')
    
    # Other options
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for data loading')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def validate_date_format(date_str):
    """Validate and normalize date format."""
    if len(date_str) == 6:  # YYMMDD
        return date_str
    elif len(date_str) == 8:  # YYYYMMDD
        return date_str[2:]  # Convert to YYMMDD
    else:
        raise ValueError(f"Invalid date format: {date_str}. Use YYMMDD or YYYYMMDD")


def build_date_ranges(args):
    """Build date_ranges dictionary from command-line arguments."""
    date_ranges = {}
    
    if args.train_dates:
        start, end = args.train_dates
        date_ranges['train'] = (validate_date_format(start), validate_date_format(end))
    
    if args.valid_dates:
        start, end = args.valid_dates
        date_ranges['valid'] = (validate_date_format(start), validate_date_format(end))
    
    if args.test_dates:
        start, end = args.test_dates
        date_ranges['test'] = (validate_date_format(start), validate_date_format(end))
    
    return date_ranges if date_ranges else None


def test_date_based_splitting(args, date_ranges):
    """Test date-based splitting with provided date ranges."""
    print("=" * 60)
    print("TESTING DATE-BASED SPLITTING")
    print("=" * 60)
    
    if not date_ranges:
        print("No date ranges provided. Use --train-dates, --valid-dates, and/or --test-dates")
        return False
    
    print("Date ranges defined:")
    for split, (start, end) in date_ranges.items():
        print(f"  {split}: {start} to {end}")
    print()
    
    # Determine which splits to test
    splits_to_test = []
    if args.train_only:
        splits_to_test = ['train'] if 'train' in date_ranges else []
    elif args.valid_only:
        splits_to_test = ['valid'] if 'valid' in date_ranges else []
    elif args.test_only:
        splits_to_test = ['test'] if 'test' in date_ranges else []
    else:
        splits_to_test = list(date_ranges.keys())
    
    if not splits_to_test:
        print("No valid splits to test with provided arguments.")
        return False
    
    # Test each split
    for split_name in splits_to_test:
        print(f"Loading {split_name} split...")
        
        try:
            dataset, _, dataloader = get_aia2stix_data_objects(
                vis_path=args.vis_path,
                data_path=args.data_path,
                batch_size=args.batch_size,
                distributed=False,
                num_data_workers=0,
                split=split_name,
                seed=args.seed,
                use_augmented=False,
                date_ranges=date_ranges
            )
            
            print(f"  ✓ {split_name} dataset size: {len(dataset)}")
            
            # Test loading a batch
            if len(dataset) > 0:
                batch = next(iter(dataloader))
                images, visibilities = batch
                print(f"  ✓ Batch shape: Images {images.shape}, Visibilities {visibilities.shape}")
            else:
                print(f"  ⚠ No data found for {split_name} split with specified date range")
            
        except Exception as e:
            print(f"  ✗ Error loading {split_name}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    return True


def test_augmented_date_splitting(args, date_ranges):
    """Test date-based splitting with augmented dataset."""
    print("=" * 60)
    print("TESTING AUGMENTED DATE-BASED SPLITTING")
    print("=" * 60)
    
    # Check if augmented dataset exists
    augmented_path = Path(args.augmented_path)
    if not augmented_path.exists():
        print(f"⚠  Augmented dataset not found at {augmented_path}")
        print("   Skipping augmented date splitting test")
        return False
    
    if not date_ranges:
        print("No date ranges provided for augmented testing.")
        return False
    
    print("Testing augmented dataset with date ranges:")
    for split, (start, end) in date_ranges.items():
        print(f"  {split}: {start} to {end}")
    print()
    
    # Test train split with augmented data
    try:
        dataset, _, dataloader = get_aia2stix_data_objects(
            vis_path=None,  # Not needed for augmented
            data_path=args.augmented_path,
            batch_size=args.batch_size,
            distributed=False,
            num_data_workers=0,
            split='train',
            seed=args.seed,
            use_augmented=True,
            date_ranges=date_ranges
        )
        
        print(f"✓ Augmented train dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            batch = next(iter(dataloader))
            images, visibilities = batch
            print(f"✓ Batch shape: Images {images.shape}, Visibilities {visibilities.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with augmented date splitting: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_percentage_vs_date_splitting(args, date_ranges):
    """Compare percentage-based vs date-based splitting."""
    print("=" * 60)
    print("COMPARING PERCENTAGE VS DATE-BASED SPLITTING")
    print("=" * 60)
    
    # Test percentage-based (original)
    print("1. Percentage-based splitting (original method):")
    try:
        dataset_pct, _, _ = get_aia2stix_data_objects(
            vis_path=args.vis_path,
            data_path=args.data_path,
            batch_size=args.batch_size,
            distributed=False,
            num_data_workers=0,
            split='train',
            seed=args.seed,
            use_augmented=False,
            date_ranges=None  # Use percentage-based
        )
        print(f"   Train size with percentage split: {len(dataset_pct)}")
    except Exception as e:
        print(f"   Error with percentage splitting: {e}")
    
    # Test date-based
    if date_ranges and 'train' in date_ranges:
        print("\n2. Date-based splitting:")
        try:
            dataset_date, _, _ = get_aia2stix_data_objects(
                vis_path=args.vis_path,
                data_path=args.data_path,
                batch_size=args.batch_size,
                distributed=False,
                num_data_workers=0,
                split='train',
                seed=args.seed,
                use_augmented=False,
                date_ranges=date_ranges  # Use date-based
            )
            print(f"   Train size with date-based split: {len(dataset_date)}")
        except Exception as e:
            print(f"   Error with date splitting: {e}")
    else:
        print("\n2. Date-based splitting: No train date range provided")
    
    return True


def main():
    args = parse_arguments()
    
    print("AIA2STIX DATE-BASED SPLITTING TEST")
    print("This script tests the new date-based splitting functionality")
    print()
    
    # Build date ranges from arguments
    try:
        date_ranges = build_date_ranges(args)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    if not date_ranges:
        print("No date ranges provided. Please specify at least one of:")
        print("  --train-dates START END")
        print("  --valid-dates START END") 
        print("  --test-dates START END")
        print("\nUse --help for more information.")
        return False
    
    # Run tests based on arguments
    success = True
    
    # Always test basic date splitting
    success &= test_date_based_splitting(args, date_ranges)
    
    # Optional tests
    if args.test_augmented:
        success &= test_augmented_date_splitting(args, date_ranges)
    
    if args.test_comparison:
        success &= test_percentage_vs_date_splitting(args, date_ranges)
    
    # Print usage examples
    print("=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    print("# Basic date-based splitting:")
    print("python test_date_splitting.py --train-dates 210520 210630 --valid-dates 210701 210730")
    print()
    print("# Test all splits:")
    print("python test_date_splitting.py --train-dates 210520 210630 --valid-dates 210701 210730 --test-dates 210801 210831")
    print()
    print("# Test only train split:")
    print("python test_date_splitting.py --train-dates 210520 210630 --train-only")
    print()
    print("# Test with augmented dataset:")
    print("python test_date_splitting.py --train-dates 210520 210630 --test-augmented")
    print()
    print("# Compare splitting methods:")
    print("python test_date_splitting.py --train-dates 210520 210630 --test-comparison")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)