#!/usr/bin/env python3
"""
Gene clustering preprocessing entry point.

Reorders genes based on training-slide expression to match GenAR model expectations.

Usage:
    # Process all datasets
    python src/preprocess/run_clustering.py --all-datasets

    # Process a single dataset
    python src/preprocess/run_clustering.py --dataset PRAD

Author: Assistant
Date: 2024
"""

import argparse
import sys
import os
import logging

# Add project src directory to PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocess.gene_clustering import GeneClusteringProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Gene clustering preprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/preprocess/run_clustering.py --all-datasets
  python src/preprocess/run_clustering.py --dataset PRAD
  python src/preprocess/run_clustering.py --dataset her2st

Notes:
  - A backup gene list is written to unclustered_selected_gene_list.txt.
  - The reordered list overwrites selected_gene_list.txt.
  - Clustering metadata is stored in clustering_info.json.
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['PRAD', 'her2st', 'kidney', 'mouse_brain', 'ccRCC'],
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--all-datasets', 
        action='store_true',
        help='Process all datasets'
    )

    parser.add_argument(
        '--data-root',
        type=str,
        default=os.environ.get('GENAR_DATA_ROOT', './data'),
        help='Root directory containing dataset folders '
             '(default: $GENAR_DATA_ROOT or ./data)',
    )

    parser.add_argument(
        '--h5ad-root',
        type=str,
        default=os.environ.get('GENAR_H5AD_ROOT'),
        help='Root directory containing slide h5ad files '
             '(default: $GENAR_H5AD_ROOT)',
    )
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all_datasets:
        parser.print_help()
        print("\nError: specify --dataset or --all-datasets")
        return 1
    
    # Processor
    processor = GeneClusteringProcessor(
        data_root=args.data_root,
        h5ad_root=args.h5ad_root,
    )
    
    try:
        if args.all_datasets:
            logger.info("Processing all datasets")
            processor.process_all_datasets()
            logger.info("All datasets processed")
            
        elif args.dataset:
            logger.info(f"Processing dataset: {args.dataset}")
            processor.process_dataset(args.dataset)
            logger.info(f"Dataset processed: {args.dataset}")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    print("\nGene clustering preprocessing complete.")
    print("Next steps:")
    print("   1. Review clustering_info.json")
    print("   2. Train models with the updated gene order")
    print("   3. Restore unclustered_selected_gene_list.txt to revert")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
