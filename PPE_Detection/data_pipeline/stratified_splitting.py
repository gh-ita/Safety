import os
import shutil
import argparse
import numpy as np
from glob import glob
from sklearn.model_selection import StratifiedShuffleSplit
import yaml
import json

def get_label_class(label_path):
    """Extract class information from label file.
    This function needs to be adapted to your specific label format.
    """
    try:
        # For YOLO format (txt files with class id as first number)
        with open(label_path, 'r') as f:
            content = f.read().strip()
            if content:
                # Get the first number from the first line (class id in YOLO format)
                return content.split('\n')[0].split(' ')[0]
            return 'empty'  # For empty label files
    except:
        # Default case if parsing fails
        return 'unknown'

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train, test, and validation sets using stratified sampling.')
    parser.add_argument('--images_dir', required=True, help='Directory containing images')
    parser.add_argument('--labels_dir', required=True, help='Directory containing labels')
    parser.add_argument('--output_dir', default='split_dataset', help='Output directory for split datasets')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test data')
    parser.add_argument('--image_ext', default='.jpg', help='Image file extension')
    parser.add_argument('--label_ext', default='.txt', help='Label file extension')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    if args.train_ratio + args.test_ratio >= 1.0:
        print("Error: train_ratio + test_ratio should be less than 1.0 to leave room for validation data")
        return
        
    # Create output directories
    for split in ['train', 'test', 'valid']:
        for data_type in ['images', 'labels']:
            create_directory(os.path.join(args.output_dir, split, data_type))
            
    # Get all image files
    image_files = glob(os.path.join(args.images_dir, f'*{args.image_ext}'))
    print(f"Found {len(image_files)} images in {args.images_dir}")
    
    # Create a list to store image names and their corresponding class
    data = []
    valid_pairs = 0
    
    for image_path in image_files:
        # Get filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(args.labels_dir, f"{filename}{args.label_ext}")
        
        # Check if label file exists
        if os.path.exists(label_path):
            # Get class from label file
            label_class = get_label_class(label_path)
            data.append((image_path, label_path, label_class))
            valid_pairs += 1
    
    print(f"Found {valid_pairs} valid image-label pairs")
    
    if valid_pairs == 0:
        print("No valid image-label pairs found. Please check your directories and file extensions.")
        return
    
    # Convert to numpy arrays for stratified splitting
    X = np.array([item[0] for item in data])
    y = np.array([item[2] for item in data])
    label_paths = np.array([item[1] for item in data])
    
    # Get unique classes and their counts
    unique_classes, counts = np.unique(y, return_counts=True)
    print("\nClass distribution in the dataset:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count} samples")
    
    # First split: separate train from (test+valid)
    train_split = StratifiedShuffleSplit(n_splits=1, train_size=args.train_ratio, random_state=args.random_state)
    train_idx, temp_idx = next(train_split.split(X, y))
    
    X_train, y_train = X[train_idx], y[train_idx]
    label_paths_train = label_paths[train_idx]
    
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    label_paths_temp = label_paths[temp_idx]
    
    # Second split: separate test from valid
    # Calculate test_size relative to the temp set
    remaining_ratio = 1.0 - args.train_ratio
    test_size_adjusted = args.test_ratio / remaining_ratio
    
    test_valid_split = StratifiedShuffleSplit(n_splits=1, train_size=test_size_adjusted, random_state=args.random_state)
    test_idx, valid_idx = next(test_valid_split.split(X_temp, y_temp))
    
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    label_paths_test = label_paths_temp[test_idx]
    
    X_valid, y_valid = X_temp[valid_idx], y_temp[valid_idx]
    label_paths_valid = label_paths_temp[valid_idx]
    
    # Copy files to their respective directories
    splits = [
        ('train', X_train, label_paths_train, y_train),
        ('test', X_test, label_paths_test, y_test),
        ('valid', X_valid, label_paths_valid, y_valid)
    ]
    
    # Class distribution tracking
    class_distribution = {split: {cls: 0 for cls in unique_classes} for split in ['train', 'test', 'valid']}
    
    for split_name, img_paths, lbl_paths, classes in splits:
        for img_path, lbl_path, cls in zip(img_paths, lbl_paths, classes):
            # Get filename
            filename = os.path.basename(img_path)
            label_filename = os.path.basename(lbl_path)
            
            # Copy image
            dst_img_path = os.path.join(args.output_dir, split_name, 'images', filename)
            shutil.copy2(img_path, dst_img_path)
            
            # Copy label
            dst_lbl_path = os.path.join(args.output_dir, split_name, 'labels', label_filename)
            shutil.copy2(lbl_path, dst_lbl_path)
            
            # Update class distribution
            class_distribution[split_name][cls] += 1
    
    # Calculate actual split ratios
    total_samples = len(data)
    actual_ratios = {
        'train': len(X_train) / total_samples,
        'test': len(X_test) / total_samples,
        'valid': len(X_valid) / total_samples
    }
    
    # Print summary
    print("\nDataset split completed!")
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(X_train)} ({actual_ratios['train']:.2%})")
    print(f"Test samples: {len(X_test)} ({actual_ratios['test']:.2%})")
    print(f"Validation samples: {len(X_valid)} ({actual_ratios['valid']:.2%})")
    
    print("\nClass distribution in each split:")
    for cls in unique_classes:
        original_count = np.sum(y == cls)
        train_count = class_distribution['train'][cls]
        test_count = class_distribution['test'][cls]
        valid_count = class_distribution['valid'][cls]
        
        print(f"  Class {cls}:")
        print(f"    Original: {original_count} ({original_count/total_samples:.2%})")
        print(f"    Train: {train_count} ({train_count/len(X_train):.2%})")
        print(f"    Test: {test_count} ({test_count/len(X_test):.2%})")
        print(f"    Valid: {valid_count} ({valid_count/len(X_valid):.2%})")
    
    # Save metadata
    metadata = {
        'dataset_info': {
            'total_samples': total_samples,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'valid_samples': len(X_valid),
            'train_ratio': actual_ratios['train'],
            'test_ratio': actual_ratios['test'],
            'valid_ratio': actual_ratios['valid']
        },
        'class_distribution': class_distribution
    }
    
    # Save as both YAML and JSON
    with open(os.path.join(args.output_dir, 'dataset_metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)
    
    with open(os.path.join(args.output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {os.path.join(args.output_dir, 'dataset_metadata.yaml')} and {os.path.join(args.output_dir, 'dataset_metadata.json')}")

if __name__ == "__main__":
    main()