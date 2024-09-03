import os
import shutil
import numpy as np

def create_directories(base_dir, splits=['train', 'validation', 'test']):
    """Create directories for train, validation, and test splits."""
    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
            print(f"Created directory: {split_path}")

def split_dataset(src_labels, src_images, dest_base, splits=['train', 'validation', 'test'], split_ratio=[0.7, 0.2, 0.1]):
    """Distribute .nii files into train, validation, and test directories."""
    all_files = sorted([f for f in os.listdir(src_labels) if f.endswith('.nii')])
    np.random.shuffle(all_files)  # Shuffle files for random distribution

    # Calculate split sizes
    num_files = len(all_files)
    split_sizes = [int(num_files * ratio) for ratio in split_ratio]

    # Determine file ranges for each split
    split_ranges = {
        'train': (0, split_sizes[0]),
        'validation': (split_sizes[0], split_sizes[0] + split_sizes[1]),
        'test': (split_sizes[0] + split_sizes[1], num_files)
    }

    missing_labels = []
    missing_images = []

    for split, (start, end) in split_ranges.items():
        create_directories(os.path.join(dest_base, 'labels', split))
        create_directories(os.path.join(dest_base, 'images', split))
        for file in all_files[start:end]:
            src_label_file = os.path.join(src_labels, file)
            src_image_file = os.path.join(src_images, file)
            dest_label_file = os.path.join(dest_base, 'labels', split, file)
            dest_image_file = os.path.join(dest_base, 'images', split, file)
            
            if os.path.exists(src_label_file):
                shutil.move(src_label_file, dest_label_file)
                print(f"Moved {file} to {os.path.join(dest_base, 'labels', split)}")
            else:
                missing_labels.append(file)

            if os.path.exists(src_image_file):
                shutil.move(src_image_file, dest_image_file)
                print(f"Moved {file} to {os.path.join(dest_base, 'images', split)}")
            else:
                missing_images.append(file)

    if missing_labels:
        print("\nMissing label files:")
        for file in missing_labels:
            print(f"  {file}")

    if missing_images:
        print("\nMissing image files:")
        for file in missing_images:
            print(f"  {file}")

if __name__ == "__main__":
    # Define paths
    base_dir = r"C:\Users\Dell\OneDrive\Desktop\data"
    src_labels = os.path.join(base_dir, 'labels')
    src_images = os.path.join(base_dir, 'images')
    dest_base = base_dir

    # Create directories for train, validation, and test splits
    create_directories(os.path.join(dest_base, 'labels'))
    create_directories(os.path.join(dest_base, 'images'))
    split_dataset(src_labels, src_images, dest_base)
