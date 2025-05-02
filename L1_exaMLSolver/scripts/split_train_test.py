import os
import random
from concurrent.futures import ThreadPoolExecutor

# Define paths
processed_dir = "<data_folder_to_split>"
output_dir = "<output_dir_split_symbolic_links>"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Create output directories if they don’t exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all files in the processed directory
all_files = [f for f in os.listdir(processed_dir) if f.endswith(".npz")]

# Define split ratio
test_ratio = 0.15  # 15% for test set
test_size = int(len(all_files) * test_ratio)

# Randomly select files for the test set
test_files = set(random.sample(all_files, test_size))
train_files = [f for f in all_files if f not in test_files]

print(f"Total files: {len(all_files)}")
print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")

def batch_create_symlink(files, source_dir, target_dir, batch_size=1000):
    files = list(files)  # Convert set to list for indexing
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        for file_name in batch_files:
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            try:
                os.symlink(source_path, target_path)
            except FileExistsError:
                print(f"Symbolic link already exists for {file_name}")
            except Exception as e:
                print(f"Failed to create symbolic link for {file_name}. Error: {e}")


# Parallelized symlink creation
def parallel_symlink(files, source_dir, target_dir):
    with ThreadPoolExecutor() as executor:
        executor.map(lambda f: os.symlink(os.path.join(source_dir, f), os.path.join(target_dir, f)), files)

# Create symbolic links for train and test directories
print("Creating symbolic links for training files...")
batch_create_symlink(train_files, processed_dir, train_dir, batch_size=1000)
print("Creating symbolic links for testing files...")
batch_create_symlink(test_files, processed_dir, test_dir, batch_size=1000)

print("Dataset split and symbolic linking completed!")
print("Output dir: ")
print(output_dir)


# Verify train and test sets are disjoint
assert set(train_files).isdisjoint(test_files), "Train and Test sets overlap!"