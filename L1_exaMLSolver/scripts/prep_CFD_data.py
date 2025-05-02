from scipy.sparse import load_npz, coo_matrix
import numpy as np
import os
from scipy.sparse.linalg import spsolve

def calculate_x_and_save_components(input_dir, output_dir):
    """
    Calculates x from A and b, then saves the components as A_indices, A_values, x, and b in a new .npz file.
    :param input_dir: Directory containing .npz files for A and .npy files for b.
    :param output_dir: Directory to save the resulting .npz files.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all matrix A and vector b files
    matrix_files = sorted([f for f in os.listdir(input_dir) if f.startswith("A_") and f.endswith(".npz")])
    vector_files = sorted([f for f in os.listdir(input_dir) if f.startswith("b_") and f.endswith(".npy")])

    # Ensure matching pairs of matrix and vector files
    if len(matrix_files) != len(vector_files):
        raise ValueError(f"Mismatch in the number of matrix and vector files. Found {len(matrix_files)} matrices and {len(vector_files)} vectors.")

    # Process each pair of files
    for matrix_file, vector_file in zip(matrix_files, vector_files):
        try:
            # Load A and b
            matrix_path = os.path.join(input_dir, matrix_file)
            vector_path = os.path.join(input_dir, vector_file)

            # Attempt to load the matrix and vector
            A = load_npz(matrix_path).tocoo()  # Convert to COO format
            b = np.load(vector_path)

            # Ensure dimensions of A and b are compatible
            if A.shape[0] != len(b):
                print(f"Dimension mismatch for {matrix_file} and {vector_file}: A.shape[0] = {A.shape[0]}, len(b) = {len(b)}. Skipping.")
                continue

            # Solve Ax = b
            x = spsolve(A.tocsr(), b)

            # Extract components for saving
            A_indices = np.vstack((A.row, A.col)).astype(np.int32)  # Combine row and column indices
            A_values = A.data.astype(np.float64)  # Non-zero values
            x = x.astype(np.float64)  # Solution vector
            b = b.astype(np.float64)  # Ensure consistent data type for b

            # Generate a unique filename
            base_name = os.path.splitext(matrix_file)[0].replace("A_", "matrix_")
            npz_path = os.path.join(output_dir, f"{base_name}.npz")

            # Save the processed data
            np.savez(npz_path, A_indices=A_indices, A_values=A_values, x=x, b=b)
            print(f"Processed and saved: {npz_path}")

        except Exception as e:
            # Catch and log any exceptions during processing
            print(f"Error processing {matrix_file} and {vector_file}: {e}")

# Directories
input_dir = "<npz_folder_input_dir>"
output_dir = "<processed_matrices_output_dir>"

# Run the processing
calculate_x_and_save_components(input_dir, output_dir)

print(f"All files processed. Results saved in {output_dir}")
