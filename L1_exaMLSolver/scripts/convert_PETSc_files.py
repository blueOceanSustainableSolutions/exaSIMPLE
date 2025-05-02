from petsc4py import PETSc
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import os
import re

# Function to read a PETSc matrix from a binary file
def read_petsc_matrix_binary(filename):
    """
    Reads a PETSc binary matrix file and converts it to a SciPy CSR matrix.
    :param filename: Path to the PETSc binary file.
    :return: SciPy CSR matrix.
    """
    viewer = PETSc.Viewer().createBinary(filename, mode='r')
    matrix = PETSc.Mat().load(viewer)

    # Extract CSR components
    csr_data = matrix.getValuesCSR()
    indptr, indices, data = csr_data  # Unpack components

    # Construct CSR matrix explicitly
    csr = csr_matrix((data, indices, indptr), shape=matrix.getSize())
    return csr

# Function to read a PETSc vector from a binary file
def read_petsc_vector_binary(filename):
    """
    Reads a PETSc binary vector file and converts it to a NumPy array.
    :param filename: Path to the PETSc binary file.
    :return: NumPy array.
    """
    viewer = PETSc.Viewer().createBinary(filename, mode='r')
    vec = PETSc.Vec().load(viewer)
    return vec.getArray()

def get_next_file_index(output_dir, base_name):
    """
    Gets the next available file index for unique naming.
    Checks existing files in the output directory.
    """
    existing_files = [f for f in os.listdir(output_dir) if f.startswith(base_name)]
    indices = [int(re.search(r"_(\d+)\.", fname).group(1)) for fname in existing_files if re.search(r"_(\d+)\.", fname)]
    return max(indices, default=0) + 1 if indices else 1

# Main function to process all files in a folder
def process_mass_transport_files(input_dir, output_dir):
    """
    Processes all mass transport matrix and vector files in a directory,
    saving them as .npz and .npy files with unique names.
    """
    # Extract grid number from the input directory
    grid_match = re.search(r"grid(\d+)", input_dir)
    grid_number = grid_match.group(1) if grid_match else "unknown"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find matrix and vector files with improved validation
    matrix_files = sorted(
        [f for f in os.listdir(input_dir) if re.match(r'a_mat_massTransport_outerloop_\d+\.dat$', f)]
    )
    vector_files = sorted(
        [f for f in os.listdir(input_dir) if re.match(r'b_vec_massTransport_outerloop_\d+\.dat$', f)]
    )

    if len(matrix_files) != len(vector_files):
        raise ValueError("Mismatch in the number of matrix and vector files.")

    # Enhanced validation for matching file pairs
    for matrix_file, vector_file in zip(matrix_files, vector_files):
        if re.search(r'outerloop_\d+', matrix_file).group(0) != re.search(r'outerloop_\d+', vector_file).group(0):
            raise ValueError(f"Mismatched matrix and vector pair: {matrix_file}, {vector_file}")

        matrix_path = os.path.join(input_dir, matrix_file)
        vector_path = os.path.join(input_dir, vector_file)

        # Try-except block for robustness
        try:
            matrix = read_petsc_matrix_binary(matrix_path)
            vector = read_petsc_vector_binary(vector_path)
        except Exception as e:
            print(f"Error processing files {matrix_file} and {vector_file}: {e}")
            continue

        # Ensure matrix and vector are compatible
        if matrix.shape[0] != len(vector):
            print(f"Dimension mismatch: {matrix.shape[0]} rows in matrix, {len(vector)} elements in vector. Skipping.")
            continue

        # Generate unique filenames
        base_name = re.search(r'outerloop_\d+', matrix_file).group(0)
        next_index = get_next_file_index(output_dir, f"A_grid{grid_number}_{base_name}")
        matrix_output_path = os.path.join(output_dir, f"A_grid{grid_number}_{base_name}_{next_index}.npz")
        vector_output_path = os.path.join(output_dir, f"b_grid{grid_number}_{base_name}_{next_index}.npy")

        # Save files
        save_npz(matrix_output_path, matrix)
        np.save(vector_output_path, vector)

        print(f"Processed: {matrix_file} and {vector_file} -> Saved as {matrix_output_path} and {vector_output_path}")

    print(f"All files processed for grid {grid_number}. Results saved in {output_dir}")

# Input and output directories
input_dir = "PETSc_Files_Python"
output_dir = "FlatPlate_ALL_mass_transport_npz"

# Process files
process_mass_transport_files(input_dir, output_dir)
