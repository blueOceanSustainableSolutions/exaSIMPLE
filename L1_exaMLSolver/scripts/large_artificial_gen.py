import numpy as np
import scipy.sparse as sp
import os


def generate_simplified_matrices(num_matrices, matrix_size):
    matrices = []

    def generate_matrix(matrix_size):
        # Generate a sparse random matrix in LIL format for efficient modification
        A = sp.random(matrix_size, matrix_size, density=0.005, format='lil', random_state=42)

        # Make the matrix symmetric by averaging with its transpose
        A = (A + A.T) * 0.5

        # Make the matrix strongly diagonally dominant
        for i in range(matrix_size):
            row_sum = np.abs(A[i, :]).sum() - np.abs(A[i, i])  # Sum of off-diagonal elements in the row
            A[i, i] = row_sum + 1  # Set diagonal element to ensure dominance

        # Convert to COO format for extracting indices and values after all modifications
        A = A.tocoo()
        A_indices = np.vstack((A.row, A.col))
        A_values = np.clip(A.data, -1e3, 1e3)  # Limit values for stability

        # Generate a solution vector x and calculate b
        x = np.random.randn(matrix_size)
        b = A.dot(x)

        return {'A_indices': A_indices, 'A_values': A_values, 'x': x, 'b': b}

    # Generate specified number of matrices
    for _ in range(num_matrices):
        matrices.append(generate_matrix(matrix_size))

    return matrices  # Ensure the list is returned


# Generate simplified 20x20 matrices for training and testing
train_matrices = generate_simplified_matrices(100000, 20)
test_matrices = generate_simplified_matrices(15000, 20)

# # Check for NaN or extreme
# for i, matrix_data in enumerate(train_matrices):
#     if np.any(np.isnan(matrix_data['A_values'])) or np.any(np.isnan(matrix_data['b'])):
#         print(f"Matrix {i} contains NaNs")
#         continue
#     if np.any(np.abs(matrix_data['A_values']) > 1e3) or np.any(np.abs(matrix_data['b']) > 1e3):
#         print(f"Matrix {i} contains extreme values and may cause instability.")

# # Check for NaN or extreme values
# for i, matrix_data in enumerate(test_matrices):
#     if np.any(np.isnan(matrix_data['A_values'])) or np.any(np.isnan(matrix_data['b'])):
#         print(f"Matrix {i} contains NaNs")
#         continue
#     if np.any(np.abs(matrix_data['A_values']) > 1e3) or np.any(np.abs(matrix_data['b']) > 1e3):
#         print(f"Matrix {i} contains extreme values and may cause instability.")

# Define the folder paths
train_folder = "<train_folder_dir>"
test_folder = "<test_folder_dir>"

# Create directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Function to save matrices as npz files
def save_matrices_to_folder(matrices, folder_path):
    for i, matrix_data in enumerate(matrices):
        file_number = 100001 + i
        file_path = os.path.join(folder_path, f"matrix_{file_number}.npz")
        np.savez(file_path,
                 A_indices=matrix_data['A_indices'],
                 A_values=matrix_data['A_values'],
                 x=matrix_data['x'],
                 b=matrix_data['b'])


# Save the training and testing matrices
save_matrices_to_folder(train_matrices, train_folder)
save_matrices_to_folder(test_matrices, test_folder)

# Output paths for confirmation
print("Train folder path:", train_folder)
print("Test folder path:", test_folder)

# Show first item in each set as a sample
print("Sample training matrix:", train_matrices[0])
print("Sample testing matrix:", test_matrices[0])
