from scipy.sparse import load_npz
import numpy as np
import matplotlib.pyplot as plt
import json
import torch


def load_ab(matrix_path, vector_path):
    """
    Takes a matrix A and vector B and 
    loads them as numpy objects
    :param matrix_path: 
    :param vector_path: 
    :return: 
    """
    # load sample
    matrix_path = matrix_path
    vector_path = vector_path

    # Load the sparse matrix and the vector
    matrix_a = load_npz(matrix_path)

    vector_b = np.load(vector_path)
    return matrix_a, vector_b


def sample_shape(matrix_a, vector_b):
    """
    Checks shapes of passed matrix & vector
    :param matrix_a:
    :param vector_b:
    :return:
    """

    # Check the properties of the matrix and vector
    matrix_shape = matrix_a.shape
    matrix_nonzeros = matrix_a.nnz

    vector_length = vector_b.shape[0]
    return matrix_shape, matrix_nonzeros, vector_length


def matrix_summary(matrix_a):
    """
    Prints out a JSON dict of summary statistics
    :param matrix_a:
    :param vector_b:
    :return:
    """
    # Compute sparsity and density
    total_elements = matrix_a.shape[0] * matrix_a.shape[1]
    density = matrix_a.nnz / total_elements
    sparsity = 1 - density
    
    # Check diagonal dominance
    diagonal = matrix_a.diagonal()
    row_sums = np.abs(matrix_a).sum(axis=1).A1  # Convert to dense array for row sum
    diagonal_dominance = np.mean(np.abs(diagonal) >= (row_sums - np.abs(diagonal)))

    # Summary statistics
    summary = {
        "Matrix Shape": matrix_a.shape,
        "Number of Non-Zero Elements": matrix_a.nnz,
        "Density": density,
        "Sparsity": sparsity,
        "Diagonal Dominance (mean)": diagonal_dominance,
    }

    print(json.dumps(summary, indent=2))
    

def vis_matrix(matrix_a):
    """
    Visualize the sparsity pattern
    :param matrix_a:
    :return:
    """
    plt.figure(figsize=(10, 10))
    plt.spy(matrix_a, markersize=4)
    plt.title("Sparsity Pattern of matrix_a")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


def zoomed_vis_matrix(matrix_a, zoom):
    """
    Visualize the sparsity pattern (ZOOMED)
    :param matrix_a:
    :return:
    """
    # Convert the sparse matrix to CSR format for slicing
    if not hasattr(matrix_a, 'tocsr'):
        raise TypeError("Input matrix must be convertible to CSR format.")
    csr_matrix = matrix_a.tocsr()

    # Visualize a zoomed-in section of the sparsity pattern
    zoom_range = zoom  # Zoom into the top-left zoom x zoom section
    plt.figure(figsize=(8, 8))
    plt.spy(csr_matrix[:zoom_range, :zoom_range], markersize=5)
    plt.title(f"Sparsity Pattern (Top-Left {zoom_range}x{zoom_range})")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


def is_positive_definite(matrix):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=torch.float32)
    try:
        torch.linalg.cholesky(matrix)
        return True
    except torch.linalg.LinAlgError:
        return False



