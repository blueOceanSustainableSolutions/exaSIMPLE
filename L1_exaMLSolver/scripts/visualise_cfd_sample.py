from scripts.functions.data_analysis_functions import is_positive_definite, load_ab, sample_shape, matrix_summary, vis_matrix, zoomed_vis_matrix

# sample A
matrix_path = "mass_transport_npz_a/A_mass_transport.npz" # e.g.
vector_path = "mass_transport_npz_a/b_mass_transport.npy" # e.g.

# load as numpy objects
matrix_a, matrix_b = load_ab(matrix_path, vector_path)

# check if matrix is positive definite
print(is_positive_definite(matrix_a))

# check shape
matrix_shape, matrix_nonzeros, vector_length = sample_shape(matrix_a, matrix_b)
print(matrix_shape, matrix_nonzeros, vector_length)

"""
Matrix Shape: 6400 × 6400 matching the expected dimensions.
Number of Non-Zero Elements: 31,600, consistent with the sparsity.
Vector Length: 6,400, matching the number of rows in the matrix.
"""

# print matrix summary
matrix_summary(matrix_a)

# show matrix vis
vis_matrix(matrix_a)

# show zoomed matrix vis
zoomed_vis_matrix(matrix_a, 100)

"""
(6400, 6400) 31600 6400
{
  "Matrix Shape": [
    6400,
    6400
  ],
  "Number of Non-Zero Elements": 31600,
  "Density": 0.000771484375,
  "Sparsity": 0.999228515625,
  "Diagonal Dominance (mean)": 0.8715625
}
"""