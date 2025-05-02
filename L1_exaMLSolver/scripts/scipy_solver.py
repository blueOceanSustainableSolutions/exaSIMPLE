import numpy as np
import scipy.linalg
import os

# model predictions on reduced test set
predictions_folder = '<x_vector_predictions_dir>'
batch_files = [f for f in os.listdir(predictions_folder) if f.endswith('.npz')]

# init vars to accumulate RMSE, max difference, and residuals
total_scipy_rmse = 0
total_scipy_residual = 0
total_model_vs_scipy_rmse = 0
total_normalised_scipy_residual_norm = 0
total_normalised_scipy_residual_diag = 0
num_batches = len(batch_files)

# iterate over each batch file & compute the metrics
for batch_file in batch_files:
    batch_file_path = os.path.join(predictions_folder, batch_file)

    batch_data = np.load(batch_file_path)

    # extract A_indices, A_values, model-generated x, and b
    A_indices = batch_data['A_indices']
    A_values = batch_data['A_values']
    # x_model = batch_data['x_model']  # model-generated x (y_direction)
    b = batch_data['b']  # target vector b

    # reconstruct sparse matrix A using scipy.sparse
    A_shape = (b.size, b.size)
    A = scipy.sparse.coo_matrix((A_values, (A_indices[0], A_indices[1])), shape=A_shape)

    # convert A to dense for inversion
    A_dense = A.todense()

    # solve for x using Scipy
    x_scipy = scipy.linalg.solve(A_dense, b)

    # convert result to NumPy array
    b_scipy_predicted = np.asarray(A_dense @ x_scipy)  # A * x_scipy

    # calculate RMSE for Scipy-computed x (how well Scipy satisfies A*x = b)
    scipy_rmse = np.sqrt(np.mean((b_scipy_predicted - b) ** 2))  # RMSE of residual

    # compute residual for Scipy-computed x
    scipy_residual = np.linalg.norm(b_scipy_predicted - b)

    # normalising residuals
    norm_A = np.linalg.norm(A_dense)
    max_diag_A = np.max(np.diag(A_dense))

    normalised_scipy_residual_norm = scipy_residual / norm_A
    normalised_scipy_residual_diag = scipy_residual / max_diag_A

    # compute the RMSE between model-generated x and Scipy-computed x
    # x_diff = x_model - x_scipy
    # model_vs_scipy_rmse = np.sqrt(np.mean(x_diff ** 2))

    # accumulate the metrics
    total_scipy_rmse += scipy_rmse
    total_scipy_residual += scipy_residual
    # total_model_vs_scipy_rmse += model_vs_scipy_rmse
    total_normalised_scipy_residual_norm += normalised_scipy_residual_norm
    total_normalised_scipy_residual_diag += normalised_scipy_residual_diag

    # individual results for each batch
    print(f"Results for {batch_file}:")
    print(f"  Scipy RMSE (A*x_scipy - b): {scipy_rmse}")
    print(f"  Scipy residual: {scipy_residual}")
    # print(f"  RMSE between model-generated x and Scipy-computed x: {model_vs_scipy_rmse}")
    print("\n")

# compute average RMSE, residual for Scipy, and RMSE between model vs Scipy
avg_scipy_rmse = total_scipy_rmse / num_batches
avg_scipy_residual = total_scipy_residual / num_batches
# avg_model_vs_scipy_rmse = total_model_vs_scipy_rmse / num_batches

# display averages
print(f"Scipy RMSE over all batches: {avg_scipy_rmse}")
print(f"Scipy RMSE residual over all batches: {avg_scipy_residual}")
# print(f"RMSE between model-generated x and Scipy-computed x over all batches: {avg_model_vs_scipy_rmse}")
print(f"Total norm A {total_normalised_scipy_residual_norm}")
print(f"Total diag A {total_normalised_scipy_residual_diag}")
