import numpy as np

data_path = '<data_path.npz>'

loaded_data = np.load(data_path)

loaded_data_keys = list(loaded_data.keys())

# check shapes & types of the contents
A_indices = loaded_data['A_indices']
A_values = loaded_data['A_values']
x = loaded_data['x']
b = loaded_data['b']

dicti = {
    "A_indices_shape": A_indices.shape,
    "A_values_shape": A_values.shape,
    "x_shape": x.shape,
    "b_shape": b.shape,
    "A_indices": A_indices[:, :5],
    "A_values": A_values[:5],
    "x": x[:5],
    "b": b[:5],
}

print(dicti)