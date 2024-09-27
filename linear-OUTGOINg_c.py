import numpy as np

def initialize_projections(Ntokens, Cz):
    # Initialize 3D tensors for 'a', 'b', and 'g'
    a = np.random.rand(Ntokens, Ntokens, Cz)  # Transformation matrix for 'a'
    b = np.random.rand(Ntokens, Ntokens, Cz)  # Transformation matrix for 'b'
    g = np.random.rand(Ntokens, Ntokens, Cz)  # Gating matrix for 'g'
    return a, b, g

def apply_triangle_update_outgoing(Z, a, b, g):
    Ntokens, _, Cz = Z.shape
    updated_Z = np.zeros_like(Z)

    # Apply the triangle update logic for outgoing edges
    for i in range(Ntokens):
        for j in range(Ntokens):
            update_value = np.zeros(Cz)
            for k in range(Ntokens):
                # Correct element-wise multiplication of projections a and b
                update_value += a[i, k, :] * b[j, k, :]
            # Apply gating after summing all contributions
            updated_Z[i, j, :] = update_value * g[i, j, :]

    return updated_Z

def simple_training_loop_outgoing(Z, epochs=10):
    Ntokens, _, Cz = Z.shape
    a, b, g = initialize_projections(Ntokens, Cz)

    for epoch in range(epochs):
        Z = apply_triangle_update_outgoing(Z, a, b, g)
        # Simulate slight adjustments
        a -= 0.01 * np.random.rand(Ntokens, Ntokens, Cz)
        b -= 0.01 * np.random.rand(Ntokens, Ntokens, Cz)
        g -= 0.01 * np.random.rand(Ntokens, Ntokens, Cz)

    return a, b, g, Z

# Example usage
Ntokens = 5
Cz = 3
Z = np.random.rand(Ntokens, Ntokens, Cz)  # Feature tensor

# 'Train' the model and obtain projections
a, b, g, updated_Z = simple_training_loop_outgoing(Z)

# Print results
print("Projection a:\n", a)
print("Projection b:\n", b)
print("Projection g:\n", g)
print("Updated tensor Z after triangle operations:\n", updated_Z)

