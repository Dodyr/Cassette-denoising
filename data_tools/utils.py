def scaled_in(matrix_spec):
    matrix_spec = (matrix_spec / 40.0) + 1.0
    return matrix_spec

def scaled_ou(matrix_spec):
    matrix_spec = (matrix_spec / 40.0) + 1.0
    return matrix_spec

def inv_scaled_in(matrix_spec):
    matrix_spec = (matrix_spec - 1.0) * 40.0
    return matrix_spec

def inv_scaled_ou(matrix_spec):
    matrix_spec = (matrix_spec - 1.0) * 40.0
    return matrix_spec