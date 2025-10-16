import numpy as np
import random as rd

def generate_instance(size):
       # Generate random normal coefficients
       coefficients = np.random.normal(size=(size, size))

       # Set diagonal elements to zero
       np.fill_diagonal(coefficients, 0)

       # Generate symmetric matrix
       M = (coefficients + coefficients.T) / 2

       H = np.zeros(size)

       return M, H