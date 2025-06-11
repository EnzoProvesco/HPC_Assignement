import numpy as np
import csv

def generate_random_matrix_csv(n, filename, min_val=0, max_val=100):
    # Generate an n x n matrix with random integers between min_val and max_val
    matrix = np.random.randint(min_val, max_val + 1, size=(n, n))
    
    # Write the matrix to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(matrix)
    
    print(f"Random {n}x{n} matrix saved to '{filename}'.")

# Example usage:
generate_random_matrix_csv(2000, 'matrix.csv')
