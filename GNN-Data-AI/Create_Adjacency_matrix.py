import pandas as pd
import numpy as np
import sys

def create_normalized_adjacency_matrix(input_file_path, output_file_path):
    # Load the CSV file
    data = pd.read_csv(input_file_path)

    # Extracting the relevant columns for creating the adjacency matrix
    edges = data[['Inlet Node', 'Outlet Node', 'Length (ft)']]

    # Creating a set of all unique nodes
    nodes = set(edges['Inlet Node']).union(set(edges['Outlet Node']))

    # Mapping nodes to indices in the adjacency matrix
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Initializing the adjacency matrix with zeros
    adjacency_matrix = np.zeros((len(nodes), len(nodes)))

    # Filling the adjacency matrix with edge lengths
    for _, row in edges.iterrows():
        i = node_to_index[row['Inlet Node']]
        j = node_to_index[row['Outlet Node']]
        adjacency_matrix[i, j] = row['Length (ft)']

    # Normalizing the distances
    max_distance = np.max(adjacency_matrix[adjacency_matrix > 0])
    normalized_matrix = np.where(adjacency_matrix > 0, adjacency_matrix / max_distance, 0)

    # Save the normalized adjacency matrix to the specified output file
    np.savetxt(output_file_path, normalized_matrix, delimiter=',')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file_path> <output_file_path>")
    else:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
        create_normalized_adjacency_matrix(input_file_path, output_file_path)
        print("Normalized adjacency matrix has been saved to:", output_file_path)
