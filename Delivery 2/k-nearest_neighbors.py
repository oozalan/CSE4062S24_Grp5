import pandas as pd
import numpy as np

# Load the Excel file containing combined k-nearest neighbors
combined_data_path = 'combined_k_nearest_neighbors.xlsx'  # Update the file path
combined_data = pd.read_excel(combined_data_path)

# Create a dictionary to store recommended EPS values
eps_values = {}

# Process each column
for column in combined_data.columns:
    # Take only numerical values (excluding Distance values)
    if column != 'Unnamed: 0':
        # Calculate the recommended EPS value (as the mean of distance values)
        eps = np.mean(combined_data[column])
        eps_values[column] = eps

# Create a DataFrame containing EPS values
eps_df = pd.DataFrame(list(eps_values.items()), columns=['Column Name', 'EPS Value'])

# Write EPS values to an Excel file
eps_excel_path = 'recommended_eps_values.xlsx'  # Path to the Excel file to be saved
eps_df.to_excel(eps_excel_path, index=False)

print("Recommended EPS values successfully calculated and written to an Excel file.")
