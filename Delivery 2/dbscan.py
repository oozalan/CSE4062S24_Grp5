import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

# Load the dataset
data_path = 'data.csv'
data = pd.read_csv(data_path)

# Load the Excel file containing EPS values
eps_values_path = 'recommended_eps_values.xlsx'
eps_values_df = pd.read_excel(eps_values_path)

# Columns of interest (55th and 56th columns, 0-based indexing 54 and 55)
columns = [data.columns[55], data.columns[56]]

# Create a text file to write the results
with open('dbscan_cluster_stats_results.txt', 'w') as output_file:
    # Process each column
    for column in columns:
        # Get the EPS value
        if column in eps_values_df['Column Name'].values:
            eps = eps_values_df.loc[eps_values_df['Column Name'] == column, 'EPS Value'].iloc[0]
        else:
            eps = 0  # If there is no EPS value in the table, use default 0

        # Apply DBSCAN algorithm if EPS is not 0
        if eps != 0:
            # Get column data and clean NaN values
            column_data = data[[column]].dropna()

            # Scale the data (using StandardScaler)
            scaler = StandardScaler()
            scaled_column_data = scaler.fit_transform(column_data)

            # Apply the DBSCAN algorithm
            dbscan = DBSCAN(eps=eps, min_samples=5, metric='euclidean')
            cluster_labels = dbscan.fit_predict(scaled_column_data)

            # Perform statistical calculations for each cluster
            unique_clusters = np.unique(cluster_labels)
            for cluster in unique_clusters:
                cluster_data = column_data[cluster_labels == cluster]

                # Intra-cluster statistical calculations
                cluster_mean = np.mean(cluster_data[column])
                cluster_variance = np.var(cluster_data[column], ddof=1)
                cluster_std_dev = np.std(cluster_data[column], ddof=1)

                output_file.write(f"{column} Cluster {cluster} Statistics:\n")
                output_file.write(f"Mean: {cluster_mean}\n")
                output_file.write(f"Variance: {cluster_variance}\n")
                output_file.write(f"Standard Deviation: {cluster_std_dev}\n")
                output_file.write("\n")
        else:
            output_file.write(f"{column} Cluster Labels: None, No clustering performed due to EPS=0\n")

print("Cluster statistics with DBSCAN were successfully calculated and written to 'dbscan_cluster_stats_results.txt'.")
