import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np

# Veri setini yükle
data_path = 'data.csv'
data = pd.read_csv(data_path)

# EPS değerlerini içeren Excel dosyasını yükle
eps_values_path = 'recommended_eps_values.xlsx'
eps_values_df = pd.read_excel(eps_values_path)

# İlgilendiğimiz sütunlar (55 ve 56. sütunlar, 0-tabanlı indeksleme ile 54 ve 55)
columns = [data.columns[55], data.columns[56]]

# Sonuçları yazmak için bir metin dosyası oluştur
with open('dbscan_cluster_stats_results.txt', 'w') as output_file:
    # Her bir sütun için işlem yap
    for column in columns:
        # EPS değerini al
        if column in eps_values_df['Column Name'].values:
            eps = eps_values_df.loc[eps_values_df['Column Name'] == column, 'EPS Value'].iloc[0]
        else:
            eps = 0  # Eğer EPS değeri tabloda yoksa, default olarak 0 kullan

        # EPS değeri 0 değilse DBSCAN algoritmasını uygula
        if eps != 0:
            # Sütun verilerini al ve NaN değerlerini temizle
            column_data = data[[column]].dropna()

            # Veriyi ölçeklendir (StandardScaler kullanarak)
            scaler = StandardScaler()
            scaled_column_data = scaler.fit_transform(column_data)

            # DBSCAN algoritmasını uygula
            dbscan = DBSCAN(eps=eps, min_samples=5, metric='euclidean')
            cluster_labels = dbscan.fit_predict(scaled_column_data)

            # Her bir küme için istatistiksel hesaplamaları yap
            unique_clusters = np.unique(cluster_labels)
            for cluster in unique_clusters:
                cluster_data = column_data[cluster_labels == cluster]

                # Küme içi istatistiksel hesaplamalar
                cluster_mean = np.mean(cluster_data)
                cluster_variance = np.var(cluster_data)
                cluster_z_scores = (cluster_data - cluster_mean) / np.std(cluster_data)

                output_file.write(f"{column} Cluster {cluster} Statistics:\n")
                output_file.write(f"Mean: {cluster_mean}\n")


                output_file.write("\n")
        else:
            output_file.write(f"{column} Cluster Labels: None, No clustering performed due to EPS=0\n")

print("DBSCAN ile küme istatistikleri başarıyla hesaplandı ve 'dbscan_cluster_stats_results.txt' dosyasına yazıldı.")
import matplotlib.pyplot as plt
import seaborn as sns

# Sonuçları görselleştirme fonksiyonu
def visualize_clusters(data, labels, column):
    plt.figure(figsize=(12, 6))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(f'Scatter Plot for {column}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # Density plot
    plt.subplot(1, 2, 2)
    sns.kdeplot(x=data[:, 0], y=data[:, 1], cmap="Reds", shade=True, bw_adjust=.5)
    plt.title(f'Density Plot for {column}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    plt.tight_layout()
    plt.show()

# EPS değeri 0 olmayan sütunlar için görselleştirmeyi çağır
for column in columns:
    if column in eps_values_df['Column Name'].values:
        eps = eps_values_df.loc[eps_values_df['Column Name'] == column, 'EPS Value'].iloc[0]
        if eps != 0:
            column_data = data[[column]].dropna()
            scaler = StandardScaler()
            scaled_column_data = scaler.fit_transform(column_data)
            dbscan = DBSCAN(eps=eps, min_samples=5, metric='euclidean')
            cluster_labels = dbscan.fit_predict(scaled_column_data)
            visualize_clusters(scaled_column_data, cluster_labels, column)
