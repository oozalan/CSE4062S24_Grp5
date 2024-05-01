import pandas as pd
import matplotlib.pyplot as plt

# Excel dosyasını yükle
data = pd.read_excel('combined_k_nearest_neighbors.xlsx')

# Grafik çizimi için matplotlib kullan
plt.figure(figsize=(12, 6))

# Quick Assets/Total Assets için çizgi grafiği çiz
plt.subplot(1, 2, 1)  # 1 satır, 2 sütun, ilk grafik
plt.plot(data[' Quick Assets/Total Assets'], color='blue')
plt.title('Line Plot of Quick Assets/Total Assets')
plt.xlabel('Index')
plt.ylabel('Value')

# Current Assets/Total Assets için çizgi grafiği çiz
plt.subplot(1, 2, 2)  # 1 satır, 2 sütun, ikinci grafik
plt.plot(data[' Current Assets/Total Assets'], color='green')
plt.title('Line Plot of Current Assets/Total Assets')
plt.xlabel('Index')
plt.ylabel('Value')

# Grafikleri göster
plt.tight_layout()
plt.show()
