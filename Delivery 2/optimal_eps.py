import pandas as pd
import numpy as np

# Birleştirilmiş k-en yakın komşuların olduğu Excel dosyasını yükle
combined_data_path = 'combined_k_nearest_neighbors.xlsx'  # Dosya yolunu güncelleyin
combined_data = pd.read_excel(combined_data_path)

# Önerilen EPS değerlerini depolamak için bir sözlük oluştur
eps_values = {}

# Her bir kolon için işlem yap
for column in combined_data.columns:
    # Sadece sayısal değerleri al (Mesafe değerleri hariç)
    if column != 'Unnamed: 0':
        # Önerilen EPS değerini hesapla (Ortalama mesafe değeri olarak)
        eps = np.mean(combined_data[column])
        eps_values[column] = eps

# EPS değerlerini içeren bir DataFrame oluştur
eps_df = pd.DataFrame(list(eps_values.items()), columns=['Column Name', 'EPS Value'])

# EPS değerlerini bir Excel dosyasına yaz
eps_excel_path = 'recommended_eps_values.xlsx'  # Kaydedilecek Excel dosyasının yolu
eps_df.to_excel(eps_excel_path, index=False)

print("Önerilen EPS değerleri başarıyla hesaplandı ve Excel dosyasına yazıldı.")

