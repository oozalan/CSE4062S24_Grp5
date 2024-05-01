import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
data = pd.read_excel('combined_k_nearest_neighbors.xlsx')

# Use matplotlib for plotting
plt.figure(figsize=(12, 6))

# Draw a line graph for Quick Assets/Total Assets
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
plt.plot(data[' Quick Assets/Total Assets'], color='blue')
plt.title('Line Plot of Quick Assets/Total Assets')
plt.xlabel('Index')
plt.ylabel('Value')

# Draw a line graph for Current Assets/Total Assets
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
plt.plot(data[' Current Assets/Total Assets'], color='green')
plt.title('Line Plot of Current Assets/Total Assets')
plt.xlabel('Index')
plt.ylabel('Value')

# Display the plots
plt.tight_layout()
plt.show()
