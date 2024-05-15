import pandas
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
import xlsxwriter
import matplotlib.pyplot as plt

# Read the dataset
df = pandas.read_csv("Dataset/data.csv")
df.rename(columns=lambda x: x.strip(), inplace=True)
df.drop(columns=["Net Income Flag"], inplace=True)

# Seperate the class label from other features
X = df.drop(columns=["Bankrupt?"])
y = df["Bankrupt?"]

# Calculate the ANOVA F-values
f = f_values = f_classif(X, y)[0]
for i in range(len(f_values)):
    f_values[i] = round(f_values[i], 4)

f_values = dict(zip(X.columns, f_values)) # Create dictionary

# Calculate the mutual information values
mi = mi_values = mutual_info_classif(X, y, random_state=0)
for i in range(len(mi_values)):
    mi_values[i] = round(mi_values[i], 4)

mi_values = dict(zip(X.columns, mi_values)) # Create dictionary

# Pick the best features for each feature selection method
f_features = SelectKBest(f_classif, k=10).fit(X, y).get_feature_names_out()
mi_features = SelectKBest(mutual_info_classif, k=10).fit(X, y).get_feature_names_out()

X_f = X.filter(items=f_features, axis=1)
X_mi = X.filter(items=mi_features, axis=1)

# Write to Excel
# file_paths = [
#     "Excel/feature_selection_values.xlsx",
#     "Excel/features_f.xlsx",
#     "Excel/features_mi.xlsx"
# ]

# for i, elem in enumerate([X, X_f, X_mi]):
#     workbook = xlsxwriter.Workbook(file_paths[i])
#     worksheet = workbook.add_worksheet("Values")

#     worksheet.write(0, 0, "No")
#     worksheet.write(0, 1, "Feature Name")
#     worksheet.write(0, 2, "Type")
#     worksheet.write(0, 3, "ANOVA F-value")
#     worksheet.write(0, 4, "Mutual Information Value")

#     row = 1
#     col = 0
#     for column_name, column in elem.items():
#         column_type = "Discrete" if column.dtype == "int64" else "Continuous"
#         worksheet.write(row, col, row)
#         worksheet.write(row, col + 1, column_name)
#         worksheet.write(row, col + 2, column_type)
#         worksheet.write(row, col + 3, f_values[column_name])
#         worksheet.write(row, col + 4, mi_values[column_name])
#         row += 1

#     worksheet.autofit()
#     workbook.close()

# Perform discretization on both datasets
enc = KBinsDiscretizer(n_bins=3, encode="onehot")
X_f_binned = enc.fit_transform(X_f)
X_mi_binned = enc.fit_transform(X_mi)

# Visualization of feature selection methods
# mi = pandas.Series(mi)
# mi.index = X.columns
# mi.sort_values(ascending=False).plot.bar(figsize=(20, 6))
# plt.ylabel('Mutual Information')
# plt.show()

f = pandas.Series(f)
f.index = X.columns
f.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.ylabel('ANOVA F-value')
plt.show()