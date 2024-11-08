import os
import sys
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Habilitar o IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Caminho para o dataset configurado no config.py
sys.path.append(os.path.abspath('../'))
from config import DIABETES_DIAGNOSIS_DATASET

# Lê o banco de dados (csv)
df = pd.read_csv(DIABETES_DIAGNOSIS_DATASET, delimiter=';')

# Exibindo a quantidade de registros e colunas ao carregar o dataset completo
print(f"Quantidade de registros e colunas ao carregar o dataset completo: {df.shape}")

# Colunas que queremos manter com valores zero (Pregnancies e Outcome)
cols_to_exclude = ['Pregnancies', 'Outcome']

# Substituir valores iguais a 0 por NaN em todas as colunas, exceto as especificadas
df[df.columns.difference(cols_to_exclude)] = df[df.columns.difference(cols_to_exclude)].replace(0, np.nan)

# Usando o Iterative Imputer para substituir os valores NaN
imputer = IterativeImputer(random_state=42)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Exibindo a quantidade de registros e colunas após a imputação
print(f"Quantidade de registros e colunas após a imputação: {df_imputed.shape}")

# Exibindo o DataFrame imputado
print("Os 10 primeiros registros após a imputação:")
print(df_imputed.head(10))

# Separar as características
X = df_imputed.drop(columns=['Outcome'])  # Removendo apenas a coluna 'Outcome'

# Pesos desejados para PC1 e PC2
weights_pc1 = np.array([0.185988, 0.569239, 0.521334, 0.636927, 0.629979, 0.654475, 0.391883, 0.286771])
weights_pc2 = np.array([0.781786, 0.229129, 0.242115, -0.437070, -0.330181, -0.132925, -0.160718, 0.817074])

# Calculando os componentes principais manualmente
PC1 = X.values @ weights_pc1
PC2 = X.values @ weights_pc2

# Criando um novo DataFrame com os componentes principais
X_pca_manual = np.column_stack((PC1, PC2))

# Aplicando K-means nos componentes principais
kmeans_pca = KMeans(n_clusters=3, random_state=42)
clusters_pca = kmeans_pca.fit_predict(X_pca_manual)

# Exibindo os centros dos clusters no espaço PCA
print("Centros dos Clusters (PCA):\n", kmeans_pca.cluster_centers_)

# Exibindo o percentual de diabéticos em cada cluster PCA
percent_diabetic_per_cluster_pca = df_imputed.groupby(clusters_pca)['Outcome'].mean() * 100
print("\nPercentual de pessoas diabéticas em cada cluster (PCA):")
print(percent_diabetic_per_cluster_pca)

# Visualizando a distribuição dos clusters (utilizando componentes principais)
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green']
for cluster in range(3):
    plt.scatter(X_pca_manual[clusters_pca == cluster][:, 0],
                X_pca_manual[clusters_pca == cluster][:, 1],
                color=colors[cluster],
                label=f'Cluster {cluster}')

# Adicionando título e legendas
plt.title('Distribuição dos Clusters K-means')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid()
plt.legend()
plt.show()

# Informações sobre o DataFrame e os componentes principais
print("Formato do DataFrame imputado:", df_imputed.shape)
print("Formato dos componentes principais:", X_pca_manual.shape)

# 8. Separar as características
columns_to_drop = ['Outcome']
if 'Cluster' in df_imputed.columns:
    columns_to_drop.append('Cluster')

X = df_imputed.drop(columns=columns_to_drop)  # Apenas as características numéricas

# Pesos desejados para PC1 e PC2
weights_pc1 = np.array([0.185988, 0.569239, 0.521334, 0.636927, 0.629979, 0.654475, 0.391883, 0.286771])
weights_pc2 = np.array([0.781786, 0.229129, 0.242115, -0.437070, -0.330181, -0.132925, -0.160718, 0.817074])

# Calculando os componentes principais manualmente
PC1 = X.values @ weights_pc1
PC2 = X.values @ weights_pc2

# Criando um novo DataFrame com os componentes principais
X_pca_manual = np.column_stack((PC1, PC2))

# 9. Aplicando K-means nos componentes principais
kmeans_pca = KMeans(n_clusters=3, random_state=42)
clusters_pca = kmeans_pca.fit_predict(X_pca_manual)

# 10. Exibindo o percentual de diabéticos em cada cluster
# Calculando o percentual de diabéticos por cluster sem adicionar a coluna 'Cluster'
percent_diabetic_per_cluster_pca = []
for cluster in range(3):
    # Calculando o percentual de diabéticos para o cluster específico
    cluster_indices = (clusters_pca == cluster)
    percent_diabetic = df_imputed.loc[cluster_indices, 'Outcome'].mean() * 100
    percent_diabetic_per_cluster_pca.append(percent_diabetic)

# Exibindo o percentual
print("\nPercentual de pessoas diabéticas em cada cluster (PCA):")
for i, percent in enumerate(percent_diabetic_per_cluster_pca):
    print(f"Cluster {i}: {percent:.2f}%")

