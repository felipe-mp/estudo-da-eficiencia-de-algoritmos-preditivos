import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
sys.path.append(os.path.abspath('../'))
from config import DEFAULT_CREDIT_DATASET

# Lê o banco de dados (xlsx)
df = pd.read_excel(DEFAULT_CREDIT_DATASET)

# Define as variáveis independentes
X = df[['LIMIT_BAL', 'AGE', 'EDUCATION', 'MARRIAGE', 'SEX', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

# Normaliza as variáveis independentes
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Aplica K-Means para encontrar grupos (clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_normalized)

# Adiciona os grupos ao dataframe
df['Cluster'] = clusters

# Calcula a taxa de inadimplência para cada cluster
inadimplencia_por_cluster = df.groupby('Cluster')['default payment next month'].mean()

# Renomeia os clusters
cluster_names = {
    0: "Cluster 0 (Roxo)",
    1: "Cluster 1 (Azul Escuro)",
    2: "Cluster 2 (Verde Escuro)",
    3: "Cluster 3 (Verde Claro)",
    4: "Cluster 4 (Amarelo)"
}

# Cria uma lista de cores
colors = plt.cm.viridis(np.linspace(0, 1, len(inadimplencia_por_cluster)))

# Exibe as taxas de inadimplência
print("Taxa de inadimplência por cluster e cores:")
for i, cluster in enumerate(inadimplencia_por_cluster.index):
    print(f"{cluster_names[cluster]}: Taxa de Inadimplência {inadimplencia_por_cluster[cluster]:.2f}")

# Aplica PCA para redução de dimensionalidade
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

# Plota a matriz de agrupamentos com redução de dimensionalidade (PCA)
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Agrupamento dos clientes - Redução de Dimensão com PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.colorbar(label='Cluster')
plt.show()

# Exibe a variância explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
print("Variância explicada por cada componente:", explained_variance)

# Exibe os pesos de cada variável original para os componentes principais
components = pca.components_
df_components = pd.DataFrame(components, columns=X.columns, index=[f'PC{i+1}' for i in range(components.shape[0])])
print("Coeficientes dos componentes principais:")
print(df_components)

# Calcula as características médias de cada cluster
cluster_means = df.groupby('Cluster').mean()
print("Características médias de cada cluster:")
print(cluster_means)

# Exibe a interpretação dos clusters
for cluster in cluster_means.index:
    print(f"\nAnálise do {cluster_names[cluster]}:")
    print(cluster_means.loc[cluster])
