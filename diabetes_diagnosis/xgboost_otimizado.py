import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from config import DIABETES_DIAGNOSIS_DATASET

# Lê o banco de dados (csv)
df = pd.read_csv(DIABETES_DIAGNOSIS_DATASET, delimiter=';')

# Identifica colunas para substituição de zeros por NaN
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']

# Substitui valores 0 por NaN diretamente no DataFrame 'df'
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

# Insere valores faltantes (média para colunas numéricas)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separa recursos e alvo
X = df_imputed.drop('Outcome', axis=1)
y = df_imputed['Outcome']

# Divide em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normaliza os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# balanceia as classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Define os parâmetros para Grid Search
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}

# Configura o modelo XGBoost
model = xgb.XGBClassifier(scale_pos_weight=2, eval_metric='logloss', random_state=42)

# Usa Grid Search para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=5, verbose=1)
grid_search.fit(X_resampled, y_resampled)

# Melhor modelo
best_model = grid_search.best_estimator_

# Previsões
y_pred = best_model.predict(X_test_scaled)

# Exibe métricas
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print(f"Quantidade de amostras para treino: {X_resampled.shape[0]}")
print(f"Quantidade de amostras para teste: {X_test.shape[0]}")

# Cálculo das probabilidades
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Cálculo da Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plota a Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='Curva ROC (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Curva ROC XGBoost')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc='lower right')
plt.grid()
plt.show()
