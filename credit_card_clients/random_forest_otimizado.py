import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
sys.path.append(os.path.abspath('../'))
from config import DEFAULT_CREDIT_DATASET

# Lê o banco de dados (xlsx)
df = pd.read_excel(DEFAULT_CREDIT_DATASET)

# Mapeia as variáveis categóricas (análise exploratória)
sex_map = {1: 'Masculino', 2: 'Feminino'}
education_map = {1: 'Pós-graduação', 2: 'Universidade', 3: 'Ensino Médio', 4: 'Outros'}
marriage_map = {1: 'Casado', 2: 'Solteiro', 3: 'Outros'}

df['SEX'] = df['SEX'].map(sex_map)
df['EDUCATION'] = df['EDUCATION'].map(education_map)
df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map)

# Define as variáveis independentes e as variáveis dependentes
X = df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE', 
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

y = df['default payment next month']

# Transforma variáveis categóricas em variáveis dummies
X = pd.get_dummies(X, drop_first=True)

# Aplica SMOTE para lidar com o desbalanceamento das classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Divide os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.80, test_size=0.20, random_state=42)

# Define os hiperparâmetros para otimização
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Ajuste de pesos
weights = {0: 1, 1: 5}
rf = RandomForestClassifier(class_weight=weights, random_state=42)

# Otimiza os hiperparâmetros
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Seleciona o modelo
best_rf = grid_search.best_estimator_

# Previsões com o modelo otimizado
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

# Avalia o desempenho do modelo
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Calcula e plota a curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='AUC = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC - Random Forest (Melhor Modelo)')
plt.legend(loc='lower right')
plt.grid()
plt.show()
