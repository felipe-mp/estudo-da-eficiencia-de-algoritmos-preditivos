import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline
sys.path.append(os.path.abspath('../'))
from config import DEFAULT_CREDIT_DATASET

# Lê o banco de dados (xlsx)
df = pd.read_excel(DEFAULT_CREDIT_DATASET)

# Mapeia as variáveis categóricas (análise exploratória)
sex_map = {1: 'Masculino', 2: 'Feminino'}
education_map = {1: 'Pós-graduação', 2: 'Universidade', 3: 'Ensino Médio', 4: 'Outros'}
marriage_map = {1: 'Casado', 2: 'Solteiro', 3: 'Outros'}

df['SEX'] = df['SEX'].map(sex_map).fillna('Outros')
df['EDUCATION'] = df['EDUCATION'].map(education_map).fillna('Outros')
df['MARRIAGE'] = df['MARRIAGE'].map(marriage_map).fillna('Outros')

# Define as variáveis independentes e as variáveis dependentes
X = df[['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

y = df['default payment next month']

# Transforma variáveis categóricas em variáveis dummies
X = pd.get_dummies(X, drop_first=True)

# Normaliza as variáveis numéricas
scaler = StandardScaler()
numeric_columns = ['LIMIT_BAL', 'AGE', 
                   'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, test_size=0.2, random_state=42)

# Oversampling
oversample = SMOTE(random_state=42)
model = LogisticRegression(max_iter=1000)

# Define o pipeline
pipeline = Pipeline([('sampling', oversample), ('model', model)])

# Define os hiperparâmetros a serem otimizados
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],  # Regularização
    'model__class_weight': [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 10}],  # Ajuste de pesos
    'model__solver': ['liblinear', 'lbfgs']  # Solvers recomendados para regressão logística
}

# Otimização de hiperparâmetros com GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado pelo GridSearch
best_model = grid_search.best_estimator_

# Previsões no conjunto de teste
y_pred = best_model.predict(X_test)

# Avaliação de desempenho do modelo
print("Melhores Hiperparâmetros:")
print(grid_search.best_params_)

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Calcula a curva ROC
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plota a curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', label='Área sob a Curva (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()
