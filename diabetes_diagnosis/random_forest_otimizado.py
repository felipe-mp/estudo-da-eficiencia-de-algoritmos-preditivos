import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Caminho para o dataset configurado no config.py
sys.path.append(os.path.abspath('../'))
from config import DIABETES_DIAGNOSIS_DATASET

# Lê o banco de dados (csv)
df = pd.read_csv(DIABETES_DIAGNOSIS_DATASET, delimiter=';')

# Inserção dos dados com IterativeImputer (regressão)
X = df.drop(columns=['Outcome'])
y = df['Outcome']
imputer = IterativeImputer(random_state=0)
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42, stratify=y)

# Oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Configuração do modelo Random Forest
rf_model = RandomForestClassifier(class_weight={0: 1, 1: 10}, random_state=42)

# Ajuste de hiperparâmetros com GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

# Melhor modelo encontrado
print("\nMelhores parâmetros encontrados pelo GridSearchCV:")
print(grid_search.best_params_)
rf_best_model = grid_search.best_estimator_

# Treinamento e previsões
rf_best_model.fit(X_train_resampled, y_train_resampled)
y_pred = rf_best_model.predict(X_test)

# Avaliação do modelo
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Curva ROC e AUC
y_prob = rf_best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plotagem da Curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Curva ROC - Random Forest')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Validação cruzada
cv_scores = cross_val_score(rf_best_model, X_imputed, y, cv=5, scoring='accuracy')
print(f'Acurácia média com validação cruzada: {cv_scores.mean():.4f}')
