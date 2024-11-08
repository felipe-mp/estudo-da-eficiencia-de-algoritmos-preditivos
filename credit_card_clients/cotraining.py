import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
sys.path.append(os.path.abspath('../'))
from config import DEFAULT_CREDIT_DATASET

# Lê o banco de dados (xlsx)
df = pd.read_excel(DEFAULT_CREDIT_DATASET)

# Ajusta as variáveis independentes
X = df[['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
y = df['default payment next month']

# Normaliza as variáveis independentes
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Separa os dados rotulados e não rotulados
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_normalized, y, train_size=0.3, random_state=42)

# Dividir os dados rotulados em conjunto de treinamento e teste 
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, train_size=0.80, test_size=0.20, random_state=42)

# Verifica a distribuição das classes antes do SMOTE
print("Distribuição das classes no conjunto de treinamento antes do SMOTE:")
print(pd.Series(y_train).value_counts())

# Cria um objeto SMOTE e aplicar SMOTE nos dados rotulados
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verifica a distribuição das classes após o SMOTE
print("Distribuição das classes no conjunto de treinamento após o SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Inicializa os modelos
tree_model = DecisionTreeClassifier(random_state=42)
forest_model = RandomForestClassifier(random_state=42)

# Função co-training com múltiplas iterações
def co_training(tree_model, forest_model, X_train, y_train, X_unlabeled, epochs=10, confidence_threshold=0.7):
    for epoch in range(epochs):
        # Treina o primeiro modelo
        tree_model.fit(X_train, y_train)
        
        # Faz previsões nos dados não rotulados
        tree_preds_proba = tree_model.predict_proba(X_unlabeled)
        tree_preds = np.argmax(tree_preds_proba, axis=1)
        confidence_tree = np.max(tree_preds_proba, axis=1)
        
        # Adiciona previsões ao conjunto rotulado
        confident_indices_tree = np.where(confidence_tree >= confidence_threshold)[0]
        X_train = np.vstack((X_train, X_unlabeled[confident_indices_tree]))
        y_train = np.concatenate((y_train, tree_preds[confident_indices_tree]))
        
        # Treina o segundo modelo
        forest_model.fit(X_train, y_train)
        
        # Faz previsões nos dados não rotulados
        forest_preds_proba = forest_model.predict_proba(X_unlabeled)
        forest_preds = np.argmax(forest_preds_proba, axis=1)
        confidence_forest = np.max(forest_preds_proba, axis=1)
        
        # Adiciona previsões ao conjunto rotulado
        confident_indices_forest = np.where(confidence_forest >= confidence_threshold)[0]
        X_train = np.vstack((X_train, X_unlabeled[confident_indices_forest]))
        y_train = np.concatenate((y_train, forest_preds[confident_indices_forest]))
        
        # Remove os dados que foram adicionados
        X_unlabeled = np.delete(X_unlabeled, np.concatenate((confident_indices_tree, confident_indices_forest)), axis=0)
        
        print(f'Epoch {epoch + 1} completed.')

# Executa o modelo cooperativo
co_training(tree_model, forest_model, X_train_balanced, y_train_balanced, X_unlabeled, epochs=10)

# Testa a performance nos dados de teste
y_pred_tree = tree_model.predict(X_test)
y_pred_forest = forest_model.predict(X_test)

# Avalia o desempenho dos modelos
for y_pred, model, name in zip([y_pred_tree, y_pred_forest], [tree_model, forest_model], ['Decision Tree', 'Random Forest']):
    print(f'\nDesempenho - {name}:')
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Plota a curva ROC
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Usa a probabilidade do modelo correto
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'ROC curve {name} (area = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
