import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Carregando e preparando os dados
file_path = 'conjunto_de_dados_limpos/Campeonato_Brasileiro_de_futebol_limpo.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado.')

df = pd.read_csv(file_path)
df['data'] = pd.to_datetime(df['data'])

def define_resultado(row):
    if row['mandante_placar'] > row['visitante_placar']:
        return 'vitoria'
    elif row['mandante_placar'] < row['visitante_placar']:
        return 'derrota'
    else:
        return 'empate'

df['resultado'] = df.apply(define_resultado, axis=1)
df['ano'] = df['data'].dt.year

# Separando treino e teste
treino = df[df['ano'] < 2023].copy()
teste = df[df['ano'] == 2023].copy()

# Encoding dos times
le_mandante = LabelEncoder()
le_visitante = LabelEncoder()
treino['mandante_le'] = le_mandante.fit_transform(treino['mandante'])
treino['visitante_le'] = le_visitante.fit_transform(treino['visitante'])
teste['mandante_le'] = le_mandante.transform(teste['mandante'])
teste['visitante_le'] = le_visitante.transform(teste['visitante'])

# Features
features = ['mandante_le', 'visitante_le']
X_train = treino[features]
y_train = treino['resultado']
X_test = teste[features]
y_test = teste['resultado']

# Escalonamento
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo Original
print("\n=== Modelo Original (SVM.py) ===")
model_original = SVC(
    kernel='rbf',
    probability=True,
    random_state=42,
    class_weight='balanced',
    C=1.0,
    gamma='scale'
)
model_original.fit(X_train_scaled, y_train)
y_pred_original = model_original.predict(X_test_scaled)

# Métricas do modelo original
print("\nMétricas do Modelo Original:")
print(classification_report(y_test, y_pred_original, zero_division=0))

# Modelo Otimizado
print("\n=== Modelo Otimizado (otimizacao_svm.py) ===")
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'C': [1, 10],
    'gamma': ['scale'],
    'kernel': ['rbf']
}

svc = SVC(class_weight='balanced', probability=False)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(svc, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("\nMelhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_optimized = best_model.predict(X_test_scaled)

# Métricas do modelo otimizado
print("\nMétricas do Modelo Otimizado:")
print(classification_report(y_test, y_pred_optimized, zero_division=0))

# Comparação das métricas
def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

metrics_original = calculate_metrics(y_test, y_pred_original)
metrics_optimized = calculate_metrics(y_test, y_pred_optimized)

print("\n=== Comparação de Métricas ===")
print("\nModelo Original vs Otimizado:")
print(f"{'Métrica':<10} {'Original':>10} {'Otimizado':>10} {'Diferença':>10}")
print("-" * 42)
for metric in metrics_original.keys():
    diff = metrics_optimized[metric] - metrics_original[metric]
    print(f"{metric:<10} {metrics_original[metric]:>10.4f} {metrics_optimized[metric]:>10.4f} {diff:>10.4f}")

# Matriz de Confusão Comparativa
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Matriz de Confusão - Modelo Original
cm_original = confusion_matrix(y_test, y_pred_original)
sns.heatmap(cm_original, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=model_original.classes_,
            yticklabels=model_original.classes_)
ax1.set_title('Matriz de Confusão - Modelo Original')
ax1.set_xlabel('Previsto')
ax1.set_ylabel('Real')

# Matriz de Confusão - Modelo Otimizado
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=best_model.classes_,
            yticklabels=best_model.classes_)
ax2.set_title('Matriz de Confusão - Modelo Otimizado')
ax2.set_xlabel('Previsto')
ax2.set_ylabel('Real')

plt.tight_layout()
plt.savefig('comparacao_matriz_confusao.png')
plt.close()

# Resultados da Validação Cruzada
print("\n=== Resultados da Validação Cruzada (Modelo Otimizado) ===")
cv_results = pd.DataFrame(grid_search.cv_results_)
print("\nMédia e Desvio Padrão do F1-Score nos folds:")
print(f"Média: {cv_results['mean_test_score'].mean():.4f}")
print(f"Desvio Padrão: {cv_results['mean_test_score'].std():.4f}")

# Salvando resultados em um arquivo
with open('resultados_comparacao.txt', 'w') as f:
    f.write("=== Resultados da Comparação entre Modelos SVM ===\n\n")
    f.write("Modelo Original:\n")
    f.write(classification_report(y_test, y_pred_original, zero_division=0))
    f.write("\nModelo Otimizado:\n")
    f.write(classification_report(y_test, y_pred_optimized, zero_division=0))
    f.write("\nComparação de Métricas:\n")
    f.write(f"{'Métrica':<10} {'Original':>10} {'Otimizado':>10} {'Diferença':>10}\n")
    f.write("-" * 42 + "\n")
    for metric in metrics_original.keys():
        diff = metrics_optimized[metric] - metrics_original[metric]
        f.write(f"{metric:<10} {metrics_original[metric]:>10.4f} {metrics_optimized[metric]:>10.4f} {diff:>10.4f}\n") 