#%%
import pandas as pd
import numpy as np 

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score


import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import euclidean


# -----------------------------------------

#%%

df = pd.read_csv('dados/breast-cancer-wisconsin.data',sep=',')

df.dtypes

df = df.reset_index(drop=True)

#%%
#excluindo ? e convertendo a coluna Bare_Nuclei para inteira. 
 
df = df.loc[df['Bare_nuclei'] != '?']
df['Bare_nuclei'] = df['Bare_nuclei'].astype(int)

# %%
df.loc[df['Class'] == 2, 'Class'] = 'benigno'
df.loc[df['Class'] == 4, 'Class'] = 'maligno'

#%%
df

#%%
# SEPARAÇÂO MODELO DE TREINAMENTO e TESTE

X = df.drop(['Sample_code_number','Class'],axis=1)
y = df['Class']

X_train, X_test, y_train,y_test = train_test_split(
    X,         
    y,
    test_size=0.20,   #tamanho do conjunto de testes em 20% da base
    stratify=y,
    random_state=42
)
#%%
# Os conjuntos retornados são do mesmo tipo que os conjuntos X e y
# As dimensões de cada conjunto, matriz ou vetor podem ser obtidos por meio do atributo shape
print(f"Conjunto de treino (features): {X_train.shape}, {type(X_train)}")
print(f"Conjunto de treino (classes): {y_train.shape}, {type(y_train)}")
print(f"Conjunto de teste (features): {X_test.shape}, {type(X_test)}")
print(f"Conjunto de teste (classes): {y_test.shape}, {type(y_test)}")


print("\nDistribuição das classes no conjunto de treino:")
display(y_train.value_counts())

print("\nDistribuição das classes no conjunto de teste:")
display(y_test.value_counts())
     

#%%

#Convertenedo as colunas/Series do Dataframe para Numpy

X_train = X_train.to_numpy()  # shape: (546, 9)
X_test = X_test.to_numpy()    # shape: (137, 9)
y_train = y_train.to_numpy()  # shape: (546,)
y_test = y_test.to_numpy()    # shape: (137,)

# Dicionário para armazenar os vetores médios de cada classe
mean_vectors = {}


#%%

# Calcular o Vetor Médio de cada classe.

print("Vetores Médios para Cada Classe:")
for _, class_name in enumerate(df['Class'].unique()):
    # Filtrando X_train para obter apenas as instâncias da classe corrente
    # Como y_train contém strings dos nomes das classes, podemos filtrar diretamente
    X_train_class = X_train[y_train == class_name]

    # Calculando o vetor médio para as features desta classe
    # O argumento axis indica a 'direção' do cálculo:
    # Quando axis=0, a operação é realizada na direção vertical e portanto a quantidade de colunas é preservada.
    # Quando axis=1, a operação é realizada na direção horizontal e portanto a quantidade de linhas é preservada.
    mean_vector = np.mean(X_train_class, axis=0) # mean_vector.shape: (1, 4)
    mean_vectors[class_name] = mean_vector

    print(f"  {class_name.capitalize()}: {mean_vector}")
    



   



#%%
pairs = list(itertools.combinations(X, 2))
len(pairs)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_lineares = axes.flatten()

paleta_cores = {'benigno': 'forestgreen', 'maligno': 'crimson'}
 
for i, (x_feature, y_feature) in zip(range(6), pairs):
    cor_grafico = 'crimson' if i % 2 == 0 else 'forestgreen'
    sns.scatterplot(
        data=df, 
        x=x_feature, 
        y=y_feature, 
        ax=axes_lineares[i], 
        palette=paleta_cores,
        color=cor_grafico,
        hue = df['Class'],
        alpha=0.6,
        edgecolor='w'
    )
     
axes_lineares[i].set_title(f'{x_feature} vs {y_feature}', fontsize=12, fontweight='bold')
axes_lineares[i].set_xlabel(x_feature, fontsize=10)
axes_lineares[i].set_ylabel(y_feature, fontsize=10)
axes_lineares[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


#%%

# Lista para armazenar as previsões realizadas pelo classificador
predictions = []

for i, test_instance in enumerate(X_test):
    #representa o infinito positivo de pontos flutuantes
    min_distance = float('inf')
    predicted_class = None

    for class_name, mean_vec in mean_vectors.items():
        # Calculando a distância Euclidiana entre a instância de teste e o vetor médio da classe
        distance = euclidean(test_instance, mean_vec)
        # Para calcular a distância Euclidiana usando NumPy:
        #distance = np.linalg.norm(test_instance - mean_vec)

        if distance < min_distance:
            min_distance = distance
            predicted_class = class_name
    predictions.append(predicted_class)

# Convertendo as previsões para um array numpy para facilitar a comparação
predictions = np.array(predictions)

print("Previsões (primeiras 5):", predictions[:5])
print("Labels Reais (primeiras 5):", y_test[:5])


#%%
# Calculando a acurácia total
correct_predictions_total = np.sum(predictions == y_test) # total de acertos
total_instances_test = len(y_test)
overall_accuracy = (correct_predictions_total / total_instances_test) * 100

print(f"Acurácia Total: {overall_accuracy:.2f}%")

print("\nAcurácia por Classe:")
# Calculando a acurácia para cada classe
for class_name in y.unique():
    # Filtrando previsões e labels reais para esta classe
    class_indices = (y_test == class_name)
    true_labels_class = y_test[class_indices]
    predicted_labels_class = predictions[class_indices]

    correct_predictions_class = np.sum(predicted_labels_class == true_labels_class)
    total_instances_class = len(true_labels_class)

    accuracy_class = (correct_predictions_class / total_instances_class) * 100
    print(f" {class_name.capitalize()}: {accuracy_class:.2f}%")


#%%
# EXTRA  -  Aplicando Decision Tree Classifier para o Algoritmo de Breast_cancer



arvore_full = tree.DecisionTreeClassifier(random_state=42,max_depth=4)

#Arvore de Decisao
arvore_full.fit(X_test,y_test)
plt.figure(dpi = 400)
tree.plot_tree(arvore_full, feature_names=X.columns.tolist(),
class_names=arvore_full.classes_,
filled=True)

arvore_predict = arvore_full.predict(X_train)
#arvore_proba = arvore_full.predict_proba(X).#drop_duplicates()[:1]


pred_treino =arvore_full.predict(X_train)
acc_treino = accuracy_score(y_train, pred_treino)

# 3. Predict no teste (a prova real de desempenho)
pred_teste = arvore_full.predict(X_test)
acc_teste = accuracy_score(y_test, pred_teste)

print(f'Acurácia no Treino:{acc_treino*100:.2f}')
print(f'Acurácia no Teste:{acc_teste*100:.2f}')