#%%
import pandas as pd
import numpy as np 
import itertools


from IPython.display import display
from scipy.spatial.distance import euclidean

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns



# -----------------------------------------

#%%
# Leitura do DataFrame Breast_cancer

df = pd.read_csv('../dados/breast-cancer-wisconsin.data',sep=',')
df.dtypes
df = df.reset_index(drop=True)

#%%
#excluindo ? e convertendo a coluna Bare_Nuclei para inteira. 
 
df = df.loc[df['Bare_nuclei'] != '?']
df['Bare_nuclei'] = df['Bare_nuclei'].astype(int)

# %%

#Renomeando categorias 2 = benigno e  = maligno
df.loc[df['Class'] == 2, 'Class'] = 'benigno'
df.loc[df['Class'] == 4, 'Class'] = 'maligno'

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
    
ordered_centroids = [mean_vectors[name] for name in y] # insere os vetores em uma lista
centroids_array = np.array(ordered_centroids) # cria a matriz a partir da lista

#%%
#%%
#Funcao para geração de grafico de dispersão usando Centroids
colors = {'benigno':'green','maligno':'red'}
def plot_Breast_cancer_scatter_with_centroids(
    df,
    class_names,
    feature_names,
    colors_map,
    centroids_array=None
):
    """
    Função que gera gráficos de dispersão para cada par de features do Breast Cancer dataset.
    Tem como opção o desenho dos centróides das classes.
    Args:
        df (pd.DataFrame): DataFrame com os dados do Breast Cancer.
        class_names (list): Lista com os nomes das classes (e.g., ['Benigno', 'maligno']).
        feature_names (list): Lista com os nomes das features (e.g., 'Clump_thickness', 'Uniformity_of_cell_size',
       'Uniformity_of_cell_shape', 'Marginal_adhesion',
       'Single_epithelial_cell_size', 'Bare_nuclei', 'Bland_chromatin',
       'Normal_nucleoli', 'Mitoses']).
       
        colors_map (dict): Dicionário mapeando nomes de classes a cores.
        centroids_array (np.ndarray, optional): Matriz NumPy na qual cada linha contém o centróide de uma classe,ordenados pela lista `class_names`. As colunas devem seguir a ordem da lista `feature_names`.
    """
    # Usando itertools para gerar as combinações únicas de pares de features
    # O resultado é uma lista onde cada elemento é uma tupla com os nomes de
    # duas features, por exemplo ('Clump_thickness', 'Uniformity_of_cell_size')
    pairs = list(itertools.combinations(feature_names, 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_lineares = axes.flatten()
    # Inicializando a área do gráfico. Os valores indicam a largura e altura
    # da área em polegadas
    #plt.figure(figsize=(15, 10))

    # Criando um subplot para cada par de features
    for i, (x_feature, y_feature) in zip(range(6), pairs[8:]):
        # A função plt.subplot divide a área do gráfico em linhas e colunas.
        # O primeiro argumento indica o número de linhas, o segundo o número de colunas
        # e o terceiro argumento é o índice dessa grade na qual o gráfico será desenhado.
        # Repare que o primeiro índice tem valor 1 ao invés de zero. A sequência de
        # índices é orientada de cima para baixo e da direita para a esquerda.
        plt.subplot(2, 3, i+1)
    
        # Traça os pontos de cada espécie considerando apenas os atributos
        # indicados em x_feature e y_feature
        for species_name, color in colors_map.items():
            # Obtém um DataFrame contendo somente as linhas da espécie corrente
            subset = df[df['Class'] == species_name]

            # plt.scatter desenha um gráfico de dispersão
            plt.scatter(
                subset[x_feature],  # obtém um Series com os valores do atributo indicado em x_feature
                subset[y_feature],  # obtém um Series com os valores do atributo indicado em y_feature
                color=color,        # cor dos pontos
                label=species_name, # texto da legenda
                s=70,               # tamanho dos pontos
                alpha=0.8           # transparência dos pontos
            )

        # Desenha os centróides caso sejam fornecidos
        if centroids_array is not None:
            for j, class_name in enumerate(class_names):
                centroid_row = centroids_array[j]

                # Encontra o índice dos x_feature e y_features correntes na lista feature_names
                x_idx = feature_names.get_loc(x_feature)
                y_idx = feature_names.get_loc(y_feature)


                plt.scatter(
                    centroid_row[x_idx],          # coordenada x do centróide
                    centroid_row[y_idx],          # coordenada y do centróide
                    marker='D',                   # marcador em formato de diamante
                    color=colors_map[class_name], # Centróide com a mesma cor dos pontos da classe
                    s=150,                        # Tamanho do marcador
                    edgecolors='black',           # Cor da borda do marcador
                    linewidths=1.5                # Espessura da borda do marcador
                )

        title_x = x_feature.replace(' (scale)', '')
        title_y = y_feature.replace(' (scale)', '')
        plt.title(f'{title_x} vs {title_y}')
        plt.xlabel(title_x)
        plt.ylabel(title_y)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout() # Ajusta o espaçamento entre os elementos do gráfico
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

#GERANDO O GRAFICO DE DISPERSÃO COM OS CENTROIDS MEDIOS DAS CLASSES BENIGNO E MALIGNO.
plot_Breast_cancer_scatter_with_centroids(df,df['Class'],X.columns,colors,centroids_array)

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







