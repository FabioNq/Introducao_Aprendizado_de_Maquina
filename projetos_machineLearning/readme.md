# Breast Cancer Classification

Projeto de análise e classificação de tumores utilizando técnicas de Ciência de Dados e Machine Learning.

O conjunto de dados contém características relacionadas às células de amostras de tumores, com o objetivo de classificar cada amostra como **benigna** ou **maligna**.

## Dicionário de Dados

| Nome do Atributo | Tipo de Dado | Descrição | Valores Possíveis |
|---|---|---|---|
| `Sample code number` | Numérico (ID) | Código de identificação exclusivo da amostra. | Código numérico |
| `Clump Thickness` | Inteiro | Espessura do agrupamento de células. Células benignas tendem a se agrupar em camadas únicas, enquanto células malignas podem se agrupar em várias camadas. | 1 (normal) a 10 (anormal) |
| `Uniformity of Cell Size` | Inteiro | Uniformidade do tamanho das células. Células cancerígenas costumam apresentar maior variação de tamanho. | 1 (normal) a 10 (anormal) |
| `Uniformity of Cell Shape` | Inteiro | Uniformidade do formato das células. Avalia se as células mantêm uma forma padrão ou apresentam características anômalas/irregulares. | 1 (normal) a 10 (anormal) |
| `Marginal Adhesion` | Inteiro | Adesão marginal. Avalia a aderência entre as células. | 1 (normal) a 10 (anormal) |
| `Single Epithelial Cell Size` | Inteiro | Tamanho de uma célula epitelial isolada. Células epiteliais significativamente aumentadas podem estar associadas à malignidade. | 1 (normal) a 10 (anormal) |
| `Bare Nuclei` | Inteiro | Núcleos nus, referentes a núcleos que aparecem sem o citoplasma ao redor. Esta variável contém 16 valores ausentes representados por `?`. | 1 (normal) a 10 (anormal) ou `?` |
| `Bland Chromatin` | Inteiro | Característica relacionada à cromatina presente no núcleo celular. | 1 (normal) a 10 (anormal) |
| `Normal Nucleoli` | Inteiro | Característica relacionada aos nucléolos das células. Em células cancerígenas, os nucléolos podem se tornar mais proeminentes e numerosos. | 1 (normal) a 10 (anormal) |
| `Mitoses` | Inteiro | Mede a taxa de divisão celular. Valores elevados podem estar associados a maior atividade de divisão celular. | 1 (normal) a 10 (anormal) |
| `Class` | Inteiro (Alvo) | Variável resposta utilizada para classificar o diagnóstico do tumor. | `2` = Benigno / `4` = Maligno |

## Variáveis do Dataset

### Variáveis preditoras

As seguintes variáveis são utilizadas como características (`features`) para o modelo de Machine Learning:

- `Clump Thickness`
- `Uniformity of Cell Size`
- `Uniformity of Cell Shape`
- `Marginal Adhesion`
- `Single Epithelial Cell Size`
- `Bare Nuclei`
- `Bland Chromatin`
- `Normal Nucleoli`
- `Mitoses`

### Variável alvo

A variável `Class` representa o diagnóstico da amostra:

| Valor | Diagnóstico |
|---:|---|
| `2` | Benigno |
| `4` | Maligno |


# Classificação Breast Cancer utilizando Centróides e Distância Euclidiana

## 1. Sobre o projeto

Este projeto utiliza o dataset **Breast Cancer Wisconsin** para realizar uma classificação entre duas classes:

* **Benigno**
* **Maligno**

A classificação é realizada utilizando uma abordagem baseada no cálculo do **vetor médio (centróide)** das características de cada classe.

Para classificar uma nova observação, é calculada a **distância Euclidiana** entre essa observação e o vetor médio de cada classe. A classe cujo centróide estiver mais próximo é utilizada como previsão.

O projeto também realiza uma comparação da acurácia obtida no conjunto de teste.

---

# Bibliotecas utilizadas

As principais bibliotecas utilizadas no projeto são:

* **Pandas:** manipulação do DataFrame.
* **NumPy:** operações matemáticas e manipulação de arrays.
* **SciPy:** cálculo da distância Euclidiana.
* **Scikit-learn:** separação dos dados e cálculo de métricas.
* **Matplotlib:** criação dos gráficos.
* **Seaborn:** utilizado como biblioteca auxiliar para visualização.

---

# Leitura dos dados

O dataset é carregado utilizando o Pandas:

```python
df = pd.read_csv(
    '../dados/breast-cancer-wisconsin.data',
    sep=','
)

df = df.reset_index(drop=True)
```

O arquivo contém informações relacionadas às características das células analisadas.

Entre as variáveis utilizadas estão:

* `Clump_thickness`
* `Uniformity_of_cell_size`
* `Uniformity_of_cell_shape`
* `Marginal_adhesion`
* `Single_epithelial_cell_size`
* `Bare_nuclei`
* `Bland_chromatin`
* `Normal_nucleoli`
* `Mitoses`

A variável `Class` representa a classe da amostra.

---

# Limpeza dos valores da coluna Bare_nuclei

A coluna `Bare_nuclei` possui alguns valores representados pelo caractere `?`.

Antes de utilizar os dados no modelo, esses registros são removidos:

```python
df = df.loc[df['Bare_nuclei'] != '?']
```

Depois disso, a coluna é convertida para o tipo inteiro:

```python
df['Bare_nuclei'] = df['Bare_nuclei'].astype(int)
```

Esse tratamento é necessário porque o cálculo das médias e das distâncias Euclidianas exige valores numéricos.

O código também transforma os valores da variável `Class`:

```python
df.loc[df['Class'] == 2, 'Class'] = 'benigno'
df.loc[df['Class'] == 4, 'Class'] = 'maligno'
```

Dessa forma:

| Valor original | Nova classificação |
| -------------- | ------------------ |
| 2              | benigno            |
| 4              | maligno            |

---

# Separação entre variáveis X e y

As variáveis utilizadas para realizar a previsão são separadas da variável que representa a classe.

```python
X = df.drop(
    ['Sample_code_number', 'Class'],
    axis=1
)

y = df['Class']
```

## X

`X` contém as **features** utilizadas para realizar a classificação.

A coluna `Sample_code_number` é removida porque representa o identificador da amostra e não uma característica utilizada para classificação.

A coluna `Class` também é removida porque representa justamente o resultado que queremos prever.

## y

`y` contém a variável alvo:

```text
benigno
maligno
```

Podemos representar:

```text
X → características das células

y → classe da célula
```

---

# Separação entre treino e teste

Os dados são divididos utilizando `train_test_split()`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

Foi utilizado:

```python
test_size=0.20
```

Isso significa que:

* **80% dos dados → treinamento**
* **20% dos dados → teste**

Representação:

```text
Dataset
│
├── 80% → Treinamento
│
└── 20% → Teste
```

Também foi utilizado:

```python
stratify=y
```

para preservar a proporção das classes durante a divisão.

O parâmetro:

```python
random_state=42
```

permite reproduzir a mesma divisão dos dados em diferentes execuções.

---

# Conversão das variáveis para NumPy

Depois da separação dos dados, os DataFrames e Series são convertidos para arrays NumPy:

```python
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
```

Essa conversão facilita a realização das operações matemáticas utilizadas posteriormente.

A estrutura passa a ser aproximadamente:

```text
X_train → matriz de características
X_test  → matriz de características

y_train → vetor de classes
y_test  → vetor de classes
```

No código, o conjunto de treinamento possui formato:

```text
(546, 9)
```

e o conjunto de teste:

```text
(137, 9)
```

Ou seja, cada amostra possui **9 características**.

---

# Criação dos vetores médios

Nesta etapa é calculado o **vetor médio de cada classe**.

Primeiro é criado um dicionário:

```python
mean_vectors = {}
```

Depois, os dados de treinamento são separados de acordo com a classe:

```python
X_train_class = X_train[y_train == class_name]
```

Para cada classe é calculada a média das características:

```python
mean_vector = np.mean(
    X_train_class,
    axis=0
)
```

O parâmetro:

```python
axis=0
```

indica que a média será calculada para cada coluna.

Por exemplo, podemos imaginar uma classe com três características:

```text
Feature 1    Feature 2    Feature 3
   2             5            8
   4             7            6
   6             9            10
```

O vetor médio será:

```text
[4, 7, 8]
```

Esse vetor representa o **centróide médio da classe**.

O processo é realizado separadamente para:

```text
Benigno → vetor médio
Maligno → vetor médio
```

---

# Classificação utilizando distância Euclidiana

Depois de calcular os centróides, cada observação do conjunto de teste é comparada com os vetores médios das classes.

Para cada observação:

```python
for test_instance in X_test:
```

é calculada a distância até cada vetor médio:

```python
distance = euclidean(
    test_instance,
    mean_vec
)
```

A distância Euclidiana pode ser representada matematicamente como:

```text
d(x,y) = √ Σ(xᵢ - yᵢ)²
```

Quanto menor a distância, mais próximos são os dois pontos.

O algoritmo mantém a menor distância:

```python
if distance < min_distance:
    min_distance = distance
    predicted_class = class_name
```

Portanto:

```text
Nova amostra
      │
      ├── distância → centróide benigno
      │
      └── distância → centróide maligno
                    │
                    ↓
          menor distância
                    │
                    ↓
             classe prevista
```

---

# Treinamento

Nesta abordagem, o "treinamento" consiste em utilizar os dados de `X_train` para calcular o vetor médio de cada classe.

```python
mean_vector = np.mean(
    X_train_class,
    axis=0
)
```

Assim, os centróides são construídos **somente utilizando os dados de treinamento**.

Isso é importante porque o conjunto de teste deve ser utilizado para verificar o comportamento do método em dados que não participaram da criação dos centróides.

---

# Teste

Após a criação dos centróides, os dados de `X_test` são utilizados para realizar as previsões.

Para cada observação:

1. Calcula-se a distância até o centróide benigno.
2. Calcula-se a distância até o centróide maligno.
3. Comparam-se as distâncias.
4. A classe do centróide mais próximo é escolhida.

As previsões são armazenadas:

```python
predictions.append(predicted_class)
```

No final:

```python
predictions = np.array(predictions)
```

As previsões podem então ser comparadas com os valores reais:

```python
predictions == y_test
```

---

# Cálculo da acurácia

A acurácia representa a proporção de previsões que foram classificadas corretamente.

No código:

```python
correct_predictions_total = np.sum(
    predictions == y_test
)

total_instances_test = len(y_test)

overall_accuracy = (
    correct_predictions_total /
    total_instances_test
) * 100
```

A fórmula pode ser representada como:

```text
Acurácia =
número de previsões corretas
----------------------------
número total de previsões
```

Multiplicando por 100, obtemos a porcentagem de acerto.

O resultado é apresentado através de:

```python
print(f"Acurácia Total: {overall_accuracy:.2f}%")
```

---

# Gráfico do centróide médio

Um dos gráficos importantes para análise é a relação entre:

```text
Uniformity_of_cell_size
```

e outra característica do dataset.

O centróide médio de cada classe é representado no gráfico junto com as observações.


<img src="https://github.com/FabioNq/Introducao_Aprendizado_de_Maquina/blob/main/projetos_machineLearning/imagens/Centroid_Medio_dispersao.png" alt="Gráfico dos centróides" heigth="1500" width="1000">


O gráfico permite visualizar a posição média das classes no espaço das características utilizadas, foi utilizada a variavel Uniformity_of_cell_size pois ela influencia bastante nos resultados dos testes, como mostra o gráfico.

O centróide representa uma espécie de **ponto central da classe**, calculado a partir da média das características das amostras utilizadas no treinamento.

---

### Resultado do teste

```text
Acurácia Total: 96.35%
```

### Acurácia por classe

```text
Benigno: 95.51%
Maligno: 97.92%
```


# Conclusão

O projeto demonstra uma abordagem simples de classificação baseada em **centróides**.

Em vez de utilizar um algoritmo tradicional de Machine Learning para aprender uma função de decisão, o método calcula o vetor médio das características de cada classe e utiliza a **distância Euclidiana** para determinar qual classe está mais próxima de uma nova observação.

O fluxo utilizado foi:

1. Limpeza dos dados.
2. Tratamento da variável `Bare_nuclei`.
3. Separação das features (`X`) e variável alvo (`y`).
4. Divisão em 80% para treinamento e 20% para teste.
5. Conversão dos dados para NumPy.
6. Cálculo dos vetores médios das classes.
7. Comparação das amostras utilizando distância Euclidiana.
8. Classificação das amostras do conjunto de teste.
9. Cálculo da acurácia.
10. Visualização dos centróides através de gráficos de dispersão.

O projeto permite visualizar de forma prática como **distância, média, centróide e classificação** podem ser utilizados conjuntamente para resolver um problema de classificação.

# Extra Estudo 

Foi realizado o mesmo treinamento utilizando o Algoritmo de machine Learning Arvore de decisao como modelo de classificação e o resultado é bastante interessante pois a arvore começa a se dividir a partir da variavel Uniformity_of_cell_size sendo como principal ponto de partida para definir se uma pessoa tem um tumor maligno ou benigo, segue a imagem a seguir : 



<img src="https://github.com/FabioNq/Introducao_Aprendizado_de_Maquina/blob/main/projetos_machineLearning/imagens/Decision_tree_Classifier.png" alt="Gráfico dos centróides" heigth="1500" width="1000">



