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
