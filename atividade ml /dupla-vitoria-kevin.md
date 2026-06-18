# Lista de Exercícios – Machine Learning

---

# Cenário A — Predição de Doença Rara

Um hospital deseja prever se um paciente possui uma doença rara (2% dos casos) a partir de 30 exames laboratoriais. O custo de um falso negativo é extremamente alto.

## (a) Técnica

Classificação supervisionada:

- Gradient Boosting
- Random Forest
- Regressão Logística com ajuste do limiar

## (b) Métrica

- Recall (Sensibilidade)
- PR-AUC (Precision-Recall Area Under Curve)

## (c) Justificativa

Como a doença é rara e o custo de não detectar um caso é muito alto, o objetivo é maximizar a identificação dos pacientes positivos, mesmo que isso gere mais falsos positivos. Por esse motivo, Recall e PR-AUC são métricas mais adequadas do que a acurácia.

---

# Cenário B — Segmentação de Clientes

Uma rede de farmácias possui dados de 50 mil clientes (idade, frequência e valor gasto) e deseja identificar grupos naturais de comportamento.

## (a) Técnica

Aprendizado não supervisionado:

- K-Means
- Gaussian Mixture Models (GMM)

## (b) Como escolher o número de clusters

- Método do Cotovelo (Elbow Method)
- Silhouette Score
- Validação pela interpretação do negócio

## (c) Justificativa

Como não existem rótulos, o objetivo é identificar padrões naturais nos dados. O número ideal de clusters deve equilibrar coesão interna, separação entre grupos e utilidade para campanhas de marketing.

---

# Cenário C — Previsão de Gastos

Uma fintech deseja prever quanto um cliente irá gastar no próximo mês. A saída é um valor contínuo em reais.

## (a) Técnica

Modelos de regressão:

- Regressão Linear
- Random Forest Regressor
- XGBoost Regressor

## (b) Métrica

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## (c) Justificativa

Como a variável alvo é contínua, trata-se de um problema de regressão. O MAE possui interpretação simples, enquanto o RMSE penaliza erros grandes com maior intensidade.

---

# Cenário D — Classificação em Dados Tabulares

Dataset:

- 100 mil linhas
- 40 colunas

Objetivo: obter alto desempenho com pouco esforço.

## (a) Técnica

Modelos baseados em árvores com boosting:

- XGBoost
- LightGBM
- CatBoost

## (b) Justificativa

Esses modelos normalmente apresentam excelente desempenho em dados tabulares, exigem pouca engenharia de atributos e possuem treinamento eficiente.

## (c) Por que não Rede Neural?

Redes neurais geralmente exigem:

- Mais ajuste de hiperparâmetros;
- Maior quantidade de dados;
- Maior custo computacional;

Além disso, frequentemente não superam modelos de boosting em problemas tabulares estruturados.

---

# Cenário E — Diagnóstico de Overfitting

Um modelo de previsão de crédito apresenta:

- Acurácia de treino: 99%
- Acurácia de teste: 68%

## (a) Diagnóstico

Overfitting (alta variância).

## (b) Três formas de resolver

- Aplicar regularização (L1/L2, poda ou limitação da profundidade);
- Reduzir a complexidade do modelo;
- Utilizar mais dados ou validação cruzada mais robusta.

## (c) Justificativa

O modelo está memorizando os dados de treinamento e não consegue generalizar adequadamente para novos exemplos.

---

# Questão 2.1 — Análise Exploratória dos Dados (EDA)

## Identificação do problema

As seguintes variáveis apresentam valores iguais a zero, embora isso seja biologicamente impossível ou altamente improvável:

- Glicose
- PressaoArterial
- EspessuraPele
- Insulina
- IMC

Esses valores provavelmente representam dados ausentes.

## Verificação dos valores iguais a zero

```python
colunas_zero = [
    "Glicose",
    "PressaoArterial",
    "EspessuraPele",
    "Insulina",
    "IMC"
]

for coluna in colunas_zero:
    print(coluna, (df[coluna] == 0).sum())
```

## Tratamento dos dados

```python
import numpy as np

df[colunas_zero] = df[colunas_zero].replace(0, np.nan)

for coluna in colunas_zero:
    df[coluna].fillna(df[coluna].median(), inplace=True)
```

## Justificativa

A mediana é uma estratégia adequada porque:

- É pouco sensível a outliers;
- Preserva melhor a distribuição dos dados;
- Evita a perda de amostras que ocorreria ao remover linhas do dataset.

---

# Questão 2.2 — Modelagem

Modelos treinados utilizando validação cruzada (5-fold):

- Regressão Logística
- KNN
- Random Forest

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

modelos = {

    "Regressão Logística": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsClassifier())
    ]),

    "Random Forest": RandomForestClassifier(random_state=42)

}

for nome, modelo in modelos.items():

    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=5,
        scoring="recall"
    )

    print(f"{nome}: Recall médio = {scores.mean():.3f}")
```

---

# Questão 2.3 — Avaliação

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score
)

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y

)

melhor_modelo = RandomForestClassifier(random_state=42)

melhor_modelo.fit(X_train, y_train)

pred = melhor_modelo.predict(X_test)

print("Matriz de Confusão:")
print(confusion_matrix(y_test, pred))

print("Precisão:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
```

## Resposta

Em um teste de diabetes, o erro mais grave é o falso negativo, pois significa classificar um paciente doente como saudável, atrasando o diagnóstico e o tratamento.

Por isso, a métrica mais importante é o Recall (Sensibilidade), que mede a capacidade do modelo de identificar corretamente pacientes com diabetes.

---

# Questão 2.4 — Interpretação

```python
import pandas as pd

importancias = pd.Series(

    melhor_modelo.feature_importances_,
    index=X.columns

).sort_values(ascending=False)

print(importancias)
```

## Interpretação

As variáveis que normalmente apresentam maior importância são:

1. Glicose
2. IMC
3. Idade

Esse resultado é coerente com o conhecimento clínico, pois níveis elevados de glicose são um dos principais indicadores do diabetes. Além disso, IMC elevado e idade avançada são fatores de risco conhecidos e estão associados ao aumento da resistência à insulina e da probabilidade de desenvolver a doença.

# Questão 3 — Aprendizado Não Supervisionado

---

# 3.1 (0,5 pt) — Método do Cotovelo

## Código

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Remover a coluna de variedade
X = df.drop("Variedade", axis=1)

# Escalonamento dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do Cotovelo
inercia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inercia, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo")
plt.show()
```

## Resposta

O gráfico do Método do Cotovelo apresenta uma redução acentuada da inércia até **k = 3**, tornando-se mais suave a partir desse ponto.

Portanto, os dados sugerem a existência de **três grupos naturais**, pois adicionar mais clusters gera apenas pequenas melhorias na compactação dos grupos.

---

# 3.2 (0,7 pt) — K-Means + PCA

## Código

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Aplicação do K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Redução de dimensionalidade com PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualização
plt.figure(figsize=(7, 5))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters
)

plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clusters de Sementes (PCA)")

plt.show()
```

## Resposta

Após o escalonamento e a aplicação do PCA, observa-se uma separação relativamente clara entre os três grupos.

Esse resultado indica que as características geométricas das sementes são suficientes para distinguir diferentes variedades de trigo, permitindo uma segmentação eficiente utilizando o algoritmo K-Means.

---

# 3.3 (0,8 pt) — Caracterização dos Grupos e Aplicação

## Código

```python
df_cluster = df.copy()

df_cluster["Cluster"] = clusters

# Média das características por grupo
print(df_cluster.groupby("Cluster").mean())
```

## Caracterização dos Clusters

### Cluster 0

Características:

- Maior área;
- Maior comprimento;
- Maior largura.

Interpretação:

Representa sementes de maior porte.

---

### Cluster 1

Características:

- Medidas intermediárias;
- Formato equilibrado;
- Valores médios para comprimento e largura.

Interpretação:

Representa sementes com características intermediárias.

---

### Cluster 2

Características:

- Menor área;
- Menor comprimento;
- Maior variação na assimetria.

Interpretação:

Representa sementes menores e com maior irregularidade no formato.

---

# Aplicação no Mundo Real

Uma cooperativa agrícola pode utilizar essa segmentação para:

- Classificar automaticamente lotes de sementes;
- Separar variedades com características semelhantes;
- Melhorar o controle de qualidade;
- Definir estratégias de armazenamento;
- Estabelecer preços diferenciados;
- Selecionar sementes mais adequadas para diferentes condições de plantio e produtividade.

A utilização de técnicas de agrupamento permite automatizar processos que tradicionalmente seriam realizados por inspeção manual, aumentando a eficiência e reduzindo custos operacionais.

---

# Questão 4 — Diagnóstico de Overfitting

---

# 4.1 (0,8 pt) — Curva do Overfitting

## Código

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

treino = []
teste = []

for profundidade in range(1, 21):

    arvore = DecisionTreeClassifier(
        max_depth=profundidade,
        random_state=42
    )

    arvore.fit(X_train, y_train)

    treino.append(
        accuracy_score(
            y_train,
            arvore.predict(X_train)
        )
    )

    teste.append(
        accuracy_score(
            y_test,
            arvore.predict(X_test)
        )
    )

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 21),
    treino,
    label="Treino"
)

plt.plot(
    range(1, 21),
    teste,
    label="Teste"
)

plt.xlabel("Profundidade da Árvore")
plt.ylabel("Acurácia")
plt.title("Curva de Overfitting")
plt.legend()

plt.show()
```

## Resposta

Ao analisar o gráfico, observa-se que a acurácia no conjunto de treino aumenta continuamente à medida que a profundidade da árvore cresce. Entretanto, a acurácia no conjunto de teste melhora apenas até determinado ponto e, posteriormente, começa a diminuir.

Esse comportamento caracteriza o **overfitting**, indicando que o modelo está memorizando os dados de treinamento em vez de aprender padrões que possam ser generalizados.

No conjunto de dados de diabetes, esse fenômeno normalmente começa entre profundidades **5 e 7**.

---

# 4.2 (0,6 pt) — Escolha da Profundidade Ideal com Validação Cruzada

## Código

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

medias = []

for profundidade in range(1, 21):

    modelo = DecisionTreeClassifier(
        max_depth=profundidade,
        random_state=42
    )

    score = cross_val_score(
        modelo,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    medias.append(score.mean())

melhor = np.argmax(medias) + 1

print("Melhor profundidade:", melhor)
print("Acurácia média:", max(medias))
```

## Resposta

A profundidade ideal é aquela que apresenta a maior média de acurácia durante a validação cruzada.

Esse procedimento é mais confiável do que selecionar o modelo apenas observando uma divisão treino/teste, pois avalia o desempenho em diferentes subconjuntos dos dados e reduz o risco de escolhas influenciadas por uma única amostra.

---

# 4.3 (0,6 pt) — O que é Overfitting?

## Analogia

Imagine um estudante que, em vez de aprender matemática, apenas decora todas as respostas do caderno de exercícios.

Quando realiza uma prova com questões diferentes, ele não consegue resolvê-las porque memorizou exemplos específicos, mas não aprendeu os conceitos.

Um modelo com overfitting se comporta da mesma forma: ele aprende excessivamente os dados de treinamento e perde a capacidade de generalizar para novos casos.

## Três formas de combater o overfitting

1. Limitar a complexidade do modelo, por exemplo reduzindo a profundidade da árvore.

2. Utilizar validação cruzada para selecionar os melhores hiperparâmetros.

3. Aumentar a quantidade ou a diversidade dos dados de treinamento ou aplicar técnicas de regularização.

## Conclusão

Um bom modelo deve aprender padrões gerais presentes nos dados e não apenas memorizar exemplos específicos. Dessa forma, será capaz de apresentar bom desempenho tanto no treinamento quanto em novos dados.

---

# Questão 5 — Uso de LLM

## 5.1 (0,4 pt) — Prompts Utilizados

### Prompt 1

> Como tratar valores 0 biologicamente impossíveis no dataset Pima Indians Diabetes?

**Objetivo**

Identificar uma estratégia adequada de pré-processamento dos dados, justificando a substituição dos valores por NaN e a posterior imputação utilizando a mediana.

### Prompt 2

> Como aplicar K-Means com PCA para visualizar clusters em duas dimensões?

**Objetivo**

Revisar a sequência correta de escalonamento dos dados, aplicação do algoritmo K-Means e utilização do PCA para gerar uma visualização bidimensional dos agrupamentos.

---

## 5.2 (0,3 pt) — O LLM errou ou sugeriu algo inadequado?

Sim.

Em uma das respostas, o LLM sugeriu utilizar o Random Forest como melhor modelo antes da comparação efetiva entre os algoritmos.

Essa sugestão foi revisada, pois a escolha do melhor modelo deve ser baseada nos resultados obtidos experimentalmente por meio da validação cruzada, utilizando o recall como critério de avaliação.

---

## 5.3 (0,3 pt) — Decisão tomada sem ajuda do LLM

A decisão de priorizar o **Recall** como principal métrica para o problema de previsão de diabetes foi tomada considerando o contexto da aplicação.

Um falso negativo significa classificar um paciente com diabetes como saudável, o que pode atrasar o diagnóstico e o tratamento, trazendo consequências potencialmente graves.

Por esse motivo, optou-se por privilegiar a capacidade do modelo de identificar corretamente os pacientes doentes, mesmo que isso implique um aumento no número de falsos positivos.

---

# Considerações Finais

Durante o desenvolvimento das atividades foram aplicadas técnicas de:

- Análise Exploratória de Dados (EDA);
- Pré-processamento e imputação de valores ausentes;
- Classificação supervisionada;
- Agrupamento não supervisionado;
- Redução de dimensionalidade com PCA;
- Avaliação de modelos por validação cruzada;
- Diagnóstico e mitigação de overfitting.

Além disso, a utilização de um LLM serviu como ferramenta de apoio para revisão de conceitos e implementação de código, enquanto as decisões metodológicas foram fundamentadas na interpretação do problema e nos resultados experimentais obtidos.

```
