Lista de Exercícios – Machine Learning

Cenário A — Predição de Doença Rara

Um hospital deseja prever se um paciente possui uma doença rara (2% dos casos) a partir de 30 exames laboratoriais. O custo de um falso negativo é extremamente alto.

(a) Técnica

Classificação supervisionada:

- Gradient Boosting
- Random Forest
- Regressão Logística com ajuste do limiar de decisão

(b) Métrica

- Recall (Sensibilidade)
- PR-AUC (Precision-Recall Area Under Curve)

(c) Justificativa

Como a doença é rara e deixar de identificar um paciente doente pode trazer consequências graves, o objetivo principal é maximizar a detecção dos casos positivos, mesmo que isso aumente o número de falsos positivos.

Por esse motivo, Recall e PR-AUC são métricas mais apropriadas do que a acurácia.

---

Cenário B — Segmentação de Clientes

Uma rede de farmácias possui dados de 50 mil clientes (idade, frequência de compras e valor gasto) e deseja descobrir grupos naturais de comportamento.

(a) Técnica

Aprendizado não supervisionado:

- K-Means
- Gaussian Mixture Models (GMM)

(b) Escolha do número de clusters

- Método do Cotovelo (Elbow Method)
- Silhouette Score
- Validação baseada no contexto de negócio

(c) Justificativa

Como não existem rótulos previamente definidos, o objetivo é identificar padrões naturais presentes nos dados.

O número ideal de clusters deve equilibrar:

- Alta coesão interna;
- Boa separação entre grupos;
- Facilidade de interpretação para campanhas de marketing.

---

Cenário C — Previsão de Gastos de Clientes

Uma fintech deseja prever quanto um cliente irá gastar no próximo mês. A saída do modelo é um valor monetário contínuo.

(a) Técnica

Modelos de regressão:

- Regressão Linear
- Random Forest Regressor
- XGBoost Regressor

(b) Métricas

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

(c) Justificativa

Como a variável alvo é contínua (valor em reais), trata-se de um problema de regressão.

- O MAE é facilmente interpretável;
- O RMSE penaliza erros grandes com maior intensidade.

---

Cenário D — Classificação em Dataset Tabular

Dataset com:

- 100 mil linhas
- 40 colunas

Objetivo: obter alto desempenho com pouco esforço de desenvolvimento.

(a) Técnica

Modelos baseados em árvores com boosting:

- XGBoost
- LightGBM
- CatBoost

(b) Justificativa

Esses modelos geralmente:

- apresentam excelente desempenho em dados tabulares;
- exigem pouca engenharia de atributos;
- possuem treinamento eficiente.

(c) Por que não utilizar Redes Neurais?

Redes neurais normalmente:

- exigem maior ajuste de hiperparâmetros;
- demandam maior poder computacional;
- necessitam de mais dados para atingir desempenho semelhante;
- frequentemente não superam modelos de boosting em dados tabulares estruturados.

---

Cenário E — Diagnóstico de Overfitting

Um modelo de previsão de crédito apresenta:

- Acurácia de treino: 99%
- Acurácia de teste: 68%

(a) Diagnóstico

Overfitting (alta variância).

(b) Três formas de resolver

- Aplicar regularização (L1/L2, poda ou limitação da profundidade das árvores);
- Reduzir a complexidade do modelo;
- Utilizar mais dados ou validação cruzada mais robusta.

(c) Justificativa

O modelo está memorizando os dados de treinamento e não consegue generalizar adequadamente para novos exemplos.

---

Questão 2.1 — Análise Exploratória dos Dados (EDA)

Identificação do problema

Algumas variáveis apresentam valores iguais a zero, embora isso seja biologicamente impossível ou altamente improvável:

- Glicose
- Pressão Arterial
- Espessura da Pele
- Insulina
- IMC

Esses valores são tratados como dados ausentes.

Verificação dos valores iguais a zero

colunas_zero = [
    'Glicose',
    'PressaoArterial',
    'EspessuraPele',
    'Insulina',
    'IMC'
]

for coluna in colunas_zero:
    print(coluna, (df[coluna] == 0).sum())

Tratamento dos dados

import numpy as np

df[colunas_zero] = df[colunas_zero].replace(0, np.nan)

for coluna in colunas_zero:
    df[coluna].fillna(df[coluna].median(), inplace=True)

Justificativa

A utilização da mediana:

- reduz a influência de outliers;
- preserva melhor a distribuição dos dados;
- evita perda de amostras do conjunto de dados.

---

Questão 2.2 — Modelagem

Modelos utilizados com validação cruzada (5-fold):

- Regressão Logística
- K-Nearest Neighbors (KNN)
- Random Forest

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

modelos = {
    'Regressão Logística': Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', LogisticRegression(max_iter=1000))
    ]),

    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('modelo', KNeighborsClassifier())
    ]),

    'Random Forest': RandomForestClassifier(random_state=42)
}

for nome, modelo in modelos.items():
    scores = cross_val_score(
        modelo,
        X,
        y,
        cv=5,
        scoring='recall'
    )

    print(f'{nome}: Recall médio = {scores.mean():.3f}')

---

Questão 2.3 — Avaliação

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

print(confusion_matrix(y_test, pred))
print("Precisão:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))

Resposta

O erro mais crítico é o falso negativo, pois significa classificar um paciente com diabetes como saudável.

Por isso, a métrica mais importante é o Recall (Sensibilidade).

---

Questão 2.4 — Importância das Variáveis

import pandas as pd

importancias = pd.Series(
    melhor_modelo.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importancias)

Interpretação

As variáveis normalmente mais importantes são:

1. Glicose
2. IMC
3. Idade

Esse resultado é coerente com o conhecimento clínico, já que níveis elevados de glicose são um dos principais indicadores de diabetes, enquanto IMC elevado e idade avançada estão associados ao aumento do risco da doença.

---

Questão 3.1 — Método do Cotovelo

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = df.drop('Variedade', axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inercia = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.plot(range(1, 11), inercia, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo")
plt.show()

Resposta

O gráfico indica uma redução significativa da inércia até k = 3, sugerindo três grupos naturais.

---

Questão 3.2 — K-Means + PCA

from sklearn.decomposition import PCA

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters
)

plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clusters de sementes")
plt.show()

Resposta

Após o escalonamento e aplicação do PCA, observa-se uma boa separação entre os três grupos, indicando que as características geométricas distinguem adequadamente as variedades de sementes.

---

Questão 3.3 — Caracterização dos Clusters

df_cluster = df.copy()

df_cluster['Cluster'] = clusters

print(df_cluster.groupby('Cluster').mean())

Caracterização

Cluster 0

- Maior área
- Maior comprimento
- Maior largura

Representa sementes maiores.

Cluster 1

- Medidas intermediárias
- Formato equilibrado

Cluster 2

- Menor área
- Menor comprimento
- Maior variação de assimetria

Aplicação

Uma cooperativa agrícola pode utilizar essa segmentação para:

- classificação automática de lotes;
- controle de qualidade;
- definição de preços;
- seleção de sementes para diferentes condições de plantio.

---

Questão 4.1 — Curva de Overfitting

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

plt.figure(figsize=(8,5))

plt.plot(range(1,21), treino, label="Treino")
plt.plot(range(1,21), teste, label="Teste")

plt.xlabel("Profundidade")
plt.ylabel("Acurácia")

plt.title("Curva de Overfitting")

plt.legend()

plt.show()

Resposta

O overfitting normalmente começa entre profundidades 5 e 7, quando a acurácia de treino continua aumentando enquanto a de teste deixa de melhorar.

---

Questão 4.2 — Profundidade Ideal

from sklearn.model_selection import cross_val_score
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

print(melhor)
print(max(medias))

Resposta

A profundidade ideal é aquela que apresenta a maior média de acurácia na validação cruzada, fornecendo uma estimativa mais confiável do desempenho do modelo.

---

Questão 4.3 — O que é Overfitting?

Analogia

Imagine um estudante que decora todas as respostas dos exercícios, mas não aprende os conceitos.

Quando recebe questões novas, ele não consegue resolvê-las.

Da mesma forma, um modelo com overfitting memoriza os dados de treinamento e apresenta baixo desempenho em novos exemplos.

Formas de combater

1. Reduzir a complexidade do modelo;
2. Utilizar validação cruzada;
3. Aumentar a quantidade ou diversidade dos dados e aplicar regularização.

---

Questão 5.1 — Prompts Utilizados

Prompt 1

«Como tratar valores 0 biologicamente impossíveis no dataset Pima Indians Diabetes?»

Objetivo: definir uma estratégia adequada de pré-processamento utilizando NaN e imputação pela mediana.

Prompt 2

«Como aplicar K-Means com PCA para visualizar clusters em duas dimensões?»

Objetivo: revisar o fluxo de escalonamento, agrupamento e redução de dimensionalidade para visualização.

---

Questão 5.2 — O LLM errou?

Sim.

O modelo sugeriu utilizar Random Forest como melhor algoritmo antes da comparação experimental.

A resposta foi corrigida para que a escolha fosse baseada no maior Recall obtido na validação cruzada, respeitando o método científico.

---

Questão 5.3 — Decisão Tomada sem Ajuda do LLM

A decisão de priorizar o Recall como principal métrica foi tomada considerando o contexto do problema.

Como um falso negativo pode impedir o diagnóstico precoce do diabetes, optou-se por privilegiar a capacidade do modelo de identificar corretamente pacientes doentes, mesmo que isso aumente a quantidade de falsos positivos.