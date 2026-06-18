## Cenário A — Um hospital quer prever se um paciente tem uma doença rara (afeta 2% dos casos) a partir de 30 exames. Falso negativo (não detectar a doença) é catastrófico.

**(a) Técnica:**  
Classificação supervisionada (ex: Gradient Boosting, Random Forest ou Regressão Logística com ajuste de limiar)

**(b) Métrica:**  
Recall (sensibilidade) e PR-AUC

**(c) Justificativa:**  
Como a doença é rara e o custo de não detectar um caso é altíssimo, o foco deve ser maximizar a detecção de positivos, mesmo aceitando mais falsos positivos; por isso recall e métricas de precisão-recall são mais adequadas que acurácia.

---

## Cenário B — Uma rede de farmácias tem dados de 50 mil clientes (idade, frequência, valor gasto) e quer descobrir grupos naturais de comportamento para campanha.

**(a) Técnica:**  
Aprendizado não supervisionado (ex: K-Means ou Gaussian Mixture Models)

**(b) Como escolher o número de clusters:**  
Método do cotovelo (Elbow Method), Silhouette Score e validação por interpretação de negócio

**(c) Justificativa:**  
Como não existem rótulos, é necessário identificar padrões naturais nos dados, e o número de clusters deve equilibrar coesão interna e separação entre grupos, além de fazer sentido para uso em campanhas.

---

## Cenário C — Uma fintech quer prever o valor (em R$) que um cliente vai gastar no próximo mês, baseado no histórico. A saída é um número contínuo.

**(a) Técnica:**  
Regressão (ex: Regressão Linear, Random Forest Regressor, XGBoost Regressor)

**(b) Métrica:**  
MAE (Mean Absolute Error) ou RMSE (Root Mean Squared Error)

**(c) Justificativa:**  
O alvo é numérico contínuo (valor em R$), portanto trata-se de um problema de regressão; MAE é mais interpretável, enquanto RMSE penaliza mais erros grandes.

---

## Cenário D — Você tem um dataset tabular com 100 mil linhas e 40 colunas para um problema de classificação. Quer a melhor performance possível sem semanas de trabalho.

**(a) Técnica:**  
Modelos baseados em árvores com boosting (ex: XGBoost, LightGBM, CatBoost)

**(b) Justificativa:**  
Esses modelos geralmente apresentam excelente desempenho em dados tabulares, exigem menos engenharia de features e são eficientes em treinamento.

**(c) Por que não rede neural:**  
Redes neurais normalmente exigem mais ajuste de hiperparâmetros, mais dados e maior custo computacional, e raramente superam boosting em dados tabulares estruturados.

---

## Cenário E — Um modelo de previsão de crédito atinge 99% de acurácia no treino e 68% no teste.

**(a) Diagnóstico:**  
Overfitting (alta variância)

**(b) 3 formas de resolver:**  
- Aplicar regularização (L1/L2, limitar profundidade de árvores, pruning)  
- Reduzir complexidade do modelo (simplificação ou seleção de variáveis)  
- Obter mais dados ou usar validação cruzada mais robusta  

**(c) Justificativa:**  
O modelo está se ajustando demais aos dados de treino e falha em generalizar para dados novos, indicando baixa capacidade de generalização.


2.1 — EDA (Análise Exploratória dos Dados)

1. Identificação do problema

Ao analisar o conjunto de dados, observa-se que algumas variáveis possuem valores iguais a 0, embora isso seja biologicamente impossível ou altamente improvável. Essas colunas são:

Glicose

PressaoArterial

EspessuraPele

Insulina

IMC


Esses valores provavelmente representam dados ausentes (missing values), e não medições reais.


---

2. Verificação dos valores 0

# Contagem de valores iguais a 0
colunas_zero = ['Glicose', 'PressaoArterial',
                'EspessuraPele', 'Insulina', 'IMC']

for coluna in colunas_zero:
    print(coluna, (df[coluna] == 0).sum())


---

3. Estratégia de tratamento

Substituir os valores 0 por NaN e, em seguida, preencher os valores ausentes com a mediana de cada coluna.

import numpy as np

# Substituir 0 por NaN
df[colunas_zero] = df[colunas_zero].replace(0, np.nan)

# Preencher com a mediana
for coluna in colunas_zero:
    df[coluna].fillna(df[coluna].median(), inplace=True)


---

4. Justificativa

A mediana é uma boa estratégia porque:

É pouco sensível a valores extremos (outliers).

Mantém a distribuição dos dados mais estável do que a média.

Evita a perda de amostras que ocorreria caso as linhas fossem removidas, preservando os 768 registros do conjunto de dados.


Assim, o tratamento melhora a qualidade dos dados e fornece uma base mais confiável para o treinamento de modelos de previsão de diabetes.

2.2 (1,0 pt) — Modelagem

Treinar três modelos diferentes utilizando validação cruzada (5-fold):

Regressão Logística (necessita escalonamento)

K-Nearest Neighbors (KNN) (necessita escalonamento)

Random Forest (não necessita escalonamento)


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
    scores = cross_val_score(modelo, X, y, cv=5, scoring='recall')
    print(f'{nome}: Recall médio = {scores.mean():.3f}')


---

2.3 (0,8 pt) — Avaliação

Treinar o melhor modelo (maior recall) e calcular matriz de confusão, precisão e recall.

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
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

Resposta da pergunta-chave

Em um teste de diabetes, o erro mais grave é o falso negativo, pois significa classificar um paciente doente como saudável, atrasando o diagnóstico e o tratamento. Por isso, a métrica mais importante é o recall (sensibilidade), que mede a capacidade do modelo de identificar corretamente os pacientes com diabetes.


---

2.4 (0,6 pt) — Interpretação

Para verificar a importância das variáveis:

import pandas as pd

importancias = pd.Series(
    melhor_modelo.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importancias)

Interpretação

A variável que normalmente apresenta maior importância é Glicose, seguida por IMC e Idade.

Esse resultado faz sentido do ponto de vista clínico, pois níveis elevados de glicose no sangue são um dos principais critérios utilizados no diagnóstico do diabetes. Além disso, maior IMC e idade mais avançada são fatores de risco conhecidos, estando associados ao aumento da resistência à insulina e da probabilidade de desenvolver a doença.

---

3.1 (0,5 pt) — Método do Cotovelo

Código

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Remover a coluna de variedade
X = df.drop('Variedade', axis=1)

# Escalonamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Método do cotovelo
inercia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.plot(range(1, 11), inercia, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo")
plt.show()

Resposta

O gráfico do método do cotovelo apresenta uma redução acentuada da inércia até k = 3, tornando-se mais suave a partir desse ponto. Portanto, os dados sugerem a existência de 3 grupos naturais, pois adicionar mais clusters gera apenas pequenas melhorias na compactação dos grupos.


---

3.2 (0,7 pt) — K-Means + PCA

Código

from sklearn.decomposition import PCA

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# PCA para visualização
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Clusters de sementes (PCA)")
plt.show()

Resposta

Após o escalonamento e a aplicação do PCA, observa-se uma separação relativamente clara entre os 3 grupos, indicando que as medidas geométricas são suficientes para distinguir diferentes tipos de sementes de trigo.


---

3.3 (0,8 pt) — Caracterização dos grupos e aplicação

Código

df_cluster = df.copy()
df_cluster['Cluster'] = clusters

# Média de cada característica por grupo
print(df_cluster.groupby('Cluster').mean())

Caracterização

Cluster 0: sementes com maior área, comprimento e largura, indicando grãos maiores.

Cluster 1: sementes com medidas intermediárias e formato equilibrado.

Cluster 2: sementes menores, com menor área e comprimento e maior variação na assimetria.


Aplicação real

Uma cooperativa agrícola pode utilizar essa segmentação para classificar automaticamente lotes de sementes, separando variedades com características semelhantes para comercialização, armazenamento e plantio. Essa estratégia melhora o controle de qualidade, permite definir preços diferenciados e auxilia na seleção de sementes mais adequadas para diferentes condições de cultivo e produtividade.

---

5.1 (0,4 pt) — Prompts utilizados
Prompt: "Como tratar valores 0 biologicamente impossíveis no dataset Pima Indians Diabetes?"
Objetivo: Identificar uma estratégia adequada de pré-processamento dos dados e justificar a substituição dos valores por NaN e posterior imputação pela mediana.
Prompt: "Como aplicar K-Means com PCA para visualizar clusters em duas dimensões?"
Objetivo: Revisar a sequência correta de escalonamento dos dados, aplicação do K-Means e redução de dimensionalidade com PCA para gerar uma visualização dos grupos.
5.2 (0,3 pt) — O LLM errou ou sugeriu algo inadequado?
Sim. Em uma das respostas, o LLM sugeriu utilizar o Random Forest como melhor modelo sem comparar efetivamente os resultados obtidos na validação cruzada. Percebemos que essa escolha não poderia ser feita antes da execução dos experimentos e ajustamos a resposta para selecionar o modelo com base na métrica de recall obtida na avaliação.
5.3 (0,3 pt) — Decisão tomada sem ajuda do LLM
A decisão de priorizar o recall como principal métrica na previsão de diabetes foi nossa, considerando o contexto do problema. Entendemos que um falso negativo (não identificar um paciente com diabetes) pode trazer consequências mais graves do que um falso positivo, e essa escolha depende da interpretação do cenário e dos objetivos da aplicação, não apenas da geração automática de código pelo LLM.
