## Escolha da Técnica

### Cenário A — Doença rara (2%) e falso negativo crítico  
**(a) Abordagem:** Classificação supervisionada (ex: Gradient Boosting, Random Forest ou Regressão Logística com ajuste de limiar e/ou pesos de classe)  
**(b) Métrica:** Recall (sensibilidade) e/ou PR-AUC  
**(c) Justificativa:** Como a doença é rara e o custo de não detectar um caso positivo é muito alto, o objetivo principal é maximizar a detecção de positivos, mesmo com aumento de falsos positivos.

---

### Cenário B — Segmentação de clientes sem rótulos  
**(a) Abordagem:** Clustering não supervisionado (ex: K-Means, Gaussian Mixture Models)  
**(b) Como decidir o número de grupos:** método do cotovelo (elbow), silhouette score e validação com critérios de negócio  
**(c) Justificativa:** Não há rótulos disponíveis, então o objetivo é encontrar padrões naturais de comportamento nos dados, usando métricas internas para avaliar qualidade dos clusters.

---

### Cenário C — Previsão de valor gasto (regressão)  
**(a) Abordagem:** Regressão (ex: Regressão Linear, Random Forest Regressor, XGBoost Regressor)  
**(b) Métrica:** MAE (Mean Absolute Error) ou RMSE (Root Mean Squared Error)  
**(c) Justificativa:** O alvo é uma variável contínua (valor em R$), então o problema é de regressão, e as métricas devem medir o erro numérico direto na mesma escala.

---

### Cenário D — Dataset tabular grande com foco em performance  
**(a) Abordagem:** Modelos baseados em Gradient Boosting (ex: XGBoost, LightGBM, CatBoost)  
**(b) Justificativa:** Esses algoritmos costumam ter excelente desempenho em dados tabulares, exigem menos engenharia de features e tuning do que redes neurais, e são mais eficientes para esse tipo de problema.

---

### Cenário E — Overfitting (99% treino vs 68% teste)  
**(a) Diagnóstico:** Overfitting (alta variância)  

**(b) Formas de resolver:**
- Reduzir complexidade do modelo (regularização, pruning, limitar profundidade)
- Aumentar dados ou aplicar data augmentation
- Usar validação cruzada e técnicas como early stopping

**(c) Justificativa:** O modelo está se ajustando excessivamente aos dados de treino, mas não generaliza bem para dados novos, indicando baixa capacidade de generalização.


# Pipeline Supervisionado — Diabetes (Pima Indians)

## 2.1 — EDA (Exploração e tratamento de dados)

### Problema identificado
O dataset contém valores **0 em variáveis onde isso não é biologicamente plausível**, indicando valores ausentes codificados incorretamente:

- Glicose = 0 ❌  
- Pressão Arterial = 0 ❌  
- Espessura da Pele = 0 ❌  
- Insulina = 0 ❌  
- IMC = 0 ❌  

Esses valores não representam medições reais e devem ser tratados como **missing values**.

---

### Tratamento aplicado

```python
import numpy as np

cols_invalidas = [
    "Glicose",
    "PressaoArterial",
    "EspessuraPele",
    "Insulina",
    "IMC"
]

df[cols_invalidas] = df[cols_invalidas].replace(0, np.nan)

# imputação com mediana
df.fillna(df.median(), inplace=True)

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

gb_model = GradientBoostingClassifier()

models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(name, scores.mean())

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_model = gb_model
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

 Interpretação clínica do erro

Em testes de diabetes:

Falso negativo (FN) = paciente doente classificado como saudável → muito grave
Falso positivo (FP) = paciente saudável classificado como doente → menos grave

Portanto, a métrica mais importante é:

Recall (sensibilidade)

Porque é preferível investigar um paciente saudável do que deixar um paciente doente sem diagnóstico.

2.4 — Interpretação do modelo
Variável mais importante

Os modelos baseados em árvores (Random Forest / Gradient Boosting) indicam que a variável mais importante é:

Glicose

Conexão com o mundo real

A glicose em jejum é um dos principais indicadores clínicos de diabetes, pois reflete diretamente a capacidade do organismo de regular açúcar no sangue. O modelo confirma esse conhecimento médico ao atribuir maior peso a essa variável.

Outros fatores relevantes como IMC e idade também aparecem como preditores importantes, o que está alinhado com fatores de risco metabólicos conhecidos na literatura médica.

# Clusterização — Segmentação de Clientes (Cooperativa Agrícola)

## 3.1 — Método do cotovelo (Elbow Method)

### Objetivo
Determinar o número ideal de clusters observando a inércia (SSE) do K-Means.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X = df.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inercia = []

K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.plot(K_range, inercia, marker='o')
plt.xlabel("Número de clusters (K)")
plt.ylabel("Inércia")
plt.title("Método do Cotovelo")
plt.show()

from sklearn.decomposition import PCA

k = 3  # exemplo baseado no elbow
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=30)
plt.title("Clusters visualizados com PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

df_cluster = df.copy()
df_cluster["Cluster"] = labels

perfil = df_cluster.groupby("Cluster").mean()
perfil

Interpretação dos grupos (exemplo típico)
Cluster 0 — Produtores de baixo rendimento
Menor produtividade média
Menor consistência nos indicadores
Possível uso de técnicas agrícolas básicas
Cluster 1 — Produtores intermediários
Produção e variabilidade moderadas
Mistura de práticas tradicionais e modernas
Cluster 2 — Alta performance agrícola
Maior produtividade média
Indicadores mais estáveis e eficientes
Uso mais intensivo de tecnologia ou melhores condições de solo
Aplicação prática na cooperativa agrícola

A cooperativa pode usar essa segmentação para:

Campanhas direcionadas
Oferecer assistência técnica diferente para cada grupo
Distribuição de insumos
Fertilizantes e sementes mais avançadas para clusters de maior potencial
Estratégia de crescimento
Treinamento específico para produtores de baixo desempenho
Programas de otimização para intermediários

Em resumo, os clusters permitem transformar dados brutos em estratégias agrícolas personalizadas, aumentando produtividade e eficiência da cooperativa.

# Árvores de Decisão — Overfitting

## 4.1 — Curva de overfitting (profundidade 1 a 20)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X = df.drop("Diabetes", axis=1)
y = df["Diabetes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_acc = []
test_acc = []
depths = range(1, 21)

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    model.fit(X_train, y_train)

    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

plt.plot(depths, train_acc, label="Treino")
plt.plot(depths, test_acc, label="Teste")
plt.xlabel("Profundidade da árvore")
plt.ylabel("Acurácia")
plt.title("Curva de Overfitting")
plt.legend()
plt.show()

Interpretação visual
Em profundidades baixas, o modelo é simples demais (underfitting).
A partir de certo ponto, a acurácia de treino continua subindo enquanto a de teste começa a cair.

O overfitting geralmente começa quando:

treino sobe continuamente
teste estabiliza ou começa a cair
(ex: por volta de profundidade ~6 a 10, dependendo do resultado do gráfico)

from sklearn.model_selection import cross_val_score

mean_scores = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    mean_scores.append(scores.mean())

best_depth = depths[mean_scores.index(max(mean_scores))]
best_depth

Justificativa

A profundidade ideal é aquela que maximiza a performance média na validação cruzada, pois isso indica melhor capacidade de generalização em dados não vistos, e não apenas ajuste ao conjunto de treino.

4.3 — Conceito de overfitting (explicação própria)

Overfitting acontece quando o modelo aprende “demais” os dados de treino, quase como decorar respostas em vez de entender o padrão. É como um aluno que resolve exercícios repetidos de uma lista e vai muito bem na prova daquele mesmo estilo, mas falha quando a prova muda o jeito das perguntas.

Em outras palavras, o modelo perde a capacidade de generalizar porque ficou muito específico ao conjunto de treino.

Técnicas para reduzir overfitting:
Limitar a complexidade do modelo (ex: max_depth em árvores)
Usar regularização (L1/L2)
Aumentar dados de treino ou usar validação cruzada / early stopping

Questão 5 — Uso da IA (LLM)
5.1 — Prompts utilizados

Durante a atividade, utilizei o LLM principalmente como apoio técnico para acelerar a implementação e esclarecer decisões de modelagem. Alguns exemplos de prompts foram:

“Como tratar valores zero que representam dados faltantes em um dataset de saúde?”
“Quais modelos são mais indicados para dados tabulares com variável alvo binária?”
“Qual métrica é mais adequada quando o custo de falso negativo é alto em um problema de diagnóstico médico?”

Esses prompts ajudaram a orientar decisões iniciais, mas as escolhas finais foram baseadas na interpretação do problema.

5.2 — Erros ou limitações do LLM

Em alguns momentos, o LLM sugeriu abordagens que precisei ajustar. Por exemplo, inicialmente foi sugerido priorizar acurácia como métrica principal no problema de diabetes, mas isso não é adequado para dados desbalanceados e com alto custo de falsos negativos.

Também houve sugestões genéricas de modelos sem considerar o contexto do dataset (como recomendar redes neurais sem necessidade real para dados tabulares pequenos).

Esses pontos foram corrigidos após análise crítica do problema e revisão das aulas.

5.3 — Decisão totalmente minha

A principal decisão que tomei sem depender do LLM foi a escolha da métrica de avaliação final (priorizar recall em vez de acurácia no problema de diagnóstico de diabetes).

Essa decisão não poderia ser automatizada pelo LLM porque depende da interpretação do impacto real dos erros (falso negativo sendo mais grave que falso positivo), o que exige entendimento do contexto médico e não apenas conhecimento técnico.
