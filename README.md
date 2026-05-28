# 🩺 Chatbot de Saúde com RAG

Repositório destinado ao projeto da disciplina de Inteligência Artificial, desenvolvido utilizando técnicas de LLM + RAG (Retrieval-Augmented Generation) para responder perguntas sobre medicamentos com base em bulas oficiais.

# 👥 Componentes do Grupo
- Elton Silva
- Vitória Ataide
- Kevin Silva
- Hudson Silva (Líder)

# 📚 Sobre o Projeto

Atualmente, muitas pessoas possuem dificuldades para compreender informações presentes em bulas de medicamentos devido à linguagem técnica e à grande quantidade de informações contidas nesses documentos. Questões relacionadas à posologia, contraindicações, efeitos colaterais e formas corretas de uso são frequentes, levando usuários a recorrerem a fontes nem sempre confiáveis.

O objetivo deste projeto é desenvolver um chatbot inteligente capaz de responder perguntas sobre medicamentos utilizando informações extraídas diretamente de bulas oficiais como base de conhecimento. A solução utilizará técnicas de recuperação de informação e inteligência artificial generativa para fornecer respostas contextualizadas, rápidas e acessíveis aos usuários.

# 🧠 Tecnologias Utilizadas

- Python
- LangChain
- ChromaDB
- API REST
- LLM (Large Language Model)
- Embeddings
- Processamento de PDFs

# ⚙️ Funcionalidades Esperadas

- Processamento de bulas em formato PDF;
- Extração e organização de textos;
- Armazenamento vetorial utilizando ChromaDB;
- Busca semântica de informações;
- Geração de respostas contextualizadas com RAG;
- API para comunicação com o chatbot;
- Consulta sobre medicamentos e informações farmacêuticas.

# 📁 Estrutura do Projeto
projeto-ia-chatbot-saude/

 │

 ├── projeto-docs/

 │   ├── titulo-problema.md

 │   ├── dados-abordagem.md

 │   └── resultado-metricas.md
 
 │

├── README.md


# 📈 Resultado Esperado

Espera-se desenvolver um chatbot inteligente capaz de responder perguntas relacionadas a medicamentos utilizando informações extraídas diretamente de bulas oficiais como base de conhecimento. O sistema deverá fornecer respostas contextualizadas, rápidas e mais acessíveis aos usuários, auxiliando na compreensão de informações importantes sobre medicamentos.

O projeto pretende entregar uma solução funcional composta por:

* Um chatbot para consultas sobre medicamentos;
* Integração com modelo de linguagem utilizando RAG;
* Banco vetorial com armazenamento das informações das bulas;
* API para comunicação entre o sistema e a base de conhecimento;
* Processamento e recuperação de informações relevantes por meio do LangChain e ChromaDB.

Além disso, espera-se que o sistema demonstre a aplicação prática de técnicas modernas de Inteligência Artificial Generativa em um contexto de saúde e recuperação de informações.

# Métricas de Sucesso

Para avaliar a eficiência e a qualidade do projeto, serão consideradas as seguintes métricas:

* **Precisão das respostas:** capacidade do chatbot de responder corretamente com base nas informações presentes nas bulas;
* **Qualidade da recuperação de dados:** eficiência na busca dos trechos mais relevantes dentro da base vetorial;
* **Tempo de resposta:** velocidade média para processar perguntas e retornar respostas ao usuário;
* **Clareza das respostas:** capacidade do sistema de apresentar informações compreensíveis e organizadas;
* **Funcionamento da API:** estabilidade na comunicação entre o chatbot e a base de conhecimento;
* **Testes práticos com usuários:** avaliação manual da utilidade e coerência das respostas fornecidas pelo sistema.

# 📌 Status do Projeto

🚧 Projeto em desenvolvimento.
