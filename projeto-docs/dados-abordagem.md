# Dados

Os dados utilizados no projeto serão obtidos a partir de **bulas oficiais de medicamentos**, disponíveis em plataformas públicas e confiáveis, como o portal da **ANVISA** e sites farmacêuticos autorizados. Essas bulas, geralmente disponibilizadas em formato PDF, servirão como base de conhecimento para o chatbot.

Inicialmente, pretende-se trabalhar com um conjunto reduzido de bulas para fins de teste e validação do sistema, contendo medicamentos amplamente utilizados pela população. Posteriormente, a base poderá ser expandida para incluir uma maior variedade de medicamentos e categorias farmacêuticas.

Os arquivos coletados passarão por um processo de extração e tratamento de texto, permitindo que as informações sejam organizadas e utilizadas pelo modelo de inteligência artificial. Não será necessário criar dados manualmente, pois a maior parte do conteúdo será obtida de fontes públicas já disponíveis.

# Abordagem Técnica

O projeto utilizará uma arquitetura baseada em **LLM + RAG (Retrieval-Augmented Generation)** para permitir que o chatbot responda perguntas utilizando informações presentes nas bulas dos medicamentos.

As bulas serão processadas com o auxílio do framework **LangChain**, responsável pela extração, divisão e preparação dos textos para consulta. Em seguida, os conteúdos serão convertidos em embeddings e armazenados no banco vetorial **ChromaDB**, permitindo buscas semânticas mais eficientes.

Quando o usuário realizar uma pergunta, o sistema buscará trechos relevantes das bulas armazenadas no banco vetorial e utilizará essas informações como contexto para gerar respostas mais precisas e contextualizadas por meio de um modelo de linguagem.

Além disso, será desenvolvida uma **API** responsável por conectar o chatbot à base de conhecimento, permitindo o envio de perguntas e o retorno das respostas de forma organizada e escalável.
