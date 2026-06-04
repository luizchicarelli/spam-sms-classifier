# 📩 Detecção de Spam em Mensagens SMS

Projeto Avaliativo **P2** — Disciplina **A.I - Classificadores** · Universidade de Marília (UNIMAR) · 2026/1

Sistema de classificação que identifica automaticamente se uma mensagem SMS é **spam** (indesejada) ou **ham** (legítima), com aplicação web interativa em Streamlit.

---

## 👥 Integrantes e RAs

| Nome | RA |
|------|----|
| Luiz Henrique Soares Chicareli de Andrade | 2035693 |
| Gabriel Almeida Ermenegildo | 2028344 |
| Enzo Luiz Tsutsumi de Almeida José | 2027602 |

> **Grupo:** 5

---

## 🎯 Descrição do problema

O spam por SMS é uma nuisance global e um vetor comum de **fraudes e phishing**. Filtros automáticos
são essenciais para proteger usuários de conteúdo indesejado. O desafio é construir um classificador
que distinga, a partir apenas do texto, mensagens legítimas de mensagens de spam — em um cenário
**desbalanceado**, já que o spam é minoria (~13% das mensagens).

## 🎯 Objetivo do projeto

Treinar, comparar e disponibilizar um modelo de Machine Learning capaz de classificar mensagens SMS
como **ham** ou **spam**, escolhendo o melhor classificador por métricas adequadas a dados
desbalanceados e publicando-o em uma aplicação web funcional.

## 📊 Dataset utilizado

- **Nome:** SMS Spam Collection (UCI)
- **Fonte:** [Kaggle — uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Tamanho original:** 5.572 mensagens · **após remoção de 403 duplicatas:** 5.169 mensagens
- **Distribuição (sem duplicatas):** 4.516 ham (87,4%) · 653 spam (12,6%)
- **Colunas úteis:** `v1` (rótulo: ham/spam) e `v2` (texto da mensagem)

## 🤖 Tipo de problema de Machine Learning

**Classificação binária supervisionada** com **classes desbalanceadas** (NLP — processamento de
linguagem natural).

---

## 🔬 Metodologia

1. **Análise Exploratória (EDA):** distribuição das classes, tamanho médio das mensagens por classe e
   nuvens de palavras (ham × spam).
2. **Tratamento dos dados:** remoção de 403 mensagens duplicadas (evita *data leakage*).
3. **Pré-processamento de texto:** minúsculas, remoção de pontuação/números e normalização de espaços.
4. **Vetorização:** `TfidfVectorizer` com unigramas + bigramas (`ngram_range=(1,2)`), `sublinear_tf` e
   remoção de stopwords.
5. **Divisão estratificada:** 70% treino · 15% validação · 15% teste (mantendo a proporção de spam).
6. **Pipelines:** TF-IDF + classificador encapsulados em `Pipeline` (vetorização ajustada dentro de
   cada *fold* da validação cruzada → sem vazamento).
7. **Otimização:** `GridSearchCV` com `StratifiedKFold (k=5)`, otimizando F1.
8. **Avaliação:** acurácia, precisão, recall, F1, AUC-ROC e AUC-PR; matrizes de confusão, curvas
   ROC e Precision-Recall, análise de erros, *feature importance* e análise de limiar.
9. **Modelo final:** re-treinado em todos os dados rotulados e serializado com `joblib`.

## 🧪 Modelos treinados

| Modelo | Descrição |
|--------|-----------|
| **MultinomialNB** | Naive Bayes multinomial — baseline clássico e forte para texto |
| **LogisticRegression** | Modelo linear probabilístico (`class_weight='balanced'`) |
| **LinearSVC** | SVM linear, calibrado com `CalibratedClassifierCV` para gerar probabilidades |

## 🏆 Modelo final escolhido

**MultinomialNB** (`alpha=0.05`, `max_features=10000`) — obteve o melhor **F1-Score** no teste, além
das maiores **AUC-ROC** e **AUC-PR**.

## 📈 Métricas de avaliação e principais resultados

Resultados no **conjunto de teste** (776 mensagens, nunca vistas no treino):

| Modelo | Acurácia | Precisão | Recall | F1-Score | AUC-ROC | AUC-PR |
|--------|:--------:|:--------:|:------:|:--------:|:-------:|:------:|
| **MultinomialNB** ⭐ | **0,9871** | 0,9681 | **0,9286** | **0,9479** | **0,9943** | **0,9780** |
| LinearSVC | 0,9871 | 0,9783 | 0,9184 | 0,9474 | 0,9853 | 0,9643 |
| LogisticRegression | 0,9858 | 0,9780 | 0,9082 | 0,9418 | 0,9897 | 0,9699 |

**Matriz de confusão do modelo final (teste):** 675 verdadeiros negativos · 3 falsos positivos ·
7 falsos negativos · 91 verdadeiros positivos.

Palavras mais indicativas de **spam**: *free, win, prize, call, claim, txt, mobile, reply*.

---

## 📁 Estrutura dos arquivos

```
spam-sms-classifier/
│
├── app.py                          # Aplicação Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Este arquivo
│
├── notebooks/
│   └── notebook_atualizado.ipynb   # Notebook revisado da P1 (versão final)
│
├── model/
│   └── modelo_final.joblib         # Pipeline final salvo (TF-IDF + MultinomialNB)
│
├── reports/
│   └── relatorio_atualizado.pdf    # Relatório final atualizado
│
└── data/
    └── spam.csv                    # Dataset utilizado
```

## 🛠️ Tecnologias utilizadas

- **Python 3.10+**
- **scikit-learn** — vetorização, modelos, métricas e pipelines
- **pandas / numpy** — manipulação de dados
- **matplotlib / seaborn / wordcloud** — visualizações
- **joblib** — serialização do modelo
- **Streamlit** — aplicação web e deploy

---

## ▶️ Instruções para executar o notebook

```bash
# 1. Clonar o repositório
git clone https://github.com/<usuario>/spam-sms-classifier.git
cd spam-sms-classifier

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Abrir o notebook
jupyter notebook notebooks/notebook_atualizado.ipynb
```

> O notebook baixa o dataset automaticamente caso `spam.csv` não esteja presente.
> Ao final, ele regenera o arquivo `model/modelo_final.joblib`.

## ▶️ Instruções para executar o app Streamlit

```bash
# Localmente
pip install -r requirements.txt
streamlit run app.py
```

A aplicação abre no navegador (`http://localhost:8501`). Basta digitar uma mensagem e clicar em
**Classificar**.

## 🌐 Link do app publicado

👉 **`<preencher com o link do Streamlit Community Cloud>`**

---

## ⚠️ Limitações

- O dataset é majoritariamente em **inglês** e composto por SMS antigos; o desempenho pode cair em
  português ou em mensagens atuais (links encurtados, emojis, etc.).
- O modelo é **bag-of-words** (TF-IDF): ignora ordem das palavras e contexto semântico — não capta
  ironia nem paráfrases.
- Spam sofisticado, sem as palavras-gatilho típicas, ainda pode escapar (falsos negativos).

## ✅ Conclusão

O projeto entregou um classificador de spam **robusto e bem avaliado**: após o tratamento de
duplicatas e a otimização de hiperparâmetros, o **MultinomialNB** alcançou **F1-Score de 0,9479** e
**AUC-PR de 0,978** no conjunto de teste, com pouquíssimos erros. O modelo foi serializado e integrado
a uma aplicação Streamlit funcional, permitindo a classificação de novas mensagens em tempo real. As
métricas escolhidas (F1 e AUC-PR) são as adequadas ao desbalanceamento do problema, e a metodologia
(pipelines + validação cruzada estratificada) garante que os resultados sejam honestos e reproduzíveis.
