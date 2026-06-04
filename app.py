"""
Aplicação Streamlit — Detecção de Spam em SMS
Projeto Avaliativo P2 · Grupo 5 · A.I - Classificadores (UNIMAR 2026/1)

O app carrega o Pipeline final salvo (TF-IDF + MultinomialNB), recebe uma mensagem
digitada pelo usuário, aplica a MESMA limpeza usada no treino e exibe a predição
(HAM ou SPAM) com a probabilidade associada.
"""

import re
import os
import joblib
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Detector de Spam SMS", page_icon="📩", layout="centered")


# ──────────────────────────────────────────────────────────────────────────────
# Limpeza de texto — DEVE ser idêntica à do notebook (mesmo pré-processamento)
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(t: str) -> str:
    """Minúsculas, remove pontuação/números e normaliza espaços."""
    t = t.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Carregamento do modelo (cache: carrega só uma vez, não a cada interação)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Procura o modelo salvo em caminhos comuns e o carrega com joblib."""
    candidatos = [
        os.path.join("model", "modelo_final.joblib"),
        "modelo_final.joblib",
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            return joblib.load(caminho)
    return None


modelo = load_model()

# ──────────────────────────────────────────────────────────────────────────────
# Banco de exemplos rotativos (5 spam + 5 ham)
# Cada clique avança para o próximo; ao chegar no fim, volta ao início.
# ──────────────────────────────────────────────────────────────────────────────
EXEMPLOS_SPAM = [
    "Congratulations! You won a FREE prize. Call now to claim your reward!!!",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry.",
    "URGENT! Your mobile number has been awarded a £2,000 Bonus Caller Prize. Call 09066660100 now. T&Cs at www.dbuk.net.",
    "You have 1 new message. Call 0207-083-6089 now to receive your FREE ringtone and UNLIMITED texts!",
]

EXEMPLOS_HAM = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you pick up some milk on your way home? Thanks!",
    "I'll be a bit late to the meeting, traffic is terrible today.",
    "Happy birthday! Hope you have a wonderful day.",
    "Did you finish the report? The boss is asking about it.",
]

# ──────────────────────────────────────────────────────────────────────────────
# Inicialização do session_state
# ──────────────────────────────────────────────────────────────────────────────
if "msg" not in st.session_state:
    st.session_state.msg = ""
if "idx_spam" not in st.session_state:
    st.session_state.idx_spam = 0
if "idx_ham" not in st.session_state:
    st.session_state.idx_ham = 0

# ──────────────────────────────────────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────────────────────────────────────
st.title("📩 Detector de Spam em Mensagens SMS")
st.caption("Grupo 5 · A.I - Classificadores · UNIMAR 2026/1 — Modelo: TF-IDF + MultinomialNB")

st.write(
    "Digite uma mensagem SMS (em inglês) e o modelo classificará como "
    "**HAM** (legítima) ou **SPAM** (indesejada)."
)

if modelo is None:
    st.error(
        "Modelo não encontrado. Coloque o arquivo `modelo_final.joblib` na pasta "
        "`model/` (ou na raiz do projeto) antes de executar."
    )
    st.stop()

# Botões de exemplos rotativos
col_a, col_b = st.columns(2)

if col_a.button("🚫 Exemplo de SPAM", use_container_width=True):
    st.session_state.msg = EXEMPLOS_SPAM[st.session_state.idx_spam]
    st.session_state.idx_spam = (st.session_state.idx_spam + 1) % len(EXEMPLOS_SPAM)

if col_b.button("✅ Exemplo de HAM", use_container_width=True):
    st.session_state.msg = EXEMPLOS_HAM[st.session_state.idx_ham]
    st.session_state.idx_ham = (st.session_state.idx_ham + 1) % len(EXEMPLOS_HAM)

# Indicador de qual exemplo está sendo exibido (1-indexado para o usuário)
spam_atual = st.session_state.idx_spam  # já foi incrementado, aponta para o próximo
ham_atual  = st.session_state.idx_ham
if st.session_state.msg:
    col_a.caption(f"Exemplo {spam_atual if spam_atual != 0 else len(EXEMPLOS_SPAM)}/{ len(EXEMPLOS_SPAM)}")
    col_b.caption(f"Exemplo {ham_atual  if ham_atual  != 0 else len(EXEMPLOS_HAM) }/{len(EXEMPLOS_HAM)}")

# Campo de entrada
mensagem = st.text_area(
    "Mensagem:",
    value=st.session_state.msg,
    height=120,
    placeholder="Ex.: WINNER! You have been selected to receive a free gift...",
)

# Botão de predição
if st.button("🔎 Classificar", type="primary", use_container_width=True):
    if not mensagem.strip():
        st.warning("Digite uma mensagem primeiro.")
    else:
        # 1) limpa a entrada (mesmo pré-processamento do treino)
        texto_limpo = clean_text(mensagem)
        # 2) executa a predição com o pipeline (texto -> TF-IDF -> modelo)
        pred      = modelo.predict([texto_limpo])[0]           # 0=ham, 1=spam
        prob_spam = float(modelo.predict_proba([texto_limpo])[0, 1])

        # 3) exibe resultado de forma clara
        st.divider()
        if pred == 1:
            st.error(f"### 🚫 SPAM\nProbabilidade de spam: **{prob_spam:.1%}**")
            st.write("Interpretação: a mensagem tem características típicas de spam "
                     "(ofertas, prêmios, chamadas para ação). Recomenda-se cautela.")
        else:
            st.success(f"### ✅ HAM (legítima)\nProbabilidade de spam: **{prob_spam:.1%}**")
            st.write("Interpretação: a mensagem parece uma comunicação legítima e cotidiana.")

        # Barra de confiança
        st.progress(prob_spam, text=f"Confiança de que é spam: {prob_spam:.1%}")

st.divider()
with st.expander("ℹ️ Como o app funciona"):
    st.markdown(
        "1. O texto digitado passa pela função `clean_text` (a mesma do treino).\n"
        "2. O `Pipeline` salvo aplica o **TF-IDF** e o classificador **MultinomialNB**.\n"
        "3. `predict` devolve a classe (0=ham, 1=spam) e `predict_proba` a probabilidade.\n\n"
        "**Limitações:** o modelo foi treinado com SMS em inglês; mensagens em outros "
        "idiomas ou muito diferentes do dataset podem ter desempenho inferior."
    )
