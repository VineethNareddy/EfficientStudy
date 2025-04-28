import streamlit as st
import pymupdf as pmf
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY", "")
if not OPENAI_KEY:
    st.error("OPENAI_KEY not found in .env file. Please add it and restart.")
    st.stop()


st.set_page_config(
    page_title="Efficient Study",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
  --sidebar-bg: #0B0E13;   /* darkest */
  --main-bg:    #161B22;   /* lighter than sidebar */
  --card-bg:    #1E242D;   /* file‑uploader & chat bubbles */
  --border:     #2C313C;
  --text:       #F5F7FA;
  --accent:     #B3A369;   /* GT gold */
  --muted:      #9CA3AF;
}

html, body, [class*='stApp'], .block-container {
    background-color:var(--main-bg);
    color:var(--text);
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"], section[data-testid="stSidebar"] * {
    background-color:var(--sidebar-bg) !important;
    color:var(--text) !important;
}

/* File‑uploader card */
div[data-testid="stFileUploader"] > div {
    background-color:var(--card-bg) !important;
    border:1px solid var(--border) !important;
}

/* Chat bubbles */
.chat-bubble {
    max-width: 80%;
    margin: .25rem 0;
    padding: .75rem 1rem;
    border-radius: 1rem;
    line-height: 1.4;
    font-size: 0.95rem;
    border:1px solid var(--border);
    background:var(--card-bg);
    color:var(--text);
}

.user-bubble {border-color:#444B57;}
.assistant-bubble {border-color:var(--border);}

/* Buttons */
.css-9ycgxx, .stButton>button {
    background-color:var(--accent);
    color:var(--sidebar-bg);
    border:none;
    border-radius:6px;
}

.stButton>button:hover {background-color:#C8B87A;}
button:focus {outline:none;}

/* Placeholder & secondary text */
span, p, label {color:var(--text);}  /* ensure forms stay bright */
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def generate_llm_response(file, question):
    """Return answer for question based on PDF content using OpenAI embeddings."""
    content = file.read()
    doc = pmf.open(stream=content, filetype="pdf")
    pages_text = [page.get_text() for page in doc]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = splitter.create_documents(pages_text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 1})

    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_KEY),
        chain_type="stuff",
        retriever=retriever,
    )
    return qa_chain.run(question)


with st.sidebar:
    st.image("gtlogo.png", use_container_width=True)
    pdf_file = st.file_uploader("", type="pdf")

    st.markdown("---")
    with st.expander("Chat History", expanded=False):
        if "messages" in st.session_state and st.session_state.messages:
            for idx, msg in enumerate(st.session_state.messages, 1):
                role = "You" if msg["role"] == "user" else "Assistant"
                st.markdown(f"**{idx}. {role}:** {msg['text']}")
        else:
            st.caption("No messages yet. Start asking questions!")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role, text = msg["role"], msg["text"]
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    st.markdown(f'<div class="chat-bubble {bubble_class}">{text}</div>', unsafe_allow_html=True)

prompt = st.chat_input("Ask about the uploaded slides…") if hasattr(st, "chat_input") else st.text_input("Question")

if prompt:
    if not pdf_file:
        st.warning("Please upload a PDF first!", icon="⚠️")
    else:
        st.session_state.messages.append({"role": "user", "text": prompt})
        st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking…"):
            answer = generate_llm_response(pdf_file, prompt)

        st.session_state.messages.append({"role": "assistant", "text": answer})
        st.markdown(f'<div class="chat-bubble assistant-bubble">{answer}</div>', unsafe_allow_html=True)


