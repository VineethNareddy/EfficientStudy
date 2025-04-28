import streamlit as st
import pymupdf as pmf
import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from mongo import put_doc, mongo_conn
from db import insert_chatbot_messages, insert_user_messages, prompt_template

st.title("Study Helper")

def generate_llm_response_without_file(openai_api_key, question):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=st.session_state['retriever'])
    prompt = prompt_template(question, st.session_state["conversation_id"])
    return qa.run(prompt)
     

def generate_llm_response(file, openai_api_key, question):
	# Just for one response only, need to limit this when user has chat with doc
     if file is not None:
        content = file.read()
        currentDoc = pmf.open(stream=content, filetype="pdf")
        documents = [curPage.get_text() for curPage in currentDoc]
    
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
     chunks = text_splitter.create_documents(documents)
     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
     vectorDB = FAISS.from_documents(chunks, embeddings)
     retriever = vectorDB.as_retriever(search_kwargs={"k":1})
     if retriever not in st.session_state:
         st.session_state["retriever"] = retriever
     else:
         st.session_state["retriever"] = retriever
     
     qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
     prompt = prompt_template(question, st.session_state["conversation_id"])
     return qa.run(prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?", accept_file=True, file_type="pdf"):
    st.session_state.messages.append({"role": "user", "content": prompt.text})
    insert_user_messages(st.session_state["user_id"], st.session_state["conversation_id"], prompt.text)
    with st.chat_message("user"):
        st.markdown(prompt.text)

    with st.chat_message("assistant"):
        load_dotenv()
        if len(prompt.files) == 0:
            response = generate_llm_response_without_file(os.getenv("OPENAI_KEY"), prompt.text)
        else:
            response = generate_llm_response(prompt.files[0], os.getenv("OPENAI_KEY"), prompt.text)
        if len(prompt.files) != 0:
            db = mongo_conn()
            put_doc(prompt.files[0], db)
        insert_chatbot_messages(st.session_state["chatbot_id"], st.session_state["conversation_id"], response)
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})

# A lot of this is from streamlit documentation