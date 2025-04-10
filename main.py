import streamlit as st
import pymupdf as pmf
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv



def generate_llm_response(file, openai_api_key, question):

	#Just for one response only, need to limit this when user has chat with doc
	if file is not None:
		content = file.read()
		currentDoc = pmf.open(stream=content, filetype="pdf")
		documents = [curPage.get_text() for curPage in currentDoc]


	# Used to get top-K documents (K=1 in this case)
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
	chunks = text_splitter.create_documents(documents)
	embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
	vectorDB = FAISS.from_documents(chunks, embeddings)
	retriever = vectorDB.as_retriever(search_kwargs={"k":1})

	qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
	return qa.run(question)

file = st.file_uploader('Upload slides', type='pdf')
question = st.text_input('Question:', placeholder = 'Please enter your question here')

#make db eventually
result = []
with st.form('myform', clear_on_submit=True):
	load_dotenv()
	openai_api_key = os.getenv("OPENAI_KEY")
	submitted = st.form_submit_button('Submit', disabled=not(file and question))
	if submitted:
		with st.spinner('Generating Answer'):
			response = generate_llm_response(file, openai_api_key, question)
			result.append(response)

if len(result):
	st.info(response)
 
