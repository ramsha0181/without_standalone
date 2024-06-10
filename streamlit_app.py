import streamlit as st
from langchain_community.vectorstores import Qdrant
from langchain_community.chat_models import ChatOpenAI
from utils import *
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
import openai
import os
from langchain_community.document_loaders import  Docx2txtLoader,PyMuPDFLoader, TextLoader, CSVLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from pathlib import Path
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.prompts.prompt import PromptTemplate
from operator import itemgetter

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv('qdrant_url')
qdrant_api_key = os.getenv('qdrant_api_key')

embed_fn = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

st.title('Document Retrieval')
file = st.file_uploader("Upload a file", type=["pdf","csv","txt","docs"], key="my_file_uploader")


def handle_upload(file):
    try:
        contents = file.read()
        file_extension = file.name[-4:].lower()
        original_filename = file.name
        if 'session_data' not in st.session_state:
            st.session_state['session_data'] = []
        session_data = st.session_state['session_data']
        doc_list = handling_files(contents, file_extension, original_filename)
        cl_name = background_task(doc_list, embed_fn, original_filename)
        vectordb = load_local_vectordb_using_qdrant(cl_name, embed_fn)
        if session_data:
            for item in session_data:
                if item["query"]:
                    st.write(f"Query: {item['query']}")
                    st.write(f"Response: {item['response']}")
                st.write("---")
        query = st.text_input("Ask Anything")
        if query:
            response = conversation_retrieval_chain(query, vectordb)
            st.write("Response:", response or "No relevant information found.")
            session_data.append({
                "query": query,
                "response": response if query else None
            })

    except Exception as e:
        st.error("Error: {}".format(e))

        
def main():
    if file is not None:
        vectordb = handle_upload(file)

if __name__ == "__main__":
    main()
