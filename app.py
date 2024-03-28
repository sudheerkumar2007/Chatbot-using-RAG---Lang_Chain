import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.retrievers.multi_query import MultiQueryRetriever

import os


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("embeddings created")
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}) # starmpcc/Asclepius-Llama2-7B
    retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, #vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
  
  st.set_page_config(page_title = "Document chatbot", page_icon = ":books:")
  st.write(css, unsafe_allow_html=True)
  
  if "conversation" not in st.session_state:
    st.session_state.conversation = None
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

  st.image("bot_image.jpg",width=100, use_column_width=100, clamp=True, channels="RGB", output_format="auto")
  st.header("Hello!! Welcome to Doc-Bot")
  user_question=st.text_input("Ask a question on your document here:")
  if user_question:
        handle_userinput(user_question)

  with st.sidebar:
    st.subheader("Your document")
    pdf_docs = st.file_uploader("upload your document here",accept_multiple_files=True)
    
    if st.button("submit"):
        with st.spinner("processing"):
            
            #get pdf document
            raw_text = get_pdf_text(pdf_docs)
            #st.write(raw_text)
            
            #get the text chunks
            text_chunks = get_text_chunks(raw_text)
            st.write("embeddings created")
            #st.write(text_chunks)
            
            ## create vector store
            vectorstore = get_vectorstore(text_chunks)
            st.write("embeddings created")
            
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.write("Retriever is ready to retrieve the answer")
	
if __name__ == '__main__':
	main()