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
#from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_experimental.text_splitter import SemanticChunker
#import docx2txt
#from langgraph.graph import StateGraph,END



def classify(question):
    greeting_type = ''
    if question:
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
        q_class = llm("classify intent of given input as greeting or not_greeting. Output just the class.Input:{}".format(question)).strip()
        if q_class == 'greeting':
            greeting_type = greeting_classify(question)
            #st.write("greeting type is ", greeting_type)
            return greeting_type
        else:
            greeting_type = 'not greeting'
            #st.write("greeting type is ", greeting_type)
            return greeting_type

def greeting_classify(question):
    welcome_type = ['hi','hello','good morning', 'good day']
    if question.lower() in welcome_type:
        #st.write("returned welcome_type")
        return 'welcome_type_greeting'
    else:
        #st.write("returned bye_type")
        return 'bye_type_greeting'

def classify_input_node(state):
    '''class GraphState(TypedDict):
question: Optional[str] = None
response: Optional[str] = None
greeting: Optional[str] = None
classification: Optional[str] = None

workflow = StateGraph(GraphState)'''    
    question = state.get('question', '').strip()
    classification = classify(question) 
    return {"classification": classification}

def handle_greeting_node(state):
    return {"greeting": "Hello! How can I help you today?"}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def read_text_from_document(docs):
    text = ""
    for doc in docs:
        file_extension = doc.split('.')[-1].lower()
        if file_extension == 'pdf':
            return read_text_from_pdf(file_path)
        elif file_extension == 'txt':
            return read_text_from_txt(file_path)
        elif file_extension == 'csv':
            return read_text_from_csv(file_path)
        elif file_extension == 'docx':
            return read_text_from_docx(file_path)
        elif file_extension == 'xlsx':
            return read_text_from_xlsx(file_path)
        else:
            raise ValueError("Unsupported file format")


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_semantic_chunks(text):
    # Percentile - all differences between sentences are calculated, and then any difference greater than the X percentile is split
    embedding = SemanticChunker(HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl"))
    text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="percentile") # "standard_deviation", "interquartile"
    chunks = text_splitter.create_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    print("embeddings created")
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512}) # starmpcc/Asclepius-Llama2-7B
    #retriever = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)
    retriever=vectorstore.as_retriever()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever, #vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    question_class = classify(user_question)
    if question_class == 'welcome_type_greeting':
        st.write(user_question,"!! How can i help you")
    elif question_class == 'bye_type_greeting':
        st.write("Thank you! Have a great rest of the day")
#    else:
#        if doc is None:
#            st.write("Please upload your documents in the side bar and start asking questions ")
    else:
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
  if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = None

  st.image("bot_image.jpg",width=100, use_column_width=100, clamp=True, channels="RGB", output_format="auto")
  st.header("Hello!! Welcome to Doc-Bot")
  user_question=st.text_input("Ask a question on your document here:")
  

  with st.sidebar:
    st.subheader("Your document")
    pdf_docs = st.file_uploader("upload your document here",accept_multiple_files=True)
    if pdf_docs is None:
        st.write("Please upload a document Here and start asking questions")
    
    if st.button("submit"):
        with st.spinner("processing"):
            
            #get pdf document
            raw_text = get_pdf_text(pdf_docs)
            #st.write(raw_text)
            
            #get the text chunks
            #text_chunks = get_semantic_chunks(raw_text) 
            text_chunks = get_text_chunks(raw_text)
            st.write("Chunks created")
            #st.write(text_chunks)
            
            ## create vector store
            vectorstore = get_vectorstore(text_chunks)
            st.write("embeddings created")
            doc_ready = True
            
            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)
            st.write("Retriever is ready to retrieve the answer")
	
  if user_question:
    #st.write("pdf docs is: ", pdf_docs)
    #if st.session_state.pdf_docs is None:
    #    st.write("Please upload a document in the side bar and start asking questions")
    #else:
    handle_userinput(user_question)
if __name__ == '__main__':
	main()
