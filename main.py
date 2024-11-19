import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    model_id = "llama3.1"
    emb = OllamaEmbeddings(model=model_id)
    vector_store = FAISS.from_texts(text_chunks, embedding=emb)
    vector_store.save_local("faiss_index")
    # Make sure to check CWD as faiss_index gets stored in CWD

# Function to get the QA conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, respond with "answer is not available in the context". Do not provide incorrect answers.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGroq(temperature=0.3, groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate a response
def user_input(user_question):
    model_id = "llama3.1"
    emb = OllamaEmbeddings(model=model_id)
    new_db = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# Main function for Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chatbot for PDF Files")

    user_question = st.text_input("Ask a question about the uploaded PDF files")
    
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on 'Submit & Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing completed!")

if __name__ == "__main__":
    main()
