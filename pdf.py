import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import unicodedata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Google API key not found. Please check your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# Normalize and clean text to handle encoding issues
def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

# Extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text = page.extract_text() or ""
            text += normalize_text(raw_text)
    return text

# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Create and save a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Set up a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
    say, "answer is not available in the context." Do not provide incorrect answers.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user questions
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_path = os.path.join(os.getcwd(), "faiss_index", "index.faiss")
    
    if os.path.exists(faiss_path):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        st.write("Reply:", response["output_text"])
    else:
        st.error("Please upload and process the PDF files first.")

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="PDF Chat",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.header("Chat with PDF 📄")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and click Submit to Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete!")
                    else:
                        st.error("No text found in the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

# Run the app
if __name__ == "__main__":
    main()
