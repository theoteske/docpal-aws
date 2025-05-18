import boto3
import streamlit as st
import os
import uuid
from typing import List, Any

# AWS region env variable
os.environ["AWS_REGION"] = "us-east-2"

# Initialize S3 client to use bucket
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vector_stores import FAISS

# Initialize Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.getenv("AWS_REGION"))

# Configure embeddings model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

def get_unique_id() -> str:
    """
    Generate a unique identifier for each document processing request.

    Returns:
        A randomly generated UUID4 string
    """
    return str(uuid.uuid4())

def split_text(pages: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
    """
    Split document pages into smaller, overlapping chunks to improve processing.
    
    Args:
        pages: List of pages extracted from input document
        chunk_size: Maximum size of each text chunk
        chunk_overlap: Amount of overlap between adjacent chunks

    Returns:
        List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def create_vector_store(request_id: str, documents: List[Any]) -> str:
    """
    Create FAISS vector store for document chunks and upload to S3 bucket.

    Args:
        request_id: Unique identifier for this vector store
        documents: List of vector chunks to be vectorized

    Returns:
        Bool indicating whether vector store creation and upload was successful
    """
    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, bedrock_embeddings)

    # Save vector store locally
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vector_store.save_local(index_name=file_name, folder_path=folder_path)

    # Upload vector store to S3 bucket
    try:
        s3_client.upload_file(
            Filename=folder_path + "/" + file_name + ".faiss",
            Bucket=BUCKET_NAME,
            Key="my_faiss.faiss"
        )
        s3_client.upload_file(
            Filename=folder_path + "/" + file_name + ".pkl",
            Bucket=BUCKET_NAME,
            Key="my_faiss.pkl"
        )
        return True
    except Exception as e:
        st.error(f"Error uploading to S3 bucket: {str(e)}")
        return False
    
def main():
    """
    Main application function handling PDF upload, processing, and vector store creation.
    """
    # Set up the Streamlit application header
    st.title("PDF Vector Processor")
    st.subheader("Backend Interface for Chat with PDF Demo")

    # Create file uploader in the sidebar
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Process the uploaded file
    if uploaded_file is not None:
        # Create unique identifier for request
        request_id = get_unique_id()
        st.info(f"Processing request id: {request_id}")

        # Save document locally
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as f:
            f.write(uploaded_file.get_value())
        
        # Load and split the document into pages
        with st.spinner("Loading PDF document..."):
            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()
            st.success(f"Successfully loaded {len(pages)} pages from PDF")

        # Chunk pages of document
        with st.spinner("Splitting document into chunks..."):
            split_docs = split_text(pages, chunk_size=1000, chunk_overlap=200)
            st.success(f"Document split into {len(split_docs)} chunks")

        # Display sample chunks
        with st.expander("Preview document chunks"):
            st.subheader("Sample Chunk 1")
            st.write(split_docs[0])
            st.subheader("Sample Chunk 2")
            st.write(split_docs[1])

        # Create vector store
        with st.spinner("Creating vector embeddings and storing in FAISS..."):
            st.text("This may take a few moments depending on document size")
            result = create_vector_store(request_id, split_docs)
        
        # Display results
        if result:
            st.success("PDF processed successfully! Vector store created and uploaded to S3.")
            st.balloons()
        else:
            st.error("Error processing document. Please check application logs.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="PDF Vector Processor",
        page_icon="📚",
        layout="wide"
    )
    main()