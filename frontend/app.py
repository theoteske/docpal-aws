import boto3
import streamlit as st
import os
import uuid
import time
from typing import Dict, Any

# Customize page settings and appearance
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
            <style>
            .main-header {
                font-size: 2.5rem;
                color: #1E3A8A;
                margin-bottom: 1rem;
            }
            .subheader {
                font-size: 1.5rem;
                color: #3B82F6;
                margin-bottom: 2rem;
            }
            .stButton>button {
                background-color: #2563EB;
                color: white;
                border-radius: 6px;
                padding: 0.5rem 1rem;
                font-weight: bold;
            }
            .stTextInput>div>div>input {
                border-radius: 6px;
            }
            </style>
            """, unsafe_allow_html=True)

# AWS region env variable
os.environ["AWS_REGION"] = "us-east-2"

# Initialize S3 client to access bucket
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock as BedrockLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Initialize Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region=os.getenv("AWS_REGION"))

# Configure embeddings model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1", # TODO: consider changing to v2 if possible
    client=bedrock_client
)

# Define path for temporary storage
folder_path = "/tmp/"

def get_unique_id() -> str:
    """
    Generate a unique identifier for each document processing request.

    Returns:
        A randomly generated UUID4 string
    """
    return str(uuid.uuid4()) # TODO: read uuid documentation in python docs

def load_index() -> None:
    """
    Download the necessary FAISS index files (.pkl and .faiss) to reconstruct vector store locally.
    """
    try:
        with st.status("Downloading vector index from S3...") as status:
            # Download FAISS index file
            s3_client.download_file(
                Bucket=BUCKET_NAME,
                Key="my_faiss.faiss",
                Filename=f"{folder_path}my_faiss.faiss"
            )
            status.update(label="Downloaded FAISS index", state="running", expanded=True)

            # Download pickle file with metadata
            s3_client.download_file(
                Bucket=BUCKET_NAME,
                Key="my_faiss.pkl",
                Filename=f"{folder_path}my_faiss.pkl"
            )
            status.update(label="Vector index ready.", state="complete")
    except Exception as e:
        st.error(f"Error downloading index: {str(e)}")
        st.stop()

def get_llm() -> BedrockLLM:
    """
    Initialize and configure the Bedrock LLM.

    Returns:
        A configured BedrockLLM instance
    """
    llm = BedrockLLM(
        model_id="eu.meta.llama3-2-1b-instruct-v1:0", # TODO: check if this should be us instead of eu
        client=bedrock_client,
        model_kwargs={
            "temperature": 0.2,
            "maxTokens": 512, # note that Nova uses camelCase parameter names
            "topP": 0.9, # TODO: check if we need to delete this comma
            # add any other desired Nova-specific parameters here
        }
    )
    return llm

def get_response(llm: BedrockLLM, vectorstore: FAISS, question: str) -> str:
    """
    Generate contextual answer to user question using RAG pipeline.

    Args:
        llm: The language model to use for response generation
        vectorstore: The FAISS vector store containing document embeddings
        question: The text of the user's question

    Returns:
        The generated response to the user's question based on relevant context
    """
    # Define standard prompt template for the Nova model
    prompt_template = """
    Instructions: You are an expert document assistant. Please use only the information in the provided context 
    to answer the question accurately and concisely.
    
    If the context doesn't contain the information needed to answer the question, respond with:
    "I don't have enough information in the document to answer this question."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""

    # Create custom prompt with input variables
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Configure QA retrieval chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # combines all retrieved documents into one context
        retriever=vectorstore.as_retriever(
            search_type="similarity", # use similarity search
            search_kwargs={"k": 5} # perform KNN with k=5, so top 5 most relevant chunks
        ),
        return_source_documents=True, # include source documents in response
        chain_type_kwargs={"prompt": PROMPT} # use our custom prompt
    )

    # Invoke answer from QA chain
    answer = qa.invoke({"query": question})

    return answer["result"]

def main():
    """
    Main application function handling UI, index loading, and response generation.
    """
    # Display application headers with custom CSS
    st.markdown('<div class="main-header">DocPal: Chat with PDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="subheader">Powered by AWS Bedrock</div>', unsafe_allow_html=True)

    # Create sidebar
    with st.sidebar:
        st.image("https://placeholder.pics/svg/300x100/DEDEDE/555555/AWS%20Bedrock", width=300)
        st.subheader("About this app")
        st.info("""
                This application uses Retrieval-Augmented Generation (RAG) to chat with your documents.
                
                The system:
                1. Loads pre-indexed documents from S3.
                2. Matches your question with relevant document chunks.
                3. Uses Llama 3.2 1B Instruct to generate answers based on the context provided in the documents.
                """)
        st.subheader("Session Info")
        session_id = get_unique_id()
        st.info(f"Session ID: {session_id}")
    
    # Load index and construct vector store
    with st.spinner("Preparing document index..."):
        load_index()

        # Display files in temporary directory for debugging
        with st.expander("System debug info:", expanded=False):
            dir_list = os.listdir(folder_path)
            st.write(f"Files in {folder_path}:")
            st.code("\n".join(dir_list))

    # Load FAISS index with embeddings
    try:
        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embeddings,
            allow_dangerous_deserialization=True
        )
        st.success("Vector store sucessfully loaded.")
    except Exception as e:
        st.error(f"Failed to load index: {str(e)}")
        st.stop()

    # Accept user input
    st.subheader("Ask a question about your document:")
    question = st.text_input("Your question:", placeholder="For example, what are the main points discussed in the document?")

    col1, col2 = st.columns([1, 6])
    with col1:
        submit_button = st.button("⬆️")
    
    # Initialize chat history container
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Process question when button is clicked
    if submit_button and question:
        st.session_state.chat_history.append({"question": question, "answer": None})

        with st.spinner("Analyzing document(s) to generate answer..."):
            # Get LLM instance
            llm = get_llm()

            # Track response time
            start_time = time.time()

            # Get response from RAG pipeline
            response = get_response(llm, faiss_index, question)

            # Log response time
            response_time = time.time()-start_time

            # Update chat history with answer and response time
            st.session_state.chat_history[-1]["answer"] = response
            st.session_state.chat_history[-1]["time"] = f"{response_time:.2f}s"
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation")
        for idx, exchange in enumerate(st.session_state.chat_history):
            st.markdown(f"**Q{idx+1}: {exchange["question"]}**")
            if exchange["answer"]:
                st.markdown(f"{exchange["answer"]}")
                if "time" in exchange:
                    st.caption(f"Response time: {exchange["time"]}")
            st.divider()
    
    # Add footer
    st.markdown("---")
    st.caption("DocPal: Chat with PDF © 2025")

if __name__ == "__main__":
    main()