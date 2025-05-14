import boto3
import streamlit as st
import os
import uuid
from typing import List, Any

# AWS region env variable
os.environ["AWS_REGION"] = "us-east-2"

# initialize S3 client to use bucket
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vector_stores import FAISS

# initialize Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=os.getenv("AWS_REGION"))

# configure embeddings model
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client
)

def get_unique_id() -> str:
    """
    Generates a unique identifier for each document processing request.

    Returns:
        A randomly generated UUID4 string
    """
    return str(uuid.uuid4())

def split_text(pages: List[Any], chunk_size: int, chunk_overlap: int) -> List[Any]:
    """
    Splits document pages into smaller, overlapping chunks to improve processing.
    
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

