import pdfplumber
from cohere import Client
import pinecone
import numpy as np
from typing import List
import io
import logging

logger = logging.getLogger(__name__)

def parse_and_chunk_resume(pdf_content: bytes, cohere_api_key: str, pinecone_api_key: str) -> List[str]:
    logger.info("Starting resume parsing")
    # Create a file-like object from bytes
    pdf_file = io.BytesIO(pdf_content)
    logger.info("PDF file created")
    # Parse PDF
    with pdfplumber.open(pdf_file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
    logger.info("PDF text extracted")
    # Chunk text (simple split by 500 chars)
    chunk_size = 500
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"Text chunked into {len(chunks)} pieces")

    # Initialize Cohere for embeddings with input_type
    co = Client(cohere_api_key)
    embeddings = co.embed(texts=chunks, model="embed-english-v3.0", input_type="search_document").embeddings
    logger.info("Cohere embeddings generated")

    # Initialize Pinecone
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    index_name = "asha-resume-chunks"
    index = None
    if index_name not in pc.list_indexes():
        spec = {
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
        pc.create_index(index_name, spec=spec, dimension=1024)
        logger.info(f"Created Pinecone index: {index_name}")
    else:
        index = pc.Index(index_name)
        index_desc = pc.describe_index(index_name)
        if index_desc.dimension != 1024:
            raise ValueError(f"Existing index dimension ({index_desc.dimension}) does not match expected 1024. Please delete the 'asha-resume-chunks' index in the Pinecone dashboard and retry.")
        logger.info(f"Using existing Pinecone index: {index_name} with dimension {index_desc.dimension}")
    if index is None:
        index = pc.Index(index_name)

    # Store chunks with embeddings
    vectors = [(str(i), embedding, {"text": chunk}) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
    index.upsert(vectors=vectors)
    logger.info(f"Upserted {len(vectors)} vectors to Pinecone")

    return chunks
