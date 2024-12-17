import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import openai
import tiktoken
from docx import Document

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connect to Pinecone
def connect_to_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="gcp", region=PINECONE_ENVIRONMENT)
        )
    else:
        print(f"Using existing index: {PINECONE_INDEX_NAME}")
    print("Connected to Pinecone index.")
    return pc.Index(PINECONE_INDEX_NAME)

# Extract text from .docx
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

# Split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [encoding.decode(chunk) for chunk in chunks]

# Generate and store embeddings in Pinecone
def generate_and_store_embeddings(index, text_chunks):
    openai.api_key = OPENAI_API_KEY
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{len(text_chunks)}...")
        response = openai.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embedding = response.data[0].embedding  # Correct access to embedding
        index.upsert([{"id": f"chunk-{i}", "values": embedding, "metadata": {"text": chunk}}])
    print("All embeddings successfully stored in Pinecone!")

# Main execution
if __name__ == "__main__":
    file_path = "LLU Info for AI Agent.docx"

    # Step 1: Extract and split document text
    print("Extracting and splitting document text...")
    text = extract_text_from_docx(file_path)
    text_chunks = split_text_into_chunks(text)

    # Step 2: Connect to Pinecone
    index = connect_to_pinecone()

    # Step 3: Generate and store embeddings
    print("Generating and storing embeddings...")
    generate_and_store_embeddings(index, text_chunks)
