import os
import tiktoken
from docx import Document

# Step 1: Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Step 2: Extract text from .docx
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return "\n".join(full_text)

# Step 3: Split text into chunks (~500 tokens each)
def split_text_into_chunks(text, chunk_size=500):
    encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = encoding.encode(text)
    
    # Break tokens into chunks
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    # Decode chunks back to strings
    text_chunks = [encoding.decode(chunk) for chunk in chunks]
    return text_chunks

if __name__ == "__main__":
    # Path to your .docx file
    file_path = "LLU Info for AI Agent.docx"
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        exit(1)
    
    # Extract text
    print("Extracting text from document...")
    text = extract_text_from_docx(file_path)
    
    # Split text into chunks
    print("Splitting text into chunks...")
    chunks = split_text_into_chunks(text)
    
    # Print summary
    print(f"Total Chunks Created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks for verification
        print(f"\n--- Chunk {i+1} ---\n{chunk}\n")
    
    print("Document processing complete!")
