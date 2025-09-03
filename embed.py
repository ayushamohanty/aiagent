import utils as Utils
import os as OS
from tqdm import tqdm
import requests
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer

# Load a free local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def local_embedding_function(texts):
    """
    Takes a list of strings and returns their embeddings
    """
    return embedding_model.encode(texts).tolist()

def pdf_to_text(url):
    try:
        response = requests.get(url)
        pdf_data = response.content
        document = fitz.open(stream=pdf_data, filetype="pdf")
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def split_text_into_sections(text, min_chars_per_section):
    paragraphs = text.split('\n')
    sections = []
    current_section = ""
    current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)
        if current_length + paragraph_length + 2 <= min_chars_per_section:  # +2 for the double newline
            current_section += paragraph + '\n\n'
            current_length += paragraph_length + 2
        else:
            if current_section:
                sections.append(current_section.strip())
            current_section = paragraph + '\n\n'
            current_length = paragraph_length + 2

    if current_section:  # Add the last section
        sections.append(current_section.strip())

    return sections

def embed_text_in_chromadb(text, document_name, document_description, persist_directory=Utils.DB_FOLDER):
    # Split into chunks
    documents = split_text_into_sections(text, 1000)

    # Generate unique IDs for each document chunk
    ids = [str(hash(d)) for d in documents]

    # Metadata for the documents
    metadata = {
        "name": document_name,
        "description": document_description
    }
    metadatas = [metadata] * len(documents)

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = 'collection_1'

    # Create or get collection without OpenAI embeddings
    collection = client.get_or_create_collection(name=collection_name)

    # Current count
    count = collection.count()
    print(f"Collection already contains {count} documents")

    # Recalculate IDs to avoid duplicates
    ids = [str(i) for i in range(count, count + len(documents))]

    # Add in batches of 100
    for i in tqdm(range(0, len(documents), 100), desc="Adding documents", unit_scale=100):
        batch_docs = documents[i: i + 100]
        batch_metas = metadatas[i: i + 100]
        batch_ids = ids[i: i + 100]

        # Generate embeddings locally
        embeddings = local_embedding_function(batch_docs)

        # Add to collection
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings
        )

    new_count = collection.count()
    print(f"Added {new_count - count} documents")

if __name__ == "__main__":
    document_name = "Artificial Intelligence Act"
    document_description = "Artificial Intelligence Act"
    text = pdf_to_text(Utils.EUROPEAN_ACT_URL)
    embed_text_in_chromadb(text, document_name, document_description)
