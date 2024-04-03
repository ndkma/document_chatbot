from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
import chromadb

# Use reader to turn pdf into text
def extract_text(doc_path : str) -> str:
    reader = PdfReader(doc_path)
    nr_of_pages = len(reader.pages)
    text_string = ''
    for page in reader.pages:
        text_string += page.extract_text()
    nr_of_characters = len(text_string)

    print(f"Nr. of pages in document: {nr_of_pages}")
    print(f"Nr. of characters in document: {nr_of_characters}")
    
    return text_string

# Specify text spliter specifications
def create_text_splitter(chunk_size : int, chunk_overlap : int) -> CharacterTextSplitter:
    t_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        is_separator_regex = False
    )
    return t_splitter

# Use text splitter on document to create list of chunks
def create_chunks(text : str, splitter: CharacterTextSplitter) -> list:
    chunks_list = splitter.split_text(text)
    nr_of_chunks = len(chunks_list)
    print(f"Nr. of text chunks: {nr_of_chunks}")
    
    return chunks_list

# Create collection by feeding chunks into chromadb
def create_collection(doc_chunks : list) -> chromadb.Collection:
    chroma_client = chromadb.Client()
    generated_collection = chroma_client.create_collection(name = 'my_collection')
    generated_collection.add(
        documents = doc_chunks,
        ids=[f"id{n}" for n in range(len(doc_chunks))]
    )

    return generated_collection