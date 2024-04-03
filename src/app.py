from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import chromadb
import together
import os

# Constants
from config import CONTEXT_PROMPT, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY

# Credentials
load_dotenv("../.env")
together.api_key = os.environ["TOGETHER_API_KEY"]

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

text_blob = extract_text('../documents/linux-commands-handbook.pdf')
text_blob

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

text_splitter = create_text_splitter(200, 20)

# Use text splitter on document to create list of chunks
def create_chunks(text : str) -> list:
    chunks_list = text_splitter.split_text(text)
    nr_of_chunks = len(chunks_list)
    print(f"Nr. of text chunks: {nr_of_chunks}")
    
    return chunks_list

chunks = create_chunks(text_blob)

# Create collection by feeding chunks into chromadb
def create_collection(doc_chunks : list) -> chromadb.Collection:
    chroma_client = chromadb.Client()
    generated_collection = chroma_client.create_collection(name = 'my_collection')
    generated_collection.add(
        documents = doc_chunks,
        ids=[f"id{n}" for n in range(len(doc_chunks))]
    )

    return generated_collection

collection = create_collection(chunks)

# Fetch preliminary query results from the chromadb collection
def fetch_query_results(question : str, collection : chromadb.Collection) -> dict:
    results_dict = collection.query(
        query_texts = question,
        n_results = 3
    )
    print(results_dict)

    return results_dict

user_input = "What are some of the different Linux distros"
results = fetch_query_results(user_input, collection)

# for k, v in results.items():
#     print(k, v)
# print(results['documents'][0])

# Provide specific information for the LLM to base the comment off of

def create_llm_prompt(query_results, llm_context, question):
    info_page = query_results['documents'][0]
    specific_prompt = [
        f"Information page is {query_results['documents'][0]}."
        f"Question is {question}"
    ]
    llm_prompt = f"<s>[INST] <<SYS>>{llm_context}<</SYS>>\\n\\n"
    for specifics in specific_prompt:
        llm_prompt += f"[INST]{specifics}[/INST]"
    return llm_prompt
 
prompt = create_llm_prompt(results, CONTEXT_PROMPT, user_input) 

# Generates the comment with the prompt and llm parameters from config.py
def generate_llm_output(prompt, model, max_tokens, temperature, top_k, top_p, repetition_penalty):
    llm_output = together.Complete.create(
        prompt,
        model = model,
        max_tokens = max_tokens,
        temperature = temperature,
        top_k = top_k,
        top_p = top_p,
        repetition_penalty = repetition_penalty,
        stop = ['</s>']
    )
    return llm_output

output = generate_llm_output(prompt, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY)
complete_output = output['output']['choices'][0]['text']
print(complete_output)