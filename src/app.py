from dotenv import load_dotenv
import together
import os
import chromadb

# Functions
from utils.doc_process_funcs import extract_text, create_text_splitter, create_chunks, create_collection

# Constants
from config import CONTEXT_PROMPT, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY

# Credentials
load_dotenv("../.env")
together.api_key = os.environ["TOGETHER_API_KEY"]

text_blob = extract_text('../documents/linux-commands-handbook.pdf')
text_splitter = create_text_splitter(200, 20)
chunks = create_chunks(text_blob, text_splitter)
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