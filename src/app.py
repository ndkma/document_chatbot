from dotenv import load_dotenv
import together
import os

# Functions
from utils.doc_process_funcs import extract_text, create_text_splitter, create_chunks, create_collection
from utils.output_gen_funcs import fetch_query_results, create_llm_prompt, generate_llm_output

# Constants
from config import CONTEXT_PROMPT, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY

# Credentials
load_dotenv("../.env")
together.api_key = os.environ["TOGETHER_API_KEY"]

# Document processing: utils.doc_process_funcs.py
text_blob = extract_text('../documents/linux-commands-handbook.pdf')
text_splitter = create_text_splitter(200, 20)
chunks = create_chunks(text_blob, text_splitter)
collection = create_collection(chunks)

# LLM output generation processing: utils.output_gen_funcs.py
user_input = "What are some of the different Linux distros"
results = fetch_query_results(user_input, collection)
prompt = create_llm_prompt(results, CONTEXT_PROMPT, user_input) 
output = generate_llm_output(prompt, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY)

complete_output = output['output']['choices'][0]['text']
print(complete_output)