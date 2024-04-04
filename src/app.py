from dotenv import load_dotenv
import together
import os
import streamlit as st

# Functions
from utils.doc_process_funcs import extract_text, create_text_splitter, create_chunks, create_collection
from utils.output_gen_funcs import fetch_query_results, create_llm_prompt, generate_llm_output

# Constants
from config import CONTEXT_PROMPT, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY

# Credentials
load_dotenv("../.env")
together.api_key = os.environ["TOGETHER_API_KEY"]

# # Document processing: utils.doc_process_funcs.py
# text_blob = extract_text('../documents/linux-commands-handbook.pdf')
# text_splitter = create_text_splitter(200, 20)
# chunks = create_chunks(text_blob, text_splitter)
# collection = create_collection(chunks)

# # LLM output generation processing: utils.output_gen_funcs.py
# user_input = "What are some of the different Linux distros"
# results = fetch_query_results(user_input, collection)
# prompt = create_llm_prompt(results, CONTEXT_PROMPT, user_input) 
# output = generate_llm_output(prompt, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY)

# complete_output = output['output']['choices'][0]['text']
# print(complete_output)

def main():
    st.set_page_config(page_title = "Document Chatbot", page_icon = "📑")
    st.header("📑Upload a PDF and ask questions📑")
    

    if "collection" not in st.session_state:
        st.session_state.collection = None

    
    text_input = st.text_input("What do you want to know?")
    btn_ask = st.button("Ask")
    
    with st.sidebar:
        st.subheader("Upload PDF here")
        pdf_doc = st.file_uploader("Upload our PDF here and click on 'Process'")
        btn_process = st.button("Process")
        if btn_process:
            if st.session_state.collection == None:
                with st.spinner("Processing document..."):
                    # Document processing: utils.doc_process_funcs.py
                    text_blob = extract_text(pdf_doc)
                    text_splitter = create_text_splitter(200, 20)
                    chunks = create_chunks(text_blob, text_splitter)
                    st.session_state.collection = create_collection(chunks)
                    st.success("Document processed successfully")
            else:
                st.write("Document already processed!")

    if btn_ask:
        if st.session_state.collection == None:
            st.write("Please upload a document first!")
        else:
            with st.spinner("Searching document..."):
                # LLM output generation processing: utils.output_gen_funcs.py
                user_input = text_input
                results = fetch_query_results(user_input, st.session_state.collection)
                prompt = create_llm_prompt(results, CONTEXT_PROMPT, user_input) 
                output = generate_llm_output(prompt, MODEL, MAX_TOKENS, TEMPERATURE, TOP_K, TOP_P, REPETITION_PENALTY)
                complete_output = output['output']['choices'][0]['text']
                st.write(complete_output)


if __name__ == '__main__':
    main()