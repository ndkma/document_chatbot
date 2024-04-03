import chromadb
import together

# Fetch preliminary query results from the chromadb collection
def fetch_query_results(question : str, collection : chromadb.Collection) -> dict:
    results_dict = collection.query(
        query_texts = question,
        n_results = 3
    )
    print(results_dict)

    return results_dict

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