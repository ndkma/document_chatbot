# Provide background context for the LLM for generating responses
CONTEXT_PROMPT = (
    "You are a bot providing answers to queries based only out of the supplied documents. "
    "You will be given information pages. If you can answer the question by using these pages, do so. "
    "Do not give answers to questions that can't be answered using the information pages. "
    )

MODEL = "togethercomputer/llama-2-13b-chat"       # Chat model to be used
MAX_TOKENS = 250                                  # Hard limit on length of comment
TEMPERATURE = 0.5                                 # Measure of comment creativity
TOP_K = 90                                        # Measure of comment diversity
TOP_P = 0.8
REPETITION_PENALTY = 1.1