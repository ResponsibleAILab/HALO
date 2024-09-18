import json
import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import AzureChatOpenAI
from config import QUESTIONS_FILE, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION

def setup_llm():
    llm = AzureChatOpenAI(
        deployment_name="sumera-model",
        temperature=0.7,
        api_key=OPENAI_API_KEY,
        model_name="gpt-35-turbo-16k",
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_API_BASE
    )
    return llm

def setup_multi_query_retriever(retriever, llm):
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever, llm=llm)
    return retriever_from_llm

# Generate multiple queries for all questions in the dataset
def generate_multi_queries():
    with open(QUESTIONS_FILE, "r") as file:
        questions = json.load(file)

    llm = setup_llm()
    retriever = llm.as_retriever()
    retriever_from_llm = setup_multi_query_retriever(retriever, llm=llm)

    multi_queries = {}
    for question_number, question in enumerate(questions, start=1):
        input_text = f"Choose the correct option from the given question:\n{question['question']}"
        queries = retriever_from_llm.get_relevant_documents(query=input_text)
        multi_queries[question_number] = queries
        print(f"Generated {len(queries)} queries for question {question_number}.")

    return multi_queries

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
