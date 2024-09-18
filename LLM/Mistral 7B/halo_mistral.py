import json
import os
import logging
from config import QUESTIONS_FILE, DOCUMENTS_PATH
from multi_query_generator import setup_mistral_llm, setup_multi_query_retriever, generate_multi_queries
from document_loader import load_documents, create_vector_store, apply_mmr_relevance
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from load_prompts import load_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Mistral LLM and Tokenizer
model_name = "mistral-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load few-shot and CoT prompts
fewshot_prompt, cot_prompt = load_prompts('prompts/fewshot_prompt.txt', 'prompts/cot_prompt.txt')

# Load questions from JSON file
with open(QUESTIONS_FILE, "r") as file:
    questions = json.load(file)

# Initialize Mistral LLM and MultiQueryRetriever
llm = setup_mistral_llm(model, tokenizer)
retriever = llm.as_retriever()
retriever_from_llm = setup_multi_query_retriever(retriever, llm)

documents = load_documents(questions, DOCUMENTS_PATH)
vectorstore = create_vector_store(documents)

full_prompt = f"{fewshot_prompt}\n{cot_prompt}"

retrieval_chain = create_retrieval_chain(vectorstore.as_retriever())

logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multi_queries = generate_multi_queries(questions, retriever_from_llm)

def generate_response_mistral(model, tokenizer, prompt, context):
    input_text = f"{prompt}\n{context}"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

for question_number, queries in multi_queries.items():
    print(f"\nProcessing question {question_number}...")

    for i, query in enumerate(queries, start=1):
        print(f"\nQuery {i}: {query.page_content}")

        retrieved_docs = retrieval_chain.retrieve_documents(query.page_content)
        query_embedding = llm.embed(query.page_content)  # Ensure embedding works with Mistral
        relevant_docs = apply_mmr_relevance(retrieved_docs, query_embedding)

        
        answers = []
        for doc in relevant_docs:
            response = generate_response_mistral(model, tokenizer, prompt=full_prompt, context=doc.page_content)
            answer = response  
            answers.append({"document": doc.page_content, "answer": answer})
    
        output_dir = f"batch_answers/question_{question_number}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"answers.json")
        with open(output_file, 'w') as outfile:
            json.dump(answers, outfile, indent=4)
        
        print(f"Saved answers for Question {question_number} in {output_file}")
