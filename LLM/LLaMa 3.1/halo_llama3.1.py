import json
import os
import logging
from config import QUESTIONS_FILE, DOCUMENTS_PATH
from multi_query_generator import setup_multi_query_retriever, generate_multi_queries
from document_loader import load_documents, create_vector_store, apply_mmr_relevance
import ollama
from load_prompts import load_prompts

fewshot_prompt, cot_prompt = load_prompts('prompts/fewshot_prompt.txt', 'prompts/cot_prompt.txt')

with open(QUESTIONS_FILE, "r") as file:
    questions = json.load(file)

# Initialize the retriever using the Ollama model
retriever = ollama.Retriever(model="llama3.1")

# Generate multiple queries for all questions
multi_queries = generate_multi_queries(questions, retriever)
documents = load_documents(questions, DOCUMENTS_PATH)
vectorstore = create_vector_store(documents)

# Create retrieval chain with MMR (Maximal Marginal Relevance)
retrieval_chain = create_retrieval_chain(vectorstore.as_retriever())

logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

for question_number, queries in multi_queries.items():
    print(f"\nProcessing question {question_number}...")

    for i, query in enumerate(queries, start=1):
        print(f"\nQuery {i}: {query.page_content}")
        retrieved_docs = retrieval_chain.retrieve_documents(query.page_content)
        query_embedding = ollama.embed(query.page_content)
        relevant_docs = apply_mmr_relevance(retrieved_docs, query_embedding)
        answers = []
        for doc in relevant_docs:
            response = ollama.chat(
                model="llama3.1",
                messages=[
                    {"role": "system", "content": fewshot_prompt},
                    {"role": "system", "content": cot_prompt},
                    {"role": "user", "content": query.page_content},  
                    {"role": "system", "content": doc.page_content}  
                ]
            )
            answer = response["message"]["content"]
            answers.append({"document": doc.page_content, "answer": answer})

        output_dir = f"batch_answers/question_{question_number}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"answers.json")

        with open(output_file, 'w') as outfile:
            json.dump(answers, outfile, indent=4)
        
        print(f"Saved answers for Question {question_number} in {output_file}")
