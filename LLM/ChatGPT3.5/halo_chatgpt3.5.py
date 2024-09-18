import json
import os
import logging
from config import QUESTIONS_FILE, DOCUMENTS_PATH, OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION
from load_prompts import load_prompts  
from multi_query_generator import setup_llm, setup_multi_query_retriever, generate_multi_queries
from document_loader import load_documents, create_vector_store, apply_mmr_relevance
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

fewshot_prompt, cot_prompt = load_prompts('prompts/fewshot_prompt.txt', 'prompts/cot_prompt.txt')

with open(QUESTIONS_FILE, "r") as file:
    questions = json.load(file)

# Initialize LLM and MultiQueryRetriever
llm = setup_llm()
retriever = llm.as_retriever()
retriever_from_llm = setup_multi_query_retriever(retriever, llm)

documents = load_documents(questions, DOCUMENTS_PATH)
vectorstore = create_vector_store(documents)

# Combine the few-shot and CoT prompts into the full prompt
full_prompt = f"{fewshot_prompt}\n{cot_prompt}"
prompt = ChatPromptTemplate.from_messages([
    ("fewshot", fewshot_prompt),
    ("cot", cot_prompt)
])

# Create retrieval chain with MMR
retrieval_chain = create_retrieval_chain(vectorstore.as_retriever())

logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multi_queries = generate_multi_queries(questions, retriever_from_llm)

for question_number, queries in multi_queries.items():
    print(f"\nProcessing question {question_number}...")

    for i, query in enumerate(queries, start=1):
        print(f"\nQuery {i}: {query.page_content}")
        retrieved_docs = retrieval_chain.retrieve_documents(query.page_content)
        query_embedding = llm.embed(query.page_content)
        relevant_docs = apply_mmr_relevance(retrieved_docs, query_embedding)
        answers = []
        for doc in relevant_docs:
            response = llm.generate_response(prompt=prompt, context=doc.page_content)
            answer = response['answer']
            answers.append({"document": doc.page_content, "answer": answer})

        output_dir = f"batch_answers/question_{question_number}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"answers.json")
        with open(output_file, 'w') as outfile:
            json.dump(answers, outfile, indent=4)
        
        print(f"Saved answers for Question {question_number} in {output_file}")
