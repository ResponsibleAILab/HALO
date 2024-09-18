from Bio import Entrez
import json
import os
from config import OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION
from multi_query_generator import setup_llm, setup_multi_query_retriever, generate_multi_queries
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

Entrez.email = ""

# Load questions from JSON file
with open("c:/Users/asume/Downloads/SummuProject/SummuProject/dev2.json", "r") as file:
    questions = json.load(file)

llm = setup_llm()
retriever = llm.as_retriever()
retriever_from_llm = setup_multi_query_retriever(retriever, llm)

# Generate multiple queries for all questions
multi_queries = generate_multi_queries(questions, retriever_from_llm)

# Retrieve documents from PubMed for each query
def retrieve_documents_from_pubmed(queries):
    all_documents = {}
    for question_number, queries in multi_queries.items():
        query_texts = [query.page_content for query in queries]
        all_documents[question_number] = []
        for query_text in query_texts:
            handle = Entrez.esearch(db="pubmed", term=query_text, retmax=10) 
            record = Entrez.read(handle)
            handle.close()
            
            pmids = record["IdList"]
            documents = []
            for pmid in pmids:
                handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
                document = handle.read()
                handle.close()
                documents.append(document)
            
            all_documents[question_number].extend(documents)
    
    return all_documents

retrieved_documents = retrieve_documents_from_pubmed(multi_queries)

output_file = "pubmed_documents.json"
with open(output_file, 'w') as outfile:
    json.dump(retrieved_documents, outfile, indent=4)

print(f"Retrieved documents for {len(questions)} queries and saved to {output_file}.")
