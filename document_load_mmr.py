from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from config import DOCUMENTS_PATH
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load documents from files
def load_documents(questions):
    documents = []
    for question in questions:
        loader = TextLoader(f"{DOCUMENTS_PATH}medmcqa_exp_{question['id']}.txt")
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents.extend(text_splitter.split_documents(pages))
    return documents

# Create embeddings and vector store
def create_vector_store(documents):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    print("Embeddings created")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print("Vector store created")
    return vectorstore

# MMR Relevance Scoring
def apply_mmr_relevance(retrieved_docs, query_embedding, lambda_param=0.5):
    doc_embeddings = np.array([doc.embedding for doc in retrieved_docs])
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    selected_docs = []
    while len(selected_docs) < len(retrieved_docs):
        remaining_docs = [i for i in range(len(retrieved_docs)) if i not in selected_docs]
        if not selected_docs:
            selected_docs.append(np.argmax(similarities))
        else:
            mmr_scores = []
            for idx in remaining_docs:
                min_sim = min([cosine_similarity([doc_embeddings[idx]], [doc_embeddings[sel]])[0][0] for sel in selected_docs])
                mmr_score = lambda_param * similarities[idx] - (1 - lambda_param) * min_sim
                mmr_scores.append(mmr_score)
            selected_docs.append(remaining_docs[np.argmax(mmr_scores)])
    
    return [retrieved_docs[i] for i in selected_docs]
