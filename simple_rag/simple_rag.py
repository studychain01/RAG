import os  
import sys 
from dotenv import load_dotenv
from helper_functions import (EmbeddingProvider, replace_t_with_space, get_langchain_embedding_provider, show_context)
from evaluate_rag import evaluate_rag

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.makedirs('data', exist_ok=True)

path = "data/Understanding_Climate_Change.pdf"

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings/ 

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.
    
    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents 
    loader = PyPDFLoader(path)
    documents = loader.load()

    #split document into chunks 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

   
    #Create vector store 
    vectorstore = FAISS.from_documents(cleaned_texts, OpenAIEmbeddings())

    return vectorstore     


chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)

chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

test_query = "What do hyrodgen fuel cells do?"

context = chunks_query_retriever.get_relevant_documents(test_query)

for i, doc in enumerate(context):
    print(f"--- #{i+1} ---")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print(doc.page_content[:300], "...\n")
