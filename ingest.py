from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma

import os
from dotenv import load_dotenv

load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")

folder_path = "/Users/fisdom/Desktop/RAG/RAG Level2/data"
all_docs = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isdir(file_path):
        continue
    
    ext = filename.lower()

    if filename.endswith(".txt"):
        loader = TextLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded TXT: {filename} → {len(docs)} document(s)")

    if filename.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)
        print(f"Loaded PDF: {filename} → {len(docs)} document(s)")


textsplitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n"],
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = textsplitter.split_documents(all_docs)

PERSIST_DIRECTORY = "../RAG Level2/db_storage"

embeddings = OpenAIEmbeddings(openai_api_key = openai_key)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,   
)


print("\nVector database created + saved.")