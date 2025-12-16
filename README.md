<h1 align="center">RAG Level-2 Hybrid Retriever</h1>
<p align="center">Hybrid retrieval • Persistent Chroma • Deterministic Output</p>

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)

RAG Level-2: Hybrid Retriever (Similarity + MMR)

This repository contains a clean, modular Level-2 Retrieval-Augmented Generation (RAG) system implemented in Python.
It uses LangChain, ChromaDB, and OpenAI embeddings to build a realistic hybrid retrieval pipeline combining semantic similarity and Max Marginal Relevance (MMR).

The project demonstrates end-to-end RAG processing: ingestion, chunking, embedding, storage, retrieval, and deterministic LLM answering.

**Features**
  1. Document Ingestion
     
             Supports .txt and .pdf files
             Automatically detects file types
             Loads documents into a unified list for chunking

  2. Chunking
  
             Uses RecursiveCharacterTextSplitter with:
             Chunk size: 500
             Overlap: 100
             Newline-aware separators for cleaner split boundaries

  3. Embedding + Vector Storage
             
             OpenAI embeddings
             ChromaDB persistent storage under db_storage/
             Supports reloading the same database for future queries

  4. Hybrid Retrieval (Similarity + MMR)
  
             The custom retriever performs:
             Semantic similarity search (k=5)
             Max Marginal Relevance search (k=5)
             Merges both result sets
             Removes duplicates using hash(d.page_content)
             This produces more diverse, relevant, and non-repetitive context.

  5. Deterministic RAG Output
          
              The language model uses:
              temperature = 0 for deterministic responses
              Strict rules:
              Only answer using provided context
              If not found, respond exactly with:
              I don't know.

  8. Clean Project Structure
  
              ingest.py builds the vector database  
              app.py performs RAG queries
              .gitignore prevents .env and vector DB files from being tracked


**Project Structure**

        rag-level-2-hybrid-retriever/
        │
        ├── data/
        │   ├── AI_basics.txt
        │   ├── MF_basics.txt
        │   └── RAG_Overview_5_Page.pdf
        │
        ├── db_storage/
        │
        ├── app.py
        ├── ingest.py
        ├── requirements.txt
        ├── pyproject.toml
        ├── .python-version
        ├── .gitignore
        └── README.md

Installation

Clone the repository:

        git clone https://github.com/nishantranjan23/rag-level-2-hybrid-retriever.git
        cd rag-level-2-hybrid-retriever


Create a virtual environment:

        uv venv
        source .venv/bin/activate


Install dependencies:

        uv pip install -r requirements.txt


Create a .env file:

        OPENAI_API_KEY=your_api_key_here

Step 1: Build the Vector Database

Run:

        python ingest.py


This will:
Load PDF/TXT files
Chunk them
Create embeddings

Persist them in ChromaDB under db_storage/

Step 2: Run a RAG Query

        python app.py


Examples:

What are the steps in a RAG pipeline?
Explain embeddings in simple terms.
How Hybrid Retrieval Works
Similarity search returns closest semantic matches
MMR adds diversity and avoids redundancy
Both sets are merged
Duplicate chunks removed via:

        hash(d.page_content)


Unique chunks are passed to the LLM

This improves recall and context diversity.


Level-2 includes:

Chunking and embedding

Persistent vector storage

Hybrid retrieval

Deterministic LLM output

Modular RAG pipeline

This forms the foundation for Level-3 features such as metadata filtering, re-ranking, cross-encoders, and fusion retrieval.

