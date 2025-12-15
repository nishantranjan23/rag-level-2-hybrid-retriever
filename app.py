###############################################################################
# IMPORTS
###############################################################################
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

import os
from dotenv import load_dotenv



###############################################################################
# 1. LOAD ENVIRONMENT VARIABLES
###############################################################################
load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")



###############################################################################
# 2. LOAD EMBEDDING MODEL
###############################################################################
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)



###############################################################################
# 3. LOAD EXISTING CHROMA VECTORSTORE
###############################################################################
vectorstore = Chroma(
    persist_directory="../RAG Level2/db_storage",
    embedding_function=embeddings
)



###############################################################################
# 4. HYBRID RETRIEVER (Similarity + MMR + De-duplication)
###############################################################################
def hybrid_retriever(query: str):
    """
    Performs hybrid retrieval:
    - Similarity search (semantic relevance)
    - MMR search (diversity + relevance)
    - Merge results
    - Remove duplicates using full-content hashing
    """
    print("\n--- Starting Hybrid Retrieval Search ---")

    # 4A. Similarity (semantic) search
    vector_docs = vectorstore.similarity_search(query, k=5)
    print(f"Similarity Search returned: {len(vector_docs)} chunks")

    for i, doc in enumerate(vector_docs):
        doc.metadata["retrieval_source"] = "similarity"
        print(f"  Similarity #{i+1}: {doc.page_content[:120]}...")

    # 4B. MMR (diverse lexical+semantic)
    keyword_docs = vectorstore.max_marginal_relevance_search(query, k=5)
    print(f"MMR Search returned: {len(keyword_docs)} chunks")

    for i, doc in enumerate(keyword_docs):
        doc.metadata["retrieval_source"] = "mmr"
        print(f"  MMR #{i+1}: {doc.page_content[:120]}...")

    # 4C. Merge and deduplicate using hash of full content
    combined = vector_docs + keyword_docs
    seen = set()
    unique_docs = []

    for d in combined:
        key = hash(d.page_content)      # Full text hashing → correct dedup
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    print(f"\nUnique chunks after merge: {len(unique_docs)}")

    return unique_docs[:5]  # return top 5 chunks



# Wrap as Runnable so LangChain pipeline can call it
retriever = RunnableLambda(lambda q: hybrid_retriever(q))



###############################################################################
# 5. LOAD LLM
###############################################################################
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,   # deterministic answer
)



###############################################################################
# 6. PROMPT TEMPLATE (SAFE, NO TRIPLE-BACKTICKS)
###############################################################################
prompt = ChatPromptTemplate.from_template("""
You MUST follow these rules strictly:

1. Use ONLY the information in the provided context.
2. If the context does NOT explicitly contain the answer, say EXACTLY: "I don't know."
3. Do NOT use your own knowledge.
4. Do NOT infer or guess.

Context:
{context}

Question:
{question}
""")



###############################################################################
# 7. MERGE MULTIPLE CHUNKS INTO ONE CONTEXT BLOCK
###############################################################################
def combine_docs(docs):
    print("\nDEBUG - combine_docs received", len(docs), "chunks")
    return "\n\n".join(doc.page_content for doc in docs)



###############################################################################
# 8. COMPLETE RAG CHAIN
###############################################################################
rag_chain = (
    {
        "question": itemgetter("question"),
        "context": itemgetter("question") | retriever | combine_docs
    }
    | prompt
    | llm
    | StrOutputParser()
)



###############################################################################
# 9. TEST QUERY
###############################################################################
query = "what are the steps in RAG?"

print("\n--- Final Retrieved Chunks ---")
retrieved_docs = retriever.invoke(query)

for i, doc in enumerate(retrieved_docs):
    print(f"\nChunk #{i+1}")
    print("Retrieved via:", doc.metadata.get("retrieval_source"))
    print(doc.page_content)



###############################################################################
# 10. GET FINAL RAG ANSWER
###############################################################################
response = rag_chain.invoke({"question": query})

print("\n--- RAG RESULT ---")
print("Question:", query)
print("Answer:", response)
print("Done.")
