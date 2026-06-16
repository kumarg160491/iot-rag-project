from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from config import cfg

PROMPT_TEMPLATE = """
You are an expert IoT maintenance engineer assistant.
You help diagnose machine faults and suggest corrective actions.

Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I don't have enough information in the knowledge base to answer this."

Context:
{context}

Sensor Anomaly / Question:
{question}

Provide a structured diagnosis:
- Likely Cause:
- Risk Level (Low / Medium / High / Critical):
- Recommended Action:
- Source Reference:
"""

def build_rag_chain():
    embeddings = OllamaEmbeddings(
        model    = cfg.ollama.embedding,
        base_url = cfg.ollama.base_url,
    )
    vectorstore = Chroma(
        collection_name    = cfg.chroma.collection_name,
        embedding_function = embeddings,
        persist_directory  = cfg.chroma.db_path,
    )
    retriever = vectorstore.as_retriever(
        search_type  = cfg.retriever.search_type,
        search_kwargs= {"k": cfg.retriever.top_k}
    )
    llm = OllamaLLM(
        model       = cfg.ollama.model,
        base_url    = cfg.ollama.base_url,
        temperature = cfg.ollama.temperature,
    )
    prompt = PromptTemplate(
        template       = PROMPT_TEMPLATE,
        input_variables= ["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm                    = llm,
        chain_type             = "stuff",
        retriever              = retriever,
        return_source_documents= True,
        chain_type_kwargs      = {"prompt": prompt}
    )
    return chain

def query_rag(chain, question: str) -> dict:
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "source"  : doc.metadata.get("source", "Unknown"),
                "doc_type": doc.metadata.get("doc_type", "Unknown"),
                "page"    : doc.metadata.get("page", "N/A"),
            }
            for doc in result["source_documents"]
        ]
    }