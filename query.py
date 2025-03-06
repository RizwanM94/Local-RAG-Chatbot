import subprocess
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB & embedding model
chroma_client = chromadb.PersistentClient(path="embeddings/")
collection = chroma_client.get_or_create_collection(name="pdf_data")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve top-k most relevant chunks from ChromaDB."""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return [item["text"] for item in results["metadatas"][0]]

def query_llama2(prompt):
    """Send a prompt to LLaMA 2 via Ollama."""
    cmd = ["ollama", "run", "llama2", prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

if __name__ == "__main__":
    user_question = "Summarize the document."
    retrieved_chunks = retrieve_relevant_chunks(user_question)
    context = " ".join(retrieved_chunks)

    final_prompt = f"Using only the following context, answer the question:\n{context}\n\nQuestion: {user_question}"
    response = query_llama2(final_prompt)
    print("LLaMA 2 Response:", response)
