import os
import streamlit as st
from extract_text import extract_text_from_pdf
from process_text import split_text, store_embeddings
from query import retrieve_relevant_chunks, query_llama2

# Define the upload directory
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Streamlit UI
st.title("📄 Chat with your PDF (LLaMA 2 + ChromaDB)")
st.sidebar.header("Upload Your PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF to start chatting", type="pdf")

if uploaded_file:
    # Save uploaded file
    save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"Uploaded {uploaded_file.name}")

    # Extract and process text
    st.write("🔄 Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(save_path)

    st.write("🔄 Splitting text into chunks...")
    chunks = split_text(pdf_text)

    st.write("🔄 Storing embeddings in ChromaDB...")
    store_embeddings(chunks)
    st.success("✅ Processing completed! Start asking questions below.")

    # Chatbot UI
    st.header("💬 Ask a Question")
    user_question = st.text_input("Type your question here:")

    if user_question:
        retrieved_chunks = retrieve_relevant_chunks(user_question)
        context = " ".join(retrieved_chunks)

        final_prompt = f"Using only the following context, answer the question:\n{context}\n\nQuestion: {user_question}"
        response = query_llama2(final_prompt)

        st.subheader("🤖 LLaMA 2 Response")
        st.write(response)
