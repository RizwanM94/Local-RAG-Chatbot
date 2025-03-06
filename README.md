## RAG-Based Chatbot with Streamlit and LLaMA 2

This project is a **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload PDFs, extract and store text embeddings using **ChromaDB**, and interact with the documents using **LLaMA 2** locally via **Ollama**.

---

## Features 🚀
- Upload PDFs via **Streamlit UI**.
- Extract text using **pdfplumber**.
- Split text into chunks using **LangChain**.
- Generate embeddings using **SentenceTransformers**.
- Store embeddings in **ChromaDB** for retrieval.
- Retrieve relevant text chunks and generate answers using **LLaMA 2** (via Ollama).

---

## Installation & Setup 🛠️
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/local-rag-chatbot.git
cd local-rag-chatbot
```

### **2. Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run Streamlit App**
```bash
streamlit run app.py
```

---

## Usage 💡
1. **Upload a PDF** from the sidebar.
2. The document will be saved in the `uploads/` folder.
3. The text will be **extracted, chunked, and stored as embeddings** in **ChromaDB**.
4. Enter your question in the chatbox.
5. The chatbot retrieves **relevant document chunks** and generates responses using **LLaMA 2**.

---
## Notes 📝
- **Ensure you have Ollama installed** and running **LLaMA 2** locally.
- **If switching documents**, clear the ChromaDB embeddings to avoid outdated data:
  ```bash
  rm -rf embeddings/
  ```
- If you encounter **database errors**, ensure ChromaDB is correctly set up and check file paths.

🤝 Contributing
Feel free to submit issues and pull requests. Let's build this together! 🚀


📬 Contact
For queries or collaborations, reach out via rizzwanm94@gmail.com

Happy Coding! 🚀
---
