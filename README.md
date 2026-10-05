# RAG System 📚

A Retrieval-Augmented Generation (RAG) system for questioning and retrieving information from PDF documents. This system embeds your documents using `sentence-transformers` and stores them in a persistent `chromadb` vector store. It utilizes a local Ollama model (`qwen3:8b`) to generate intelligent answers based on the retrieved context.

## Features
- **Interactive UI**: A Streamlit web application that lets you chat with your documents.
- **Dynamic PDF Upload**: Upload your PDF files directly from the UI. The system automatically reads, chunks, and embeds them into the vector database on the fly.
- **Local AI**: Fully integrated with Ollama to run inference locally using the `qwen3:8b` model for maximum privacy.
- **Adjustable Retrieval**: Use the UI to control exactly how many document chunks (Top-K) are used to generate each response.

## Files Structure
- **Jupyter Notebooks**: Exploratory notebooks (`RAG (2).ipynb`, `RAG_PIPE_LINE.ipynb`, `RAG_pipe.ipynb`) used for initial RAG pipeline testing and chunking experiments.
- **data/vector_store/**: The directory where the persistent ChromaDB database saves your document embeddings.
- **app.py**: The interactive Streamlit application serving as the frontend and backend of the chatbot.
- **requirements.txt**: Python dependencies required to run the system.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed. It is recommended to use a virtual environment. Install the necessary packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Interactive UI
Launch the Streamlit app to start chatting:

```bash
streamlit run app.py
```

### 3. Application Usage
1. Open the local URL provided by Streamlit (usually `http://localhost:8501`).
2. Ensure you have Ollama installed and the `qwen3:8b` model pulled locally (`ollama run qwen3:8b`).
3. Use the **Add new PDF** uploader in the sidebar to add context.
4. Set the **Top-K Documents to Retrieve** slider to determine how much context is fetched for the AI.
5. Use the chat interface to ask questions. You can expand the retrieved context section below each answer to see the specific chunks and similarity scores.
