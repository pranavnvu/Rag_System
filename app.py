import streamlit as st
import os
import uuid
import tempfile
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Classes adapted from the RAG Pipeline Notebook
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        
    def generate_embeddings(self, text):
        return self.model.encode(text, show_progress_bar=False)

class VectorStoreManager:
    def __init__(self, persist_directory="data/vector_store", collection_name="pdf_documents"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.initialize_store()

    def initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Vector store collection for PDF embeddings in RAG", "hnsw:space": "cosine"}
        )
        
    def add_documents(self, documents, embeddings):
        if not documents: return
        ids = [f"doc_{uuid.uuid4()}" for _ in documents]
        all_metadata = []
        documents_content = []
        embeddings_list = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            metadata = dict(doc.metadata) if hasattr(doc, "metadata") else {}
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            all_metadata.append(metadata)
            documents_content.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
            
        self.collection.add(
            ids=ids,
            metadatas=all_metadata,
            documents=documents_content,
            embeddings=embeddings_list
        )

class RAGRetriever:
    def __init__(self, emb_manager, vector_store):
        self.emb_manager = emb_manager
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5, score_threshold=0.0):
        query_embedding = self.emb_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]
            for i, (doc_id, metadata, document, distance) in enumerate(zip(ids, metadatas, documents, distances)):
                similarity_score = 1 - distance
                if similarity_score >= score_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "document": document,
                        "metadata": metadata,
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })
        return retrieved_docs

# Streamlit App
st.set_page_config(page_title="Interactive RAG UI", layout="wide")

# Hide Deploy button and all menu items except the mode/theme switcher
st.markdown("""
<style>
/* Hide Deploy button */
.stDeployButton,
[data-testid="stToolbarActions"],
[data-testid="stDeployButton"] {
    display: none !important;
}

/* Inside the 3-dots menu: keep ONLY the theme/mode switcher buttons */
[data-testid="stMainMenuList"] > :not([data-testid="stThemeSwitcher"]) {
    display: none !important;
}

/* Hide menu dividers */
[data-testid="stMainMenuDivider"],
[data-testid="stMainMenuList"] [role="separator"] {
    display: none !important;
}

/* Hide version info and footer */
[data-testid="stMainMenuList"] ~ * {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Interactive RAG Assistant 📚")
st.write("Ask questions based on your embedded PDFs in the `data/vector_store` directory.")

# Cache resources to prevent reloading on every interaction
@st.cache_resource
def load_rag_components():
    with st.spinner("Loading Embedding Model and Vector Store..."):
        emb_manager = EmbeddingManager()
        vector_store = VectorStoreManager()
        rag_retriever = RAGRetriever(emb_manager, vector_store)
    return emb_manager, vector_store, rag_retriever

emb_manager, vector_store, retriever = load_rag_components()

# Sidebar Configuration
st.sidebar.title("Configuration ⚙️")
st.sidebar.info("Connected to local Ollama (model: qwen3:8b)")
llm = ChatOllama(model="qwen3:8b", temperature=0.1, max_tokens=1024)

top_k = st.sidebar.slider("Top-K Documents to Retrieve", min_value=1, max_value=10, value=3)

st.sidebar.divider()
st.sidebar.subheader("Add new PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF to the Vector Store", type=["pdf"])
if uploaded_file is not None:
    if st.sidebar.button("Process PDF"):
        with st.spinner("Processing and Embedding PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                
            try:
                # Load PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Chunk PDF
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(docs)
                
                # Embed chunks
                texts = [doc.page_content for doc in chunks]
                embeddings = emb_manager.generate_embeddings(texts)
                
                # Add to vector store
                vector_store.add_documents(chunks, embeddings)
                st.sidebar.success(f"Added {len(chunks)} chunks to the database!")
            except Exception as e:
                st.sidebar.error(f"Error processing PDF: {e}")
            finally:
                os.unlink(tmp_path)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("View Retrieved Context"):
                st.write(msg["context"])

query = st.chat_input("Ask a question about your documents...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if llm is None:
        st.error("Please configure the API Key for the selected model.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    # Retrieve docs
                    results = retriever.retrieve(query, top_k=top_k)
                    context_list = [doc["document"] for doc in results] if results else []
                    context_text = "\n\n---\n\n".join(context_list)
                    
                    if context_text:
                        # Construct Prompt
                        prompt = f"""Use the given context to answer the query.
Context: 
{context_text}

Query: {query}
Answer:"""
                        # Generate response
                        response = llm.invoke(prompt)
                        answer = response.content
                        
                        # Show answer
                        st.markdown(answer)
                        
                        # Show context in an expander
                        with st.expander("View Retrieved Context"):
                            for i, doc in enumerate(results):
                                st.markdown(f"**Document {i+1} (Score: {doc['similarity_score']:.3f})**")
                                st.markdown(doc["document"])
                                st.markdown("---")
                                
                        # Save assistant response to history
                        st.session_state.messages.append({"role": "assistant", "content": answer, "context": context_text})
                    else:
                        st.warning("No relevant context found in the vector store.")
                        st.session_state.messages.append({"role": "assistant", "content": "No relevant context found in the vector store.", "context": ""})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
