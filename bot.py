import streamlit as st
import PyPDF2
import google.generativeai as genai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure page
st.set_page_config(
    page_title="Know Your Doc",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown(
    """
    <style>
    /* Dark background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f5f5f5;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161a23;
        color: #ffffff;
    }
    /* Buttons */
    button[kind="primary"] {
        background-color: #4a90e2 !important;
        color: white !important;
        border-radius: 8px;
    }
    button {
        border-radius: 8px;
    }
    /* Answer box */
    .answer-box {
        background-color: #1e1e2f;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        margin-bottom: 15px;
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.1);
    }
    /* Context chunks */
    .context-box {
        background-color: #2b2b40;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 14px;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- STATE ----------------
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# ---------------- HELPERS ----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            last_boundary = max(last_period, last_newline)
            if last_boundary > start + chunk_size // 2:
                end = start + last_boundary + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if end >= len(text):
            break
    return chunks

@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_vector_store(text_chunks, embeddings_model):
    if not text_chunks:
        return None
    embeddings = embeddings_model.encode(text_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def search_similar_chunks(query, vector_store, text_chunks, embeddings_model, k=3):
    if vector_store is None or not text_chunks:
        return []
    query_embedding = embeddings_model.encode([query])
    distances, indices = vector_store.search(query_embedding.astype('float32'), k)
    return [text_chunks[idx] for idx in indices[0] if idx < len(text_chunks)]

def generate_answer(query, context_chunks, gemini_api_key, model_name="gemini-2.0-flash-exp"):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name)
    context = "\n\n".join(context_chunks)
    prompt = f"""
    Based on the following context from the PDF document, please answer the question.
    
    Context:
    {context}
    
    Question: {query}
    
    Please provide a detailed answer based only on the information provided in the context. 
    If the answer cannot be found in the context, please say so.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# ---------------- MAIN APP ----------------
st.title("📚 Know Your Doc")
st.markdown("Upload a PDF and ask questions about its content using AI-powered search!")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    model_option = st.selectbox(
        "Select Gemini Model:",
        ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
        index=0
    )
    gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password")

# Upload + Process Flow
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if st.button("📤 Upload PDF", type="primary"):
            st.session_state.uploaded_file = uploaded_file
            st.success(f"File uploaded: {uploaded_file.name}")

    if st.session_state.uploaded_file:
        if st.button("🔄 Process PDF", type="primary"):
            if not gemini_api_key:
                st.error("Please provide your Gemini API key first!")
            else:
                with st.spinner("Processing PDF..."):
                    pdf_text = extract_text_from_pdf(st.session_state.uploaded_file)
                    if pdf_text.strip():
                        st.session_state.text_chunks = chunk_text(pdf_text)
                        if st.session_state.embeddings_model is None:
                            st.session_state.embeddings_model = load_embeddings_model()
                        st.session_state.vector_store = create_vector_store(
                            st.session_state.text_chunks, st.session_state.embeddings_model
                        )
                        st.session_state.pdf_processed = True
                        st.success(f"✅ PDF processed successfully! {len(st.session_state.text_chunks)} text chunks created.")
                    else:
                        st.error("Could not extract text from the PDF.")

with col2:
    st.header("💬 Ask Questions")
    if st.session_state.pdf_processed:
        user_question = st.text_area("Ask a question about the PDF:", height=100)
        if st.button("🔍 Get Answer", type="primary"):
            with st.spinner("Thinking..."):
                relevant_chunks = search_similar_chunks(
                    user_question,
                    st.session_state.vector_store,
                    st.session_state.text_chunks,
                    st.session_state.embeddings_model,
                    k=5
                )
                if relevant_chunks:
                    answer = generate_answer(user_question, relevant_chunks, gemini_api_key, model_option)
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.subheader("🎯 Answer:")
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("📖 Source Context"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f'<div class="context-box"><b>Chunk {i+1}:</b><br>{chunk}</div>', unsafe_allow_html=True)
                else:
                    st.warning("No relevant context found. Try rephrasing your question.")
