import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Configuration
st.set_page_config(
    page_title="Kissan Copilot - ASZ", page_icon="🌾", layout="wide"
)

# Custom Green Executive Theme & Dark Purple High-Contrast Chat Box Styling
st.markdown(
    """
    <style>
    .stApp { 
        background-color: #0d1f17 !important; 
        color: #f0fdf4 !important; 
    }
    
    [data-testid="stSidebar"] { 
        background-color: #132e22 !important; 
        border-right: 1px solid #1f4e38 !important; 
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { 
        color: #e2e8f0 !important; 
    }
    
    .stChatInputContainer {
        background-color: #000000 !important;
        border: 1px solid #1f4e38 !important;
        border-radius: 8px !important;
    }
    .stChatInput textarea {
        color: #ffffff !important;
        background-color: #000000 !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    .stChatInput textarea::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
    }

    [data-testid="stChatMessage"] {
        background-color: #2e1065 !important;
        border: 1px solid #7c3aed !important;
        border-radius: 10px !important;
        padding: 10px !important;
        color: #f3e8ff !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] li {
        color: #f3e8ff !important;
        font-weight: 500 !important;
    }

    .main-title { font-size: 28px; font-weight: 800; color: #f0fdf4 !important; }
    .badge { background: linear-gradient(135deg, #059669, #10b981); color: #ffffff !important; padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; }
    .status-box { background-color: #064e3b !important; border-left: 4px solid #34d399; padding: 10px; border-radius: 4px; font-size: 13px; color: #d1fae5 !important; margin-bottom: 15px; }
    .disclaimer { font-size: 11px; color: #94a3b8 !important; border-top: 1px solid #1f4e38; padding-top: 10px; margin-top: 20px; }
    </style>
""",
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
  st.markdown("### 🌾 Kissan Copilot")
  st.markdown(
      '<div class="status-box">🟢 <b>System Status:</b> Online (No API Key'
      " Required)<br>⚡ <b>Engine:</b> Direct Vector Search RAG</div>",
      unsafe_allow_html=True,
  )

  st.markdown("---")
  st.markdown("### ⚙️ Enterprise Settings")
  region = st.selectbox(
      "Select Region / Province",
      ["Punjab", "Sindh", "Khyber Pakhtunkhwa (KP)", "Balochistan"],
  )
  crop_type = st.selectbox(
      "Select Target Crop",
      [
          "Wheat (Gandum)",
          "Cotton (Kapas)",
          "Rice (Chawal)",
          "Sugarcane",
          "Maize",
      ],
  )

  response_lang = st.selectbox(
      "Response Language / Zuban",
      [
          "English",
          "Roman Urdu (رومن اردو)",
          "Urdu (اردو)",
      ],
  )

  st.markdown("---")
  st.markdown("### 📚 Knowledge Base (RAG)")
  uploaded_file = st.file_uploader(
      "Upload Official Agri Guideline (PDF)", type=["pdf"]
  )

# Main UI Header
st.markdown(
    '<p class="main-title">🌾 Kissan Copilot <span class="badge">represented by'
    " ASZ</span></p>",
    unsafe_allow_html=True,
)
st.markdown(
    f"Providing certified, localized advisory for **{crop_type}** in"
    f" **{region}** in *{response_lang}* (Offline Mode)."
)

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])


@st.cache_resource
def get_vectorstore(file_bytes, file_name):
  os.makedirs("data", exist_ok=True)
  file_path = os.path.join("data", file_name)
  with open(file_path, "wb") as f:
    f.write(file_bytes)

  loader = PyPDFLoader(file_path)
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  )
  docs = text_splitter.split_documents(documents)

  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vectorstore = FAISS.from_documents(docs, embeddings)
  return vectorstore


if prompt := st.chat_input(
    f"Ask questions about {crop_type} ({response_lang})..."
):
  if not uploaded_file:
    st.warning(
        "Please upload an official reference PDF guideline in the sidebar to"
        " enable vector search."
    )
  else:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      with st.spinner("Searching directly inside PDF document..."):
        try:
          vectorstore = get_vectorstore(
              uploaded_file.getvalue(), uploaded_file.name
          )
          retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
          relevant_docs = retriever.invoke(prompt)

          extracted_snippets = "\n\n---\n\n".join(
              [
                  f"> {doc.page_content}"
                  for doc in relevant_docs
              ]
          )
          source_page = (
              relevant_docs[0].metadata.get("page", 1)
              if relevant_docs
              else "N/A"
          )

          final_output = (
              f"**Relevant Information extracted from document:**\n\n{extracted_snippets}\n\n*🔍 **Direct Source Match:** Found in `{uploaded_file.name}` (Page {source_page}) for {region} ({crop_type}).*"
          )

          st.markdown(final_output)
          st.session_state.messages.append(
              {"role": "assistant", "content": final_output}
          )

        except Exception as e:
          st.error(f"An error occurred: {e}")

st.markdown(
    '<p class="disclaimer"><b>Responsible AI & Accountability Disclaimer:</b>'
    " Direct Vector RAG search active (No API Key Required).</p>",
    unsafe_allow_html=True,
)
