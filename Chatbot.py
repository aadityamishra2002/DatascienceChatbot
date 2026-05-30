import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader

# --- Configuration ---
GOOGLE_API_KEY = st.secrets["google_api_key"]

# --- Initialisation ---
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# --- Helper Functions ---
def load_pdf(uploaded_file):
    raw_text = ""
    pdfreader = PdfReader(uploaded_file)
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def build_vector_store(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    vector_store = Chroma.from_texts(texts, embedding)
    return vector_store

# --- Streamlit App ---
st.set_page_config(page_title="DocuBot", page_icon="📄")
st.title("DocuBot — Ask Your PDF Questions")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner("Processing PDF..."):
            raw_text = load_pdf(uploaded_file)
            st.session_state.vector_store = build_vector_store(raw_text)
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.messages = [
                {"role": "assistant", "content": f"Ready! I've processed **{uploaded_file.name}**. Ask me anything about it."}
            ]
        st.success("PDF processed successfully!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a PDF above to get started."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDF..."):
    if "vector_store" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 4}
                    )
                )
                answer = qa_chain.run(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
