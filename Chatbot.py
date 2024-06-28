import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import cassio
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# Function to load and process the PDF
def load_pdf(file_path):
    pdfreader = PdfReader(file_path)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Streamlit app
def main():
    st.title("DataAnalyst - GPT")

    # Prefilled values for connection parameters and API keys
    astra_db_token = "Your_astradb_token"
    astra_db_id = "Your_astradb_id"
    openai_api_key = "Your_openai_key"

    # File uploader for PDF file
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if st.button("Load PDF and Initialize"):
        if not astra_db_token or not astra_db_id or not openai_api_key:
            st.error("Please provide all required secrets.")
        else:
            # Load PDF
            raw_text = load_pdf(uploaded_file)

            try:
                # Initialize connection to Astra DB
                cassio.init(token=astra_db_token, database_id=astra_db_id)

                # Create LangChain components
                llm = OpenAI(openai_api_key=openai_api_key)
                embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

                # Create LangChain vector store backed by Astra DB
                astra_vector_store = Cassandra(
                    embedding=embedding,
                    table_name="qa_mini_demo",
                    session=None,
                    keyspace=None,
                )

                # Split the text into chunks
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=800,
                    chunk_overlap=200,
                    length_function=len,
                )
                texts = text_splitter.split_text(raw_text)

                # Load the dataset into the vector store
                astra_vector_store.add_texts(texts[:50])
                st.success(f"Inserted {len(texts[:50])} text chunks into the vector store.")
                st.session_state.astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
                st.session_state.llm = llm

            except ValueError as e:
                st.error(f"Failed to initialize Astra DB connection: {e}")
                st.stop()

    if 'astra_vector_index' in st.session_state and 'llm' in st.session_state:
        st.header("Question-Answering")

        query_text = st.text_input("Enter your question (or type 'quit' to exit):")

        if query_text.lower() != "quit" and query_text != "":
            st.write(f"*QUESTION:* \"{query_text}\"")
            answer = st.session_state.astra_vector_index.query(query_text, llm=st.session_state.llm).strip()
            st.write(f"*ANSWER:* \"{answer}\"")

if __name__ == "_main_":
    main()
