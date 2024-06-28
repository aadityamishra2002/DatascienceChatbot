# DatascienceChatbot
PDF Question Answering System
This Python application allows you to upload PDF documents, store their content in a MySQL database, and then ask questions about the documents. The system uses LangChain and OpenAI to provide intelligent answers based on the text within the PDFs.

Features
PDF Processing: Extracts text from uploaded PDFs and splits it into meaningful chunks.
MySQL Database: Stores the extracted text chunks for efficient retrieval.
Vector Search: Uses OpenAI embeddings and LangChain's Chroma vector store to perform semantic search on the PDF content.
Question Answering: Employs OpenAI's language model to generate answers to your questions based on the relevant text found in the PDFs.

Installation

Install dependencies:

pip install -r requirements.txt
Set up MySQL:

Create a MySQL database.
Update the database connection details in the code (host, user, password, database name).
Obtain OpenAI API Key:

Get an API key from OpenAI and replace the placeholder in the code.
Usage
Run the script:

python your_script_name.py
Upload PDF:

You will be prompted to enter the path to your PDF file.
Enter a name for the document to be stored in the database.
Ask Questions:

After the PDF is processed, you can start asking questions about its content.
Type "quit" to exit the application.
Dependencies
Python 3.x
PyPDF2
mysql.connector
LangChain
OpenAI
Chroma
Acknowledgments
This project utilizes:

OpenAI API for language processing and question answering.
LangChain for simplifying the integration of language models and data.
Chroma for vector storage and similarity search.
Contributing
Contributions are welcome! Feel free to open issues or pull requests to suggest improvements or fix bugs.
