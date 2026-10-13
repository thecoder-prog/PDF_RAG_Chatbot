# PDF RAG Chatbot

An AI-powered PDF Question Answering application built using Python and Streamlit. This project uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded PDF documents and generate context-aware answers using a Large Language Model (LLM).

## Features

- **PDF Upload:** Upload PDF documents through a user-friendly interface.
- **Text Extraction:** Extract text from PDF pages for processing.
- **Text Chunking:** Split document content into smaller chunks for efficient retrieval.
- **Embeddings Generation:** Convert text chunks into vector representations.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB.
- **Semantic Search:** Retrieve relevant document chunks based on user questions.
- **AI-Powered Answers:** Generate answers using retrieved context and an LLM.
- **Interactive UI:** Use a Streamlit interface to upload documents and ask questions.

## Technology Stack

- Python
- Streamlit
- ChromaDB
- Embedding Models
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

## Project Structure

```text
PDF_RAG_Chatbot/
│
├── app.py              # Streamlit application interface
├── rag_pipeline.py    # PDF processing and RAG pipeline
├── requirements.txt   # Python dependencies
├── .gitignore         # Files excluded from Git
└── README.md          # Project documentation
```

The `chroma_db/` directory may be created locally to persist vector embeddings, depending on the application configuration. It is excluded from version control when configured in `.gitignore`.

## How It Works

1. Upload a PDF document.
2. Extract text from the PDF.
3. Split the extracted text into manageable chunks.
4. Generate embeddings for the chunks.
5. Store the embeddings in ChromaDB.
6. Enter a question about the uploaded document.
7. Retrieve the most relevant chunks using semantic search.
8. Pass the retrieved context and question to an LLM.
9. Display the generated answer in the Streamlit application.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PDF_RAG_Chatbot.git
cd PDF_RAG_Chatbot
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API credentials

If your selected LLM provider requires an API key, configure it using environment variables or Streamlit secrets.

Do not hardcode API keys in source code or commit them to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Deployment

The application can be deployed using Streamlit Community Cloud, subject to the application's dependencies, API requirements, and available resources.

Deployment: https://share.streamlit.io/

## Security and Data Handling

- Do not upload confidential PDFs or sensitive documents to a public repository.
- Do not commit API keys, passwords, or `.env` files.
- Exclude generated vector databases and temporary files when they are not required.
- Review the privacy policies of any external LLM or embedding provider before sending document content.

## Future Improvements

- Support for multiple PDF documents.
- Display page numbers and document citations in answers.
- Add chat history and conversational memory.
- Support additional document formats such as DOCX and TXT.
- Improve retrieval using hybrid search and reranking.
- Add answer evaluation and retrieval quality metrics.

## Author

**Yogesh Bhore**

GitHub: https://github.com/thecoder-prog

## License

No license has been specified yet. All rights are reserved unless a license is added to this repository. Contact the author for permission to reuse or distribute the code.
