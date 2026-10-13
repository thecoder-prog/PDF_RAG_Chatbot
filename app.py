
import hashlib

import chromadb
import streamlit as st

from rag_pipeline import (
    extract_pdf_pages,
    create_chunks,
    generate_embeddings,
    store_chunks,
    retrieve_chunks,
    generate_answer,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CSS — RESPONSIVE DARK CHAT INTERFACE
# ============================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    /* Global background */
    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #212121 !important;
        color: #eeeeee !important;
    }

    [data-testid="stHeader"] {
        background-color: #212121 !important;
    }

    [data-testid="stToolbar"],
    footer {
        visibility: hidden;
    }

    /* Main content layout */
    .block-container {
        max-width: 1000px !important;
        width: 100% !important;
        padding: 24px clamp(16px, 5vw, 60px) 120px !important;
        margin: 0 auto !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background-color: #171717 !important;
        color: #eeeeee !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #303030;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 22px 16px;
    }

    [data-testid="stSidebar"] * {
        color: #eeeeee;
    }

    /* Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        padding: 12px 0 18px;
        margin-bottom: 24px;
        border-bottom: 1px solid #383838;
        font-size: 16px;
        font-weight: 600;
    }

    .app-header .subtitle {
        color: #a3a3a3 !important;
        font-size: 13px;
        font-weight: 400;
    }

    /* Welcome screen */
    .welcome-container {
        width: 100%;
        max-width: 720px;
        margin: clamp(45px, 12vh, 110px) auto 36px;
        text-align: center;
        padding: 0 12px;
    }

    .welcome-title {
        font-size: clamp(25px, 3vw, 36px);
        line-height: 1.3;
        font-weight: 650;
        letter-spacing: -0.7px;
        color: #f5f5f5 !important;
        margin-bottom: 14px;
    }

    .welcome-subtitle {
        color: #a3a3a3 !important;
        font-size: clamp(14px, 1.5vw, 16px);
        line-height: 1.7;
    }

    /* Typography */
    h1, h2, h3, p, label {
        color: #eeeeee;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 14px 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background-color: transparent !important;
        color: #eeeeee !important;
        line-height: 1.8;
        min-width: 0;
        overflow-wrap: anywhere;
    }

    /* ========================================================
       PDF UPLOADER — DARK DROP ZONE
       ======================================================== */

    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section > div,
    [data-testid="stFileUploaderDropzone"] {
        background: #262626 !important;
        background-color: #262626 !important;
        color: #eeeeee !important;
        border-color: #484848 !important;
    }

    [data-testid="stFileUploader"] {
        border-radius: 12px !important;
        padding: 8px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #555555 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: #383838 !important;
        color: #eeeeee !important;
        border: 1px solid #555555 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background-color: #484848 !important;
    }

    /* ========================================================
       FIX WHITE UPLOADED-FILE CARD
       ======================================================== */

    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] > div,
    [data-testid="stFileUploaderFile"] > div > div,
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileDetails"],
    [data-testid="stFileUploaderFileInfo"] {
        background: #303030 !important;
        background-color: #303030 !important;
        color: #eeeeee !important;
        border-color: #484848 !important;
    }

    /* Make nested preview elements transparent so the card
       background remains visible. */
    [data-testid="stFileUploaderFile"] * {
        color: #eeeeee !important;
        border-color: #484848 !important;
    }

    /* Filename and file details */
    [data-testid="stFileUploaderFile"] p,
    [data-testid="stFileUploaderFile"] span,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] a {
        color: #eeeeee !important;
    }

    /* Remove file button */
    [data-testid="stFileUploaderFile"] button {
        background-color: #383838 !important;
        color: #eeeeee !important;
        border: 1px solid #555555 !important;
        border-radius: 50% !important;
    }

    /* ========================================================
       CHAT INPUT — DARK BACKGROUND
       ======================================================== */

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background-color: #212121 !important;
        color: #eeeeee !important;
        border: none !important;
    }

    [data-testid="stBottom"] {
        background: linear-gradient(
            to bottom,
            rgba(33, 33, 33, 0),
            #212121 22%
        ) !important;
        padding: 18px 0 12px !important;
    }

    [data-testid="stChatInput"],
    [data-testid="stBottom"] [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div {
        background-color: #303030 !important;
        color: #eeeeee !important;
        border: 1px solid #505050 !important;
        border-radius: 22px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #777777 !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus {
        background-color: #303030 !important;
        color: #eeeeee !important;
        -webkit-text-fill-color: #eeeeee !important;
        border: none !important;
        box-shadow: none !important;
        padding: 13px 16px !important;
        min-height: 48px !important;
        max-height: 180px !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #a3a3a3 !important;
        -webkit-text-fill-color: #a3a3a3 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #eeeeee !important;
        color: #171717 !important;
        border: none !important;
        border-radius: 50% !important;
        margin-right: 8px !important;
    }

    /* Standard text inputs */
    .stTextInput input,
    .stTextArea textarea {
        background-color: #303030 !important;
        color: #eeeeee !important;
        border: 1px solid #484848 !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: 12px !important;
        background-color: #303030 !important;
        color: #eeeeee !important;
        border: 1px solid #454545 !important;
        transition: background-color 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #414141 !important;
        color: #ffffff !important;
        border-color: #666666 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #262626 !important;
        color: #eeeeee !important;
        border: 1px solid #3b3b3b !important;
        border-radius: 12px !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background-color: #292929 !important;
        color: #eeeeee !important;
        border-radius: 12px !important;
    }

    /* Responsive layout */
    @media (max-width: 768px) {
        .block-container {
            padding: 16px 16px 110px !important;
        }

        .app-header {
            padding-bottom: 14px;
            margin-bottom: 18px;
        }

        .welcome-container {
            margin-top: 45px;
            margin-bottom: 24px;
        }

        .welcome-title {
            font-size: 26px;
        }

        [data-testid="stBottom"] {
            padding: 12px 8px 8px !important;
        }

        [data-testid="stChatInput"] {
            border-radius: 18px !important;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 12px !important;
            padding-right: 12px !important;
        }

        .app-header {
            font-size: 14px;
        }

        .app-header .subtitle {
            font-size: 11px;
        }

        .welcome-title {
            font-size: 23px;
        }

        .welcome-subtitle {
            font-size: 14px;
        }

        [data-testid="stChatMessage"] {
            padding: 10px 0 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_document_id" not in st.session_state:
    st.session_state.indexed_document_id = None

if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ============================================================
# 4. SIDEBAR — UPLOAD AND INDEX PDF
# ============================================================

with st.sidebar:

    st.markdown("## 📚 PDF AI Assistant")
    st.caption("Chat with your documents using RAG.")

    st.divider()

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("### Your document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Select a PDF document to ask questions about.",
    )

    current_document_id = None
    pdf_bytes = None

    if uploaded_file is not None:

        pdf_bytes = uploaded_file.getvalue()

        current_document_id = hashlib.sha256(
            pdf_bytes
        ).hexdigest()[:16]

        st.caption(f"📄 {uploaded_file.name}")

        if st.button(
            "Index PDF",
            type="primary",
            use_container_width=True,
        ):

            try:

                if not pdf_bytes:
                    st.error("The uploaded PDF is empty.")

                else:

                    with st.spinner("Extracting PDF text..."):

                        pages = extract_pdf_pages(
                            pdf_bytes,
                            uploaded_file.name,
                        )

                    if not pages:

                        st.error(
                            "No readable text was found. "
                            "This PDF may require OCR."
                        )

                    else:

                        with st.spinner("Creating text chunks..."):

                            chunks = create_chunks(
                                pages,
                                chunk_size=1000,
                                chunk_overlap=200,
                            )

                        if not chunks:

                            st.error("No text chunks were created.")

                        else:

                            with st.spinner(
                                "Generating embeddings..."
                            ):

                                embedded_chunks = generate_embeddings(
                                    chunks
                                )

                            with st.spinner(
                                "Saving vectors to ChromaDB..."
                            ):

                                collection = store_chunks(
                                    embedded_chunks,
                                    current_document_id,
                                )

                            st.session_state.indexed_document_id = (
                                current_document_id
                            )

                            st.session_state.indexed_filename = (
                                uploaded_file.name
                            )

                            st.session_state.messages = []
                            st.session_state.pending_question = None

                            st.success(
                                f"Indexed successfully! "
                                f"{collection.count()} chunks saved."
                            )

                            st.rerun()

            except Exception as error:

                st.error(f"Indexing failed: {error}")

    # Index status
    st.divider()
    st.markdown("### Index status")

    indexed_id = st.session_state.indexed_document_id

    if (
        indexed_id is not None
        and indexed_id == current_document_id
    ):

        st.success("Document ready")

        st.caption(
            f"Active PDF: {st.session_state.indexed_filename}"
        )

    elif indexed_id is not None:

        st.warning(
            "The selected PDF is not indexed. "
            "Click Index PDF to switch documents."
        )

    else:

        st.info("Upload and index a PDF to begin.")

    st.divider()
    st.caption("Powered by")
    st.markdown("**ChromaDB · Sentence Transformers · OpenAI**")


# ============================================================
# 5. MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <span>📚 PDF AI Chat</span>
        <span class="subtitle">Your document assistant</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. DOCUMENT STATUS
# ============================================================

document_is_ready = (
    uploaded_file is not None
    and current_document_id is not None
    and st.session_state.indexed_document_id
    == current_document_id
)


# ============================================================
# 7. WELCOME SCREEN
# ============================================================

if not document_is_ready:

    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-title">
                What can I help you understand?
            </div>
            <div class="welcome-subtitle">
                Upload a PDF, index it, and ask questions
                about its contents using AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Upload a PDF from the sidebar and click "
        "'Index PDF' to start chatting."
    )


# ============================================================
# 8. CHAT INTERFACE
# ============================================================

else:

    # Welcome screen and suggested questions
    if not st.session_state.messages:

        st.markdown(
            """
            <div class="welcome-container">
                <div class="welcome-title">
                    Ask anything about your PDF
                </div>
                <div class="welcome-subtitle">
                    Explore key findings, understand concepts,
                    or summarize your document.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Try asking")

        suggestions = [
            "What is the main objective of this document?",
            "Summarize the key points.",
            "What are the main conclusions?",
        ]

        cols = st.columns(3)

        for index, suggestion in enumerate(suggestions):

            with cols[index]:

                if st.button(
                    suggestion,
                    key=f"suggestion_{index}",
                    use_container_width=True,
                ):

                    st.session_state.pending_question = suggestion
                    st.rerun()

    # Display conversation history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander("📄 View source passages"):

                    for rank, result in enumerate(
                        message["sources"],
                        start=1,
                    ):

                        metadata = result["metadata"]

                        st.markdown(
                            f"**Source {rank}:** "
                            f"{metadata.get('source', 'Unknown')} "
                            f"— Page {metadata.get('page', '?')}"
                        )

                        st.write(result["text"])

                        st.caption(
                            "Cosine distance: "
                            f"{result['distance']:.4f}"
                        )

    # Chat input
    typed_question = st.chat_input(
        "Message PDF AI..."
    )

    pending_question = st.session_state.pending_question
    st.session_state.pending_question = None

    question = typed_question or pending_question

    if question:

        question = question.strip()

        if question:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            # Retrieve context and generate the answer
            try:

                with st.chat_message("assistant"):

                    with st.spinner("Searching your PDF..."):

                        client = chromadb.PersistentClient(
                            path="./chroma_db"
                        )

                        collection = client.get_collection(
                            name=f"pdf_{current_document_id}"
                        )

                        results = retrieve_chunks(
                            question,
                            collection,
                            top_k=4,
                        )

                    if not results:

                        answer = (
                            "I couldn't find relevant passages "
                            "in the indexed document for this question."
                        )

                        sources = []

                    else:

                        with st.spinner("Generating answer..."):

                            answer = generate_answer(
                                question,
                                results,
                            )

                        sources = results

                    st.markdown(answer)

                    if sources:

                        with st.expander(
                            "📄 View source passages"
                        ):

                            for rank, result in enumerate(
                                sources,
                                start=1,
                            ):

                                metadata = result["metadata"]

                                st.markdown(
                                    f"**Source {rank}:** "
                                    f"{metadata.get('source', 'Unknown')} "
                                    f"— Page {metadata.get('page', '?')}"
                                )

                                st.write(result["text"])

                                st.caption(
                                    "Cosine distance: "
                                    f"{result['distance']:.4f}"
                                )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

            except Exception as error:

                error_message = (
                    "I couldn't process your question. "
                    f"Error: {error}"
                )

                with st.chat_message("assistant"):
                    st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
