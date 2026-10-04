"""
Subject-Aware AI University Learning Assistant
Main Streamlit Application.
Provides subject management, document processing, persistent FAISS indexing,
hybrid retrieval RAG chat, deep conceptual explanations, and a 10-question timed quiz.
"""

import os
import time
import base64
import streamlit as st
from typing import List, Dict, Any, Optional

# Set page layout first
st.set_page_config(
    page_title="AI University Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import internal modules
from modules.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_KEYWORD_WEIGHT,
    DEFAULT_TOP_K,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_GROQ_MODEL,
    QUIZ_TIME_LIMIT_SECONDS,
    QUIZ_QUESTION_COUNT,
)
from modules.storage import (
    list_subjects,
    create_subject,
    delete_subject,
    get_subject_stats,
    get_global_stats,
    compute_file_hash,
    is_duplicate_document,
    save_document_chunks,
    load_all_subject_chunks,
    load_registry,
    delete_document,
    save_original_document,
    load_original_document,
    get_original_document_path,
    load_document_chunks,
    save_last_active_subject,
    get_last_active_subject,
)
from modules.extractor import extract_document
from modules.chunker import create_chunks
from modules.indexer import (
    get_embedding_model,
    build_and_save_index,
    load_subject_index,
)
from modules.retriever import hybrid_search, build_context
from modules.llm_service import (
    get_groq_api_key,
    get_groq_model,
    generate_rag_answer,
    generate_deep_explanation,
    generate_quiz,
)
from modules.quiz_engine import (
    init_quiz_session_state,
    start_quiz,
    reset_quiz,
    get_remaining_seconds,
    format_remaining_time,
    calculate_quiz_results,
    submit_quiz,
)
import streamlit.components.v1 as components


# ==============================================================================
# CUSTOM STYLING (MODERN SOLID BLACK UI / UX)
# ==============================================================================
st.markdown("""
<style>
    /* Global Solid Black Core */
    .stApp {
        background-color: #000000 !important;
        color: #f1f5f9 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Top Header Bar */
    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #171923 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #050608 !important;
        border-right: 1px solid #1a1e2e !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1a1e2e !important;
    }

    /* Modern Obsidian Header Banner */
    .main-header {
        background: linear-gradient(135deg, #090d16 0%, #030712 50%, #0d1322 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 1rem;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .main-header::after {
        content: "";
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #ffffff 20%, #93c5fd 60%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem !important;
        margin: 0 !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #94a3b8 !important;
        margin: 0.6rem 0 0 0 !important;
        font-size: 1.05rem;
        font-weight: 400;
    }
    .header-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* Modern Dashboard Stat Cards */
    .stat-card {
        background: #08090f;
        border: 1px solid #1b2133;
        border-radius: 0.85rem;
        padding: 1.25rem 1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
        box-shadow: 0 6px 24px rgba(59, 130, 246, 0.2);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.35rem;
        font-weight: 600;
    }

    /* Tabs Navigation Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #07080e;
        padding: 6px 8px;
        border-radius: 0.75rem;
        border: 1px solid #1a2035;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        color: #94a3b8 !important;
        font-weight: 600;
        padding: 8px 18px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f1f5f9 !important;
        background: rgba(255, 255, 255, 0.04);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%) !important;
        color: #60a5fa !important;
        border: 1px solid rgba(59, 130, 246, 0.45) !important;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.2);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 0.6rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: 1px solid #3b82f6 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
        transform: translateY(-1px);
    }
    .stButton > button[kind="secondary"] {
        background-color: #0c0e17 !important;
        border: 1px solid #1f273d !important;
        color: #cbd5e1 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* Cards and Expanders */
    .streamlit-expanderHeader {
        background-color: #080910 !important;
        border: 1px solid #1a2035 !important;
        border-radius: 0.6rem !important;
        color: #f1f5f9 !important;
    }
    .streamlit-expanderContent {
        background-color: #05060b !important;
        border: 1px solid #1a2035 !important;
        border-top: none !important;
        border-radius: 0 0 0.6rem 0.6rem !important;
    }

    /* Sources citation block */
    .source-box {
        background-color: #070912;
        border-left: 3px solid #3b82f6;
        border-top: 1px solid #141b2e;
        border-right: 1px solid #141b2e;
        border-bottom: 1px solid #141b2e;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.85rem;
        border-radius: 0 0.5rem 0.5rem 0;
    }

    /* Result Dashboard Card */
    .quiz-result-card {
        background: linear-gradient(135deg, #090e1a 0%, #04060c 100%);
        border: 2px solid #3b82f6;
        border-radius: 1rem;
        padding: 2.2rem 1.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 35px rgba(59, 130, 246, 0.25);
    }

    /* Custom Form & Input styles */
    .stTextInput input, .stSelectbox select, div[data-baseweb="select"] > div {
        background-color: #080a12 !important;
        border-color: #1e263d !important;
        color: #f1f5f9 !important;
    }
    .stTextInput input:focus, div[data-baseweb="select"] > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* Chat message container styling */
    [data-testid="stChatMessage"] {
        background-color: #070911 !important;
        border: 1px solid #151a2b !important;
        border-radius: 0.75rem !important;
        margin-bottom: 1rem !important;
        padding: 1rem 1.25rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #0b0f1e !important;
        border-color: #1d2745 !important;
    }

    /* Custom horizontal dividers */
    hr {
        border-color: #171c2d !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "selected_subject_id" not in st.session_state or st.session_state.selected_subject_id is None:
    st.session_state.selected_subject_id = get_last_active_subject()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of {"role": "...", "content": "...", "sources": [...]}

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "deep_explanation" not in st.session_state:
    st.session_state.deep_explanation = None

# Initialize quiz state
init_quiz_session_state()


# Pre-warm embedding model once in cache
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def load_cached_embedding_model():
    return get_embedding_model()

try:
    load_cached_embedding_model()
except Exception as e:
    st.sidebar.error(f"Warning: Embedding model initialization error: {e}")


# ==============================================================================
# SIDEBAR: SUBJECT SELECTION & MANAGEMENT
# ==============================================================================
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
    <span style="font-size: 1.3rem; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
        📚 Knowledge Hub
    </span>
    <span style="font-size: 0.8rem; color: #64748b;">Autonomous Course Agent</span>
</div>
""", unsafe_allow_html=True)

# API Key check
api_key = get_groq_api_key()
if api_key:
    st.sidebar.success("🔑 Groq API Key: Active", icon="✅")
else:
    st.sidebar.warning(
        "⚠️ Groq API Key missing!\nAdd `GROQ_API_KEY` to `.streamlit/secrets.toml` or OS environment.",
        icon="⚠️"
    )

# List all subjects
subjects = list_subjects()
subject_dict = {s["subject_id"]: s for s in subjects}

# Subject selection dropdown
if subjects:
    subject_names = {s["subject_id"]: f"{s['display_name']} ({s['num_documents']} docs)" for s in subjects}
    
    # Keep current selection valid
    current_id = st.session_state.selected_subject_id
    if current_id not in subject_names:
        current_id = subjects[0]["subject_id"]
        st.session_state.selected_subject_id = current_id
        save_last_active_subject(current_id)

    selected_sub_id = st.sidebar.selectbox(
        "Current Academic Subject:",
        options=list(subject_names.keys()),
        format_func=lambda x: subject_names[x],
        index=list(subject_names.keys()).index(current_id) if current_id in subject_names else 0,
        key="subject_select_box"
    )

    if selected_sub_id != st.session_state.selected_subject_id:
        st.session_state.selected_subject_id = selected_sub_id
        save_last_active_subject(selected_sub_id)
        # Clear chat when switching subjects to maintain subject boundary purity
        st.session_state.chat_history = []
        st.session_state.last_query = ""
        st.session_state.last_answer = ""
        st.session_state.last_context = ""
        st.session_state.deep_explanation = None
        st.rerun()
    else:
        save_last_active_subject(selected_sub_id)

    active_sub = subject_dict.get(st.session_state.selected_subject_id)
    if active_sub:
        st.sidebar.markdown(f"**Selected Subject:** `{active_sub['display_name']}`")
        st.sidebar.markdown(f"📄 **Documents:** {active_sub.get('num_documents', 0)}")
        st.sidebar.markdown(f"🧩 **Chunks:** {active_sub.get('num_chunks', 0)}")
        index_status = "✅ Built" if active_sub.get("has_index") else "⚠️ Empty / Not Built"
        st.sidebar.markdown(f"🔍 **FAISS Index:** {index_status}")
else:
    st.sidebar.info("No subjects created yet. Create your first subject below!")
    st.session_state.selected_subject_id = None
    save_last_active_subject("")

st.sidebar.divider()

# Create New Subject Form
with st.sidebar.expander("➕ Create New Subject", expanded=(len(subjects) == 0)):
    with st.form("create_subject_form", clear_on_submit=True):
        new_subject_name = st.text_input("Subject Name", placeholder="e.g. Data Structures")
        submitted = st.form_submit_button("Create Subject", use_container_width=True)
        if submitted:
            if not new_subject_name.strip():
                st.error("Please enter a subject name.")
            else:
                try:
                    new_id = create_subject(new_subject_name)
                    st.session_state.selected_subject_id = new_id
                    save_last_active_subject(new_id)
                    st.success(f"Subject '{new_subject_name}' created!")
                    st.rerun()
                except ValueError as ve:
                    st.error(str(ve))

# Delete Subject Form
if st.session_state.selected_subject_id:
    active_sub = subject_dict.get(st.session_state.selected_subject_id)
    with st.sidebar.expander("🗑️ Delete Current Subject"):
        if active_sub:
            st.warning(f"Delete **{active_sub['display_name']}** and all its documents and vectors?")
            if st.button("Confirm Delete Subject", type="primary", use_container_width=True):
                delete_subject(active_sub["subject_id"])
                st.session_state.selected_subject_id = None
                save_last_active_subject("")
                st.session_state.chat_history = []
                reset_quiz()
                st.success("Subject deleted successfully.")
                st.rerun()


# ==============================================================================
# MAIN PAGE HEADER & STATS
# ==============================================================================
st.markdown("""
<div class="main-header">
    <div class="header-badge">⚡ Autonomous Academic Intelligence</div>
    <h1>🎓 AI University Learning Assistant</h1>
    <p>Your Course Material • Grounded Knowledge Base • 5-Minute Conceptual Mastery</p>
</div>
""", unsafe_allow_html=True)

# Overview Dashboard Metrics
global_stats = get_global_stats()
active_sub_name = "None"
if st.session_state.selected_subject_id:
    active_info = subject_dict.get(st.session_state.selected_subject_id)
    if active_info:
        active_sub_name = active_info["display_name"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{global_stats['total_subjects']}</div>
        <div class="stat-label">Subjects</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{global_stats['total_documents']}</div>
        <div class="stat-label">Documents</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{global_stats['total_chunks']}</div>
        <div class="stat-label">Chunks</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number" style="font-size:1.15rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#38bdf8;">{active_sub_name}</div>
        <div class="stat-label">Active Subject</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Navigation Tabs
tab_kb, tab_chat, tab_quiz, tab_read = st.tabs([
    "📚 Knowledge Base & Upload",
    "🤖 AI Study Assistant",
    "📝 Conceptual Quiz (10 MCQs)",
    "📖 Read Documents"
])


# ==============================================================================
# TAB 1: KNOWLEDGE BASE & DOCUMENT PROCESSING
# ==============================================================================
with tab_kb:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please create or select an academic subject in the sidebar to begin.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]

        st.subheader(f"📘 Subject Knowledge Base: {sub_name}")
        st.caption("Upload lecture slides (PDF), notes (DOCX), or reading materials (TXT) to build this subject's private vector index.")

        # Document Upload Component
        uploaded_files = st.file_uploader(
            "Choose course documents (PDF, DOCX, TXT):",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key=f"uploader_{sub_id}"
        )

        if uploaded_files:
            if st.button("🚀 Process Documents into Knowledge Base", type="primary"):
                with st.status("Processing documents into subject knowledge base...", expanded=True) as status:
                    total_new_chunks = 0
                    processed_count = 0
                    skipped_count = 0

                    for up_file in uploaded_files:
                        filename = up_file.name
                        status.write(f"📄 Validating file: `{filename}`...")
                        file_bytes = up_file.read()

                        if len(file_bytes) == 0:
                            st.warning(f"File `{filename}` is empty. Skipping.")
                            continue

                        # Check duplicate hash
                        file_hash = compute_file_hash(file_bytes)
                        existing_doc = is_duplicate_document(sub_id, file_hash)
                        if existing_doc:
                            status.write(f"⚠️ `{filename}` is identical to already processed document `{existing_doc}`. Skipping.")
                            skipped_count += 1
                            continue

                        # Persist original file permanently to disk in its raw form
                        save_original_document(sub_id, filename, file_bytes)

                        # Extract text
                        status.write(f"🔍 Extracting structured text from `{filename}`...")
                        try:
                            extracted = extract_document(file_bytes, filename)
                        except Exception as e:
                            st.error(f"Error extracting `{filename}`: {str(e)}")
                            continue

                        if not extracted:
                            st.warning(f"No readable text could be extracted from `{filename}`.")
                            continue

                        # Chunk text
                        doc_id = f"doc_{int(time.time())}_{len(filename)}"
                        status.write(f"✂️ Creating overlapping chunks for `{filename}`...")
                        chunks = create_chunks(
                            extracted_documents=extracted,
                            subject_id=sub_id,
                            document_id=doc_id,
                            chunk_size=DEFAULT_CHUNK_SIZE,
                            overlap=DEFAULT_CHUNK_OVERLAP
                        )

                        if not chunks:
                            st.warning(f"No chunks created for `{filename}`.")
                            continue

                        # Persist chunk JSON and registry
                        save_document_chunks(
                            subject_id=sub_id,
                            document_id=doc_id,
                            filename=filename,
                            file_hash=file_hash,
                            chunks=chunks
                        )
                        total_new_chunks += len(chunks)
                        processed_count += 1
                        status.write(f"✓ `{filename}` processed into {len(chunks)} chunks.")

                    # Rebuild Subject FAISS Index
                    if processed_count > 0:
                        status.write("⚡ Generating embeddings & updating subject FAISS vector index...")
                        all_chunks = load_all_subject_chunks(sub_id)
                        build_and_save_index(sub_id, all_chunks)
                        status.write("💾 FAISS index persisted successfully.")

                    status.update(label="✅ Document processing complete!", state="complete", expanded=False)

                if processed_count > 0:
                    st.success(f"Successfully processed {processed_count} document(s) with {total_new_chunks} chunks.")
                    time.sleep(1)
                    st.rerun()
                elif skipped_count > 0 and processed_count == 0:
                    st.info("All uploaded files were already present in this subject.")

        st.divider()

        # Existing Documents Section
        st.subheader("📑 Preserved Documents in this Subject")
        registry = load_registry(sub_id)
        docs = registry.get("documents", {})

        if not docs:
            st.info("No documents uploaded yet for this subject.")
        else:
            for doc_id, meta in list(docs.items()):
                c_name, c_chunks, c_date, c_action = st.columns([3, 1, 2, 1])
                with c_name:
                    st.markdown(f"📄 **{meta.get('filename', 'Unknown')}**")
                with c_chunks:
                    st.markdown(f"`{meta.get('num_chunks', 0)} chunks`")
                with c_date:
                    ts = meta.get('processed_at', '')[:16].replace('T', ' ')
                    st.caption(f"Processed: {ts}")
                with c_action:
                    if st.button("Delete", key=f"del_{doc_id}", type="secondary"):
                        delete_document(sub_id, doc_id)
                        # Rebuild FAISS index with remaining chunks
                        remaining_chunks = load_all_subject_chunks(sub_id)
                        build_and_save_index(sub_id, remaining_chunks)
                        st.success(f"Deleted {meta.get('filename')}")
                        st.rerun()


# ==============================================================================
# TAB 2: AI STUDY ASSISTANT (SUBJECT-AWARE RAG CHAT)
# ==============================================================================
with tab_chat:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please create or select an academic subject in the sidebar to ask questions.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]

        st.subheader(f"🤖 AI Study Assistant — {sub_name}")
        st.caption("Ask questions about your course materials. The assistant retrieves knowledge exclusively from this subject's index.")

        # Check if subject has an index
        sub_stats = get_subject_stats(sub_id)
        if not sub_stats["has_index"] or sub_stats["num_chunks"] == 0:
            st.warning(
                f"The subject **{sub_name}** does not have any processed course documents yet. "
                "Please upload documents in the **Knowledge Base & Upload** tab first.",
                icon="⚠️"
            )

        # Display Chat History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 View Retrieved Sources"):
                        for idx, src in enumerate(msg["sources"], 1):
                            page_disp = src.get('page') if src.get('page') is not None else 'N/A'
                            st.markdown(f"**Source {idx}:** `{src.get('filename')}` — Page: `{page_disp}`")
                            st.caption(f"Scores: Hybrid: {src.get('hybrid_score', 0):.2f} | Semantic: {src.get('semantic_score', 0):.2f} | Keyword: {src.get('keyword_score', 0):.2f}")
                            st.text(src.get("text", "")[:350] + ("..." if len(src.get("text", "")) > 350 else ""))

        # Chat Input
        user_query = st.chat_input(f"Ask a question about {sub_name}...")

        if user_query:
            if not api_key:
                st.error("Cannot query AI: GROQ_API_KEY is not configured.")
            elif not sub_stats["has_index"] or sub_stats["num_chunks"] == 0:
                st.error("Please upload and process course documents before asking questions.")
            else:
                # Add user message to UI
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                # Execute RAG Pipeline
                with st.chat_message("assistant"):
                    with st.spinner(f"Searching {sub_name} knowledge base..."):
                        # 1. Hybrid Search (Semantic + Keyword)
                        retrieved_chunks, is_confident = hybrid_search(
                            query=user_query,
                            subject_id=sub_id,
                            top_k=DEFAULT_TOP_K,
                            semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
                            keyword_weight=DEFAULT_KEYWORD_WEIGHT,
                            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
                        )

                        # 2. Context Construction
                        context_str = build_context(retrieved_chunks)

                        # Save for deep explanation
                        st.session_state.last_query = user_query
                        st.session_state.last_context = context_str
                        st.session_state.deep_explanation = None

                        # 3. LLM Generation
                        try:
                            answer = generate_rag_answer(
                                question=user_query,
                                subject_name=sub_name,
                                retrieved_context=context_str,
                                is_confident=is_confident,
                                chat_history=st.session_state.chat_history[:-1]
                            )
                        except Exception as e:
                            answer = f"Error communicating with AI tutor: {str(e)}"

                        st.session_state.last_answer = answer

                        # Display Answer
                        st.markdown(answer)

                        # Display Sources
                        if retrieved_chunks and is_confident:
                            with st.expander("📚 Sources & References"):
                                for idx, src in enumerate(retrieved_chunks, 1):
                                    page_info = src.get('page') if src.get('page') is not None else 'N/A'
                                    st.markdown(f"**{idx}. {src.get('filename')}** — Page: `{page_info}`")
                                    st.caption(
                                        f"Relevance: Hybrid: {src.get('hybrid_score', 0):.2f} | "
                                        f"Semantic: {src.get('semantic_score', 0):.2f} | "
                                        f"Keyword: {src.get('keyword_score', 0):.2f}"
                                    )
                                    st.text(src.get("text", "")[:350] + ("..." if len(src.get("text", "")) > 350 else ""))

                        # Store in history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks if is_confident else []
                        })

        # Deep Explanation & Quiz Shortcuts if an answer was produced
        if st.session_state.last_answer and not st.session_state.last_answer.startswith("I couldn't find enough"):
            st.divider()
            c_exp, c_quiz, _ = st.columns([1.5, 1.5, 3])
            
            with c_exp:
                if st.button("🧠 Explain Deeper", use_container_width=True):
                    with st.spinner("Generating conceptual pedagogical breakdown..."):
                        try:
                            deep_exp = generate_deep_explanation(
                                question=st.session_state.last_query,
                                subject_name=sub_name,
                                retrieved_context=st.session_state.last_context,
                                initial_answer=st.session_state.last_answer
                            )
                            st.session_state.deep_explanation = deep_exp
                        except Exception as e:
                            st.error(f"Failed to generate deep explanation: {str(e)}")

            with c_quiz:
                if st.button("📝 Test My Knowledge on this Subject", use_container_width=True):
                    st.info("Head over to the **Conceptual Quiz (10 MCQs)** tab to start your 5-minute assessment!")

            # Display Deep Explanation if generated
            if st.session_state.deep_explanation:
                st.markdown("### 🧠 In-Depth Conceptual Explanation")
                st.markdown(st.session_state.deep_explanation)


# ==============================================================================
# TAB 3: TIMED CONCEPTUAL MCQ QUIZ (10 QUESTIONS | 10 MARKS | 5 MINUTES)
# ==============================================================================
with tab_quiz:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please create or select an academic subject in the sidebar to start a quiz.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]
        sub_stats = get_subject_stats(sub_id)

        st.subheader(f"📝 Timed Conceptual Assessment — {sub_name}")
        st.markdown("""
        <div style="background:#080a13; border: 1px solid #1c2338; border-radius: 0.75rem; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; flex-wrap: wrap; gap: 20px; align-items: center;">
                <div style="color: #60a5fa; font-weight: 700;">🎯 10 Real-World Conceptual MCQs</div>
                <div style="color: #34d399; font-weight: 700;">🏆 10 Marks (Score out of 10)</div>
                <div style="color: #f87171; font-weight: 700;">⏱ 5 Minutes Fixed Timer</div>
                <div style="color: #c084fc; font-weight: 700;">🧠 Scenario-Based Problem Solving</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not sub_stats["has_index"] or sub_stats["num_chunks"] == 0:
            st.warning("Please upload course documents before generating a quiz.", icon="⚠️")
        else:
            # Quiz is NOT currently active and not yet submitted
            if not st.session_state.quiz_active and not st.session_state.quiz_submitted:
                if st.button("🚀 Start 5-Minute Quiz", type="primary"):
                    if not api_key:
                        st.error("Cannot generate quiz: GROQ_API_KEY is not configured.")
                    else:
                        with st.spinner(f"Synthesizing 10 real-world conceptual MCQs from {sub_name} materials..."):
                            # Collect sample representative chunks from the subject for quiz context
                            all_chunks = load_all_subject_chunks(sub_id)
                            # Take up to 15 chunks to fit comfortably within prompt
                            sample_text = "\n\n".join([f"Topic excerpt from {c['filename']}:\n{c['text']}" for c in all_chunks[:15]])
                            
                            try:
                                questions = generate_quiz(subject_name=sub_name, subject_context=sample_text)
                                start_quiz(sub_id, sub_name, questions)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating quiz: {str(e)}")

            # Quiz IS currently active
            elif st.session_state.quiz_active and not st.session_state.quiz_submitted:
                # Timer evaluation
                remaining_sec = get_remaining_seconds()

                if remaining_sec <= 0:
                    st.warning("⏰ Time is up! Automatically submitting your answers.")
                    submit_quiz()
                    st.rerun()
                else:
                    col_time, col_reset = st.columns([3, 1])
                    with col_time:
                        timer_component = f"""
                        <div style="background: linear-gradient(135deg, #180808 0%, #0d0404 100%);
                                    border: 2px solid #ef4444; color: #f87171; padding: 12px 18px;
                                    border-radius: 10px; font-size: 22px; font-weight: 800; text-align: center;
                                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                                    box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);">
                            ⏱ <span id="countdown">{format_remaining_time(remaining_sec)}</span> remaining
                        </div>
                        <script>
                            var secondsLeft = {remaining_sec};
                            var timerDisplay = document.getElementById("countdown");
                            function updateTimer() {{
                                if (secondsLeft <= 0) {{
                                    timerDisplay.innerText = "00:00 (TIME UP!)";
                                    timerDisplay.style.color = "#dc2626";
                                    try {{
                                        var btns = window.parent.document.querySelectorAll('button');
                                        for (var i = 0; i < btns.length; i++) {{
                                            if (btns[i].innerText && btns[i].innerText.includes("Submit Quiz")) {{
                                                btns[i].click();
                                                break;
                                            }}
                                        }}
                                    }} catch(e) {{}}
                                    return;
                                }}
                                var m = Math.floor(secondsLeft / 60);
                                var s = secondsLeft % 60;
                                timerDisplay.innerText = (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
                                secondsLeft--;
                            }}
                            updateTimer();
                            setInterval(updateTimer, 1000);
                        </script>
                        """
                        components.html(timer_component, height=65)
                    with col_reset:
                        if st.button("Cancel Quiz", use_container_width=True):
                            reset_quiz()
                            st.rerun()

                    # Render questions
                    questions = st.session_state.quiz_questions
                    
                    with st.form("quiz_form"):
                        for idx, q in enumerate(questions):
                            st.markdown(f"**Question {idx+1} of 10**")
                            st.markdown(f"**{q['question']}**")

                            opts = q["options"]
                            # Format radio choices
                            choice_labels = [
                                f"A) {opts.get('A', '')}",
                                f"B) {opts.get('B', '')}",
                                f"C) {opts.get('C', '')}",
                                f"D) {opts.get('D', '')}"
                            ]
                            
                            prev_choice = st.session_state.quiz_user_answers.get(idx)
                            prev_idx = None
                            if prev_choice == "A": prev_idx = 0
                            elif prev_choice == "B": prev_idx = 1
                            elif prev_choice == "C": prev_idx = 2
                            elif prev_choice == "D": prev_idx = 3

                            selected_label = st.radio(
                                f"Select your answer for Question {idx+1}:",
                                options=choice_labels,
                                index=prev_idx,
                                key=f"radio_q_{idx}",
                                label_visibility="collapsed"
                            )

                            if selected_label:
                                letter = selected_label[0]  # 'A', 'B', 'C', or 'D'
                                st.session_state.quiz_user_answers[idx] = letter

                            st.write("")

                        submitted = st.form_submit_button("🏁 Submit Quiz", type="primary", use_container_width=True)
                        if submitted:
                            submit_quiz()
                            st.rerun()

            # Quiz HAS been submitted: Show Results & Detailed Feedback
            elif st.session_state.quiz_submitted:
                score, total, topics, detailed = calculate_quiz_results()
                pct = int((score / total) * 100) if total > 0 else 0

                perf_label = "Needs Improvement"
                perf_color = "#f87171"
                if score >= 9:
                    perf_label = "🌟 Excellent Mastery"
                    perf_color = "#34d399"
                elif score >= 7:
                    perf_label = "👍 Good Understanding"
                    perf_color = "#60a5fa"
                elif score >= 5:
                    perf_label = "📖 Average — Review Recommended"
                    perf_color = "#fbbf24"

                st.markdown(f"""
                <div class="quiz-result-card">
                    <span style="text-transform: uppercase; letter-spacing: 0.1em; color: #94a3b8; font-size: 0.85rem; font-weight: 700;">Quiz Performance Assessment</span>
                    <h2 style="margin: 0.5rem 0; color: #ffffff; font-size: 1.6rem;">{sub_name}</h2>
                    <h1 style="font-size: 3.5rem; margin: 0.5rem 0; font-weight: 900; background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{score} / {total}</h1>
                    <p style="font-size: 1.15rem; color: #cbd5e1; margin: 0;">Performance: <strong style="color: {perf_color};">{perf_label}</strong> ({pct}%)</p>
                </div>
                """, unsafe_allow_html=True)

                # Recommended Review Topics
                if topics:
                    st.subheader("🎯 Recommended Review Topics")
                    st.write("Focus on revising the following concepts from your course documents:")
                    for t in topics:
                        st.markdown(f"* 🔍 **{t}**")
                    st.write("")
                else:
                    st.success("🎉 Perfect score! You demonstrated mastery of all tested concepts.")

                st.subheader("📋 Detailed Question Breakdown & Solutions")
                for item in detailed:
                    status_icon = "✅" if item["is_correct"] else "❌"
                    status_text = "Correct (+1 Mark)" if item["is_correct"] else "Incorrect (0 Marks)"
                    with st.expander(f"{status_icon} Question {item['index']}: {status_text} — Topic: {item['topic']}", expanded=(not item["is_correct"])):
                        st.markdown(f"#### Question {item['index']}: {item['question']}")
                        st.write("**Answer Choices:**")
                        for opt_key in ["A", "B", "C", "D"]:
                            opt_text = item["options"].get(opt_key, "")
                            is_right = (opt_key == item["correct_choice"])
                            is_chosen = (opt_key == item["user_choice"])
                            tag = ""
                            if is_chosen and is_right:
                                tag = " 🟢 **(Your Answer — Correct!)**"
                            elif is_chosen and not is_right:
                                tag = " 🔴 **(Your Answer)**"
                            elif is_right:
                                tag = " 🟢 **(Correct Option)**"
                            st.markdown(f"* **{opt_key})** {opt_text}{tag}")

                        st.write("")
                        c_user, c_right = st.columns(2)
                        with c_user:
                            st.markdown(f"**Your Selection:** `{item['user_choice'] or 'None Selected'}`")
                        with c_right:
                            st.markdown(f"**Correct Option:** `{item['correct_choice']}`")
                        st.info(f"💡 **Deep Conceptual Explanation:**\n\n{item['explanation']}")

                st.divider()
                if st.button("🔄 Take Another Quiz", type="primary"):
                    reset_quiz()
                    st.rerun()


# ==============================================================================
# TAB 4: READ DOCUMENTS (ORIGINAL FORM VIEWER & PERMANENT MEMORY)
# ==============================================================================
with tab_read:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please select or create an academic subject in the sidebar to read its documents.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]

        st.subheader(f"📖 Read Documents — {sub_name}")
        st.caption("Read and study course materials in their original layout (PDF, DOCX, TXT). Uploaded materials and chunks persist across page refreshes until explicitly deleted.")

        registry = load_registry(sub_id)
        docs = registry.get("documents", {})

        if not docs:
            st.info(f"No documents uploaded yet for **{sub_name}**. Please go to the **'📚 Knowledge Base & Upload'** tab to upload course material.")
        else:
            doc_options = {doc_id: f"{meta.get('filename', doc_id)} ({meta.get('num_chunks', 0)} chunks)" for doc_id, meta in docs.items()}
            
            selected_doc_id = st.selectbox(
                "Select a document to read:",
                options=list(doc_options.keys()),
                format_func=lambda x: doc_options[x],
                key="read_doc_selector"
            )

            selected_meta = docs.get(selected_doc_id, {})
            filename = selected_meta.get("filename", "")
            ext = os.path.splitext(filename)[1].lower()

            # Metadata Bar
            raw_bytes = load_original_document(sub_id, filename)
            size_str = "N/A"
            if raw_bytes:
                size_kb = len(raw_bytes) / 1024
                if size_kb >= 1024:
                    size_str = f"{size_kb / 1024:.2f} MB"
                else:
                    size_str = f"{size_kb:.1f} KB"

            ts = selected_meta.get('processed_at', '')[:16].replace('T', ' ')

            c_info1, c_info2, c_info3, c_del = st.columns([3, 2, 2, 2])
            with c_info1:
                st.markdown(f"📄 **File:** `{filename}`")
            with c_info2:
                st.markdown(f"🧩 **Chunks:** `{selected_meta.get('num_chunks', 0)}`")
            with c_info3:
                st.markdown(f"💾 **Size:** `{size_str}`")
            with c_del:
                if st.button("🗑️ Delete Permanently", key=f"perm_del_{selected_doc_id}", type="secondary", use_container_width=True):
                    delete_document(sub_id, selected_doc_id)
                    rem_chunks = load_all_subject_chunks(sub_id)
                    build_and_save_index(sub_id, rem_chunks)
                    st.success(f"Permanently deleted `{filename}` and updated vector index.")
                    st.rerun()

            st.divider()

            # --- 1. PDF VIEWER (ORIGINAL FORM) ---
            if ext == ".pdf":
                if raw_bytes:
                    c_btn, c_mode = st.columns([1, 2])
                    with c_btn:
                        st.download_button(
                            label="📥 Download Original PDF",
                            data=raw_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    with c_mode:
                        read_mode = st.radio(
                            "Display Mode:",
                            options=["🖥️ Visual PDF Viewer", "📖 Formatted Text Reader"],
                            horizontal=True,
                            key=f"pdf_mode_{selected_doc_id}",
                            label_visibility="collapsed"
                        )
                    st.write("")

                    if read_mode == "🖥️ Visual PDF Viewer":
                        b64_pdf = base64.b64encode(raw_bytes).decode("utf-8")
                        pdf_js_html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                          <meta charset="utf-8">
                          <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
                          <style>
                            * {{ box-sizing: border-box; }}
                            body {{
                              margin: 0; padding: 0;
                              background: #000000;
                              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                              display: flex; flex-direction: column;
                              height: 100vh; overflow: hidden;
                            }}
                            #toolbar {{
                              background: #090b12;
                              color: #f8fafc;
                              display: flex; align-items: center; justify-content: center;
                              gap: 12px; padding: 10px 16px;
                              border-bottom: 1px solid #1a2035;
                              flex-shrink: 0;
                              flex-wrap: wrap;
                            }}
                            .t-btn {{
                              background: #2563eb; border: 1px solid #3b82f6; color: white;
                              padding: 6px 14px; border-radius: 6px; cursor: pointer;
                              font-size: 13px; font-weight: 600;
                              transition: all 0.15s ease;
                            }}
                            .t-btn:hover {{ background: #1d4ed8; }}
                            .t-btn:disabled {{ background: #1e2438; border-color: #2a334d; cursor: not-allowed; opacity: 0.5; }}
                            .badge {{ font-size: 14px; font-weight: 600; color: #cbd5e1; }}
                            #viewer-container {{
                              flex: 1; overflow: auto;
                              display: flex; justify-content: center; align-items: flex-start;
                              padding: 24px;
                              background: #040508;
                            }}
                            #pdf-canvas {{
                              box-shadow: 0 10px 35px rgba(0,0,0,0.8), 0 0 0 1px rgba(255,255,255,0.08);
                              border-radius: 6px;
                              background: white;
                              max-width: 100%;
                            }}
                          </style>
                        </head>
                        <body>
                          <div id="toolbar">
                            <button class="t-btn" id="prev-page">◀ Prev</button>
                            <span class="badge">Page <span id="page-num" style="color:#60a5fa;">1</span> of <span id="page-count">-</span></span>
                            <button class="t-btn" id="next-page">Next ▶</button>
                            <span style="border-left: 1px solid #1e263d; height: 18px; margin: 0 6px;"></span>
                            <button class="t-btn" id="zoom-out" style="background:#131828; border-color:#242e4c;">🔍 -</button>
                            <span id="zoom-pct" class="badge">100%</span>
                            <button class="t-btn" id="zoom-in" style="background:#131828; border-color:#242e4c;">🔍 +</button>
                            <button class="t-btn" id="fit-page" style="background:#0284c7; border-color:#38bdf8;">Fit Width</button>
                          </div>
                          <div id="viewer-container">
                            <canvas id="pdf-canvas"></canvas>
                          </div>

                          <script>
                            pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

                            var pdfData = atob("{b64_pdf}");
                            var uint8Array = new Uint8Array(pdfData.length);
                            for (var i = 0; i < pdfData.length; i++) {{
                              uint8Array[i] = pdfData.charCodeAt(i);
                            }}

                            var pdfDoc = null,
                                pageNum = 1,
                                pageRendering = false,
                                pageNumPending = null,
                                scale = 1.25,
                                canvas = document.getElementById('pdf-canvas'),
                                ctx = canvas.getContext('2d');

                            function renderPage(num) {{
                              pageRendering = true;
                              pdfDoc.getPage(num).then(function(page) {{
                                var viewport = page.getViewport({{ scale: scale }});
                                canvas.height = viewport.height;
                                canvas.width = viewport.width;

                                var renderContext = {{
                                  canvasContext: ctx,
                                  viewport: viewport
                                }};
                                var renderTask = page.render(renderContext);

                                renderTask.promise.then(function() {{
                                  pageRendering = false;
                                  if (pageNumPending !== null) {{
                                    renderPage(pageNumPending);
                                    pageNumPending = null;
                                  }}
                                }});
                              }});

                              document.getElementById('page-num').textContent = num;
                              document.getElementById('prev-page').disabled = (num <= 1);
                              document.getElementById('next-page').disabled = (num >= pdfDoc.numPages);
                              document.getElementById('zoom-pct').textContent = Math.round(scale * 100) + '%';
                            }}

                            function queueRenderPage(num) {{
                              if (pageRendering) {{
                                pageNumPending = num;
                              }} else {{
                                renderPage(num);
                              }}
                            }}

                            document.getElementById('prev-page').onclick = function() {{
                              if (pageNum <= 1) return;
                              pageNum--;
                              queueRenderPage(pageNum);
                            }};

                            document.getElementById('next-page').onclick = function() {{
                              if (pageNum >= pdfDoc.numPages) return;
                              pageNum++;
                              queueRenderPage(pageNum);
                            }};

                            document.getElementById('zoom-in').onclick = function() {{
                              scale = Math.min(3.0, scale + 0.2);
                              queueRenderPage(pageNum);
                            }};

                            document.getElementById('zoom-out').onclick = function() {{
                              scale = Math.max(0.5, scale - 0.2);
                              queueRenderPage(pageNum);
                            }};

                            document.getElementById('fit-page').onclick = function() {{
                              var container = document.getElementById('viewer-container');
                              var availableWidth = container.clientWidth - 50;
                              pdfDoc.getPage(pageNum).then(function(page) {{
                                var unscaledViewport = page.getViewport({{ scale: 1.0 }});
                                scale = Math.max(0.5, availableWidth / unscaledViewport.width);
                                queueRenderPage(pageNum);
                              }});
                            }};

                            pdfjsLib.getDocument({{ data: uint8Array }}).promise.then(function(doc) {{
                              pdfDoc = doc;
                              document.getElementById('page-count').textContent = doc.numPages;
                              renderPage(pageNum);
                            }}).catch(function(err) {{
                              console.error("PDF loading error:", err);
                              document.getElementById('viewer-container').innerHTML = 
                                '<div style="color:#f87171;text-align:center;padding:40px;"><h3>Unable to render PDF preview</h3><p>' + err.message + '</p></div>';
                            }});
                          </script>
                        </body>
                        </html>
                        """
                        components.html(pdf_js_html, height=850)

                    else:
                        # Formatted Text Reader in Elegant Dark Academic Mode
                        chunks = load_document_chunks(sub_id, selected_doc_id)
                        if chunks:
                            current_page = None
                            page_text_acc = []
                            for c in chunks:
                                p = c.get('page')
                                if p != current_page and page_text_acc:
                                    p_label = f"Page {current_page}" if current_page else "General Content"
                                    st.markdown(f"#### 📄 {p_label}")
                                    st.markdown(f"""
                                    <div style="background-color: #080a12; border: 1px solid #1a2035; border-radius: 8px; padding: 1.6rem; font-family: 'Georgia', serif; font-size: 1.05rem; line-height: 1.85; color: #e2e8f0; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                                        {('<br><br>'.join(page_text_acc)).replace(chr(10), '<br>')}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    page_text_acc = []
                                current_page = p
                                page_text_acc.append(c.get('text', ''))
                            if page_text_acc:
                                p_label = f"Page {current_page}" if current_page else "General Content"
                                st.markdown(f"#### 📄 {p_label}")
                                st.markdown(f"""
                                <div style="background-color: #080a12; border: 1px solid #1a2035; border-radius: 8px; padding: 1.6rem; font-family: 'Georgia', serif; font-size: 1.05rem; line-height: 1.85; color: #e2e8f0; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                                    {('<br><br>'.join(page_text_acc)).replace(chr(10), '<br>')}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("No chunk text available for this document.")

                else:
                    st.warning("Original raw PDF is not on disk. Displaying preserved chunk text:")
                    chunks = load_document_chunks(sub_id, selected_doc_id)
                    for c in chunks:
                        st.markdown(f"**Chunk {c.get('chunk_index', 0)+1}:**")
                        st.write(c.get('text', ''))

            # --- 2. DOCX VIEWER (ORIGINAL FORM) ---
            elif ext in [".docx", ".doc"]:
                if raw_bytes:
                    c_btn, _ = st.columns([2, 5])
                    with c_btn:
                        st.download_button(
                            label="📥 Download Original DOCX",
                            data=raw_bytes,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                    st.write("")

                    # Extract full original document text
                    try:
                        extracted = extract_document(raw_bytes, filename)
                        full_docx_text = "\n\n".join([page_info["text"] for page_info in extracted])
                        st.markdown("""
                        <div style="background-color: #080a12; border: 1px solid #1a2035; border-radius: 8px; padding: 2rem; max-height: 800px; overflow-y: auto; font-family: 'Georgia', serif; font-size: 1.05rem; line-height: 1.85; color: #e2e8f0; box-shadow: 0 6px 20px rgba(0,0,0,0.6);">
                        """ + full_docx_text.replace('\n', '<br>') + """
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error reading DOCX: {e}")
                else:
                    st.warning("Original raw file is not on disk. Displaying preserved chunk text:")
                    chunks = load_document_chunks(sub_id, selected_doc_id)
                    for c in chunks:
                        st.write(c.get('text', ''))

            # --- 3. TXT VIEWER (ORIGINAL FORM) ---
            elif ext == ".txt":
                if raw_bytes:
                    c_btn, _ = st.columns([2, 5])
                    with c_btn:
                        st.download_button(
                            label="📥 Download Original TXT",
                            data=raw_bytes,
                            file_name=filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                    st.write("")
                    try:
                        txt_content = raw_bytes.decode('utf-8', errors='replace')
                    except Exception:
                        txt_content = str(raw_bytes)
                    st.text_area("Original Document Text", value=txt_content, height=700, disabled=True)
                else:
                    st.warning("Original raw file is not on disk. Displaying preserved chunk text:")
                    chunks = load_document_chunks(sub_id, selected_doc_id)
                    full_txt = "\n\n".join([c.get('text', '') for c in chunks])
                    st.text_area("Preserved Document Text", value=full_txt, height=700, disabled=True)

            else:
                if raw_bytes:
                    st.download_button("📥 Download File", data=raw_bytes, file_name=filename)
                chunks = load_document_chunks(sub_id, selected_doc_id)
                for c in chunks:
                    st.write(c.get('text', ''))
