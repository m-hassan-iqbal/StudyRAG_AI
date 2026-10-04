"""
Subject-Aware AI University Learning Assistant
Main Streamlit Application.
Provides subject management, document processing, persistent FAISS indexing,
hybrid retrieval RAG chat, deep conceptual explanations, a 10-question timed quiz,
and original document reader.
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
# CUSTOM STYLING
# ==============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: #dbeafe;
        margin: 0.25rem 0 0 0;
        font-size: 1rem;
    }
    .stat-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-number {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1e40af;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .source-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        border-radius: 0 0.375rem 0.375rem 0;
    }
    .quiz-timer-box {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
        color: #991b1b;
        padding: 0.75rem;
        border-radius: 0.5rem;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
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
st.sidebar.title("📚 Subject Management")

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
    <h1>🎓 AI University Learning Assistant</h1>
    <p>Your Course Material. Your Knowledge Base. Your AI Tutor.</p>
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
        <div class="stat-number" style="font-size:1.1rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{active_sub_name}</div>
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

        st.subheader(f"📘 Subject: {sub_name}")
        st.caption("Upload lecture slides (PDF), notes (DOCX), or reading materials (TXT) to build this subject's private knowledge base.")

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
        st.subheader("📑 Documents in this Subject")
        registry = load_registry(sub_id)
        docs = registry.get("documents", {})

        if not docs:
            st.info("No documents uploaded yet for this subject.")
        else:
            for doc_id, meta in list(docs.items()):
                c_name, c_chunks, c_date, c_action = st.columns([3, 1, 2, 1])
                with c_name:
                    st.markdown(f"**{meta.get('filename', 'Unknown')}**")
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
        st.caption("Ask questions about your uploaded materials. The assistant retrieves knowledge ONLY from this subject.")

        # Check if subject has an index
        all_chunks = load_all_subject_chunks(sub_id)
        if not all_chunks:
            st.warning(f"⚠️ No documents have been processed for '{sub_name}' yet.\nPlease upload course documents in the **Knowledge Base** tab first.")
        else:
            # Display conversation history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources"):
                        with st.expander("📚 Retrieved Source References"):
                            for s in msg["sources"]:
                                page_str = f", Page {s['page']}" if s.get('page') else ""
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Document:</strong> {s['filename']}{page_str}<br>
                                    <small><strong>Score:</strong> {s['score']:.3f} | <strong>Match:</strong> {s['match_type'].capitalize()}</small><br>
                                    <em>"{s['text_snippet']}"</em>
                                </div>
                                """, unsafe_allow_html=True)

            # Chat input
            user_question = st.chat_input(f"Ask a question about {sub_name}...")

            if user_question:
                # Add user query to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner(f"Searching {sub_name} knowledge base..."):
                        # Execute hybrid search
                        retrieved = hybrid_search(
                            subject_id=sub_id,
                            query=user_question,
                            top_k=DEFAULT_TOP_K,
                            semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
                            keyword_weight=DEFAULT_KEYWORD_WEIGHT
                        )

                        # Build formatted context
                        context_text, sources = build_context(retrieved)

                        # Save for deep explanation
                        st.session_state.last_query = user_question
                        st.session_state.last_context = context_text

                        # Generate grounded LLM response
                        answer = generate_rag_answer(
                            query=user_question,
                            context=context_text,
                            subject_name=sub_name
                        )

                        st.session_state.last_answer = answer
                        st.session_state.deep_explanation = None

                        st.markdown(answer)

                        if sources:
                            with st.expander("📚 Retrieved Source References"):
                                for s in sources:
                                    page_str = f", Page {s['page']}" if s.get('page') else ""
                                    st.markdown(f"""
                                    <div class="source-box">
                                        <strong>Document:</strong> {s['filename']}{page_str}<br>
                                        <small><strong>Score:</strong> {s['score']:.3f} | <strong>Match:</strong> {s['match_type'].capitalize()}</small><br>
                                        <em>"{s['text_snippet']}"</em>
                                    </div>
                                    """, unsafe_allow_html=True)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })

            # Deep Conceptual Explanation Feature
            if st.session_state.last_answer and st.session_state.last_query:
                st.write("")
                st.divider()
                col_exp1, col_exp2 = st.columns([3, 1])
                with col_exp1:
                    st.markdown(f"**Want a deeper breakdown of:** *\"{st.session_state.last_query}\"*?")
                    st.caption("Expands on underlying principles, step-by-step intuition, mental models, and academic examples.")
                with col_exp2:
                    if st.button("🧠 Request Deep Explanation", type="secondary"):
                        with st.spinner("Synthesizing deep conceptual tutorial..."):
                            deep_exp = generate_deep_explanation(
                                query=st.session_state.last_query,
                                previous_answer=st.session_state.last_answer,
                                context=st.session_state.last_context,
                                subject_name=sub_name
                            )
                            st.session_state.deep_explanation = deep_exp

                if st.session_state.deep_explanation:
                    st.info(f"### 💡 In-Depth Conceptual Explanation\n\n{st.session_state.deep_explanation}")


# ==============================================================================
# TAB 3: CONCEPTUAL MCQ QUIZ (10 QUESTIONS WITH LIVE TIMER & FEEDBACK)
# ==============================================================================
with tab_quiz:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please create or select an academic subject in the sidebar to generate a quiz.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]

        st.subheader(f"📝 Timed Conceptual Quiz — {sub_name}")
        st.caption("10 challenging conceptual MCQs grounded strictly in your uploaded course materials.")

        all_chunks = load_all_subject_chunks(sub_id)
        if not all_chunks:
            st.warning(f"⚠️ No documents have been processed for '{sub_name}' yet.\nPlease upload lecture materials first to generate a quiz.")
        else:
            # Quiz is NOT active and NOT submitted: Show Quiz Generation Screen
            if not st.session_state.quiz_active and not st.session_state.quiz_submitted:
                st.markdown(f"""
                ### Quiz Details:
                * **Subject:** `{sub_name}`
                * **Question Count:** `10 Multiple Choice Questions (MCQs)`
                * **Time Limit:** `5 Minutes` (300 seconds)
                * **Format:** Real-world problem scenarios, algorithmic trade-offs, and conceptual mastery.
                """)

                if st.button("🚀 Generate & Start 10-Question Quiz", type="primary"):
                    with st.spinner("Reading course documents and crafting 10 deep conceptual questions..."):
                        quiz_data = generate_quiz(
                            chunks=all_chunks,
                            subject_name=sub_name,
                            num_questions=QUIZ_QUESTION_COUNT
                        )

                        if quiz_data and "questions" in quiz_data and len(quiz_data["questions"]) > 0:
                            start_quiz(quiz_data["questions"])
                            st.rerun()
                        else:
                            st.error("Failed to generate quiz questions. Please verify your Groq API key and course documents.")

            # Quiz IS ACTIVE: Show Timer & Question Form
            elif st.session_state.quiz_active and not st.session_state.quiz_submitted:
                rem_sec = get_remaining_seconds()

                # JavaScript Live Countdown Timer
                timer_html = f"""
                <div style="background-color: #fee2e2; border: 2px solid #ef4444; border-radius: 8px; padding: 12px; text-align: center; margin-bottom: 12px; font-family: monospace;">
                    <span style="font-size: 16px; font-weight: bold; color: #991b1b;">⏱️ REMAINING TIME: </span>
                    <span id="countdown_clock" style="font-size: 24px; font-weight: 800; color: #dc2626;">{format_remaining_time(rem_sec)}</span>
                </div>
                <script>
                    var secondsLeft = {rem_sec};
                    var clockEl = document.getElementById("countdown_clock");
                    var countdownInterval = setInterval(function() {{
                        secondsLeft--;
                        if (secondsLeft <= 0) {{
                            clearInterval(countdownInterval);
                            clockEl.innerText = "00:00 (TIME EXPIRED)";
                            clockEl.style.color = "#7f1d1d";
                            window.parent.postMessage({{type: "streamlit:setComponentValue", value: "timeout"}}, "*");
                        }} else {{
                            var mins = Math.floor(secondsLeft / 60);
                            var secs = secondsLeft % 60;
                            clockEl.innerText = (mins < 10 ? "0" : "") + mins + ":" + (secs < 10 ? "0" : "") + secs;
                        }}
                    }}, 1000);
                </script>
                """
                components.html(timer_html, height=75)

                if rem_sec <= 0:
                    st.warning("⚠️ Time has expired! Submitting your answers automatically...")
                    submit_quiz()
                    st.rerun()
                else:
                    st.progress((300 - rem_sec) / 300)

                    with st.form("quiz_submission_form"):
                        for idx, q in enumerate(st.session_state.quiz_questions):
                            st.markdown(f"#### Question {idx+1}: {q.get('question')}")
                            opts = q.get("options", {})
                            
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
                if score >= 9:
                    perf_label = "🌟 Excellent Mastery"
                elif score >= 7:
                    perf_label = "👍 Good Understanding"
                elif score >= 5:
                    perf_label = "📖 Average — Review Recommended"

                st.markdown(f"""
                <div style="background-color: #f8fafc; border: 2px solid #3b82f6; border-radius: 0.75rem; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem;">
                    <h2 style="margin: 0; color: #1e3a8a;">QUIZ RESULT: {sub_name}</h2>
                    <h1 style="font-size: 3rem; margin: 0.5rem 0; color: #2563eb;">{score} / {total}</h1>
                    <p style="font-size: 1.1rem; color: #475569; margin: 0;">Performance: <strong>{perf_label}</strong> ({pct}%)</p>
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
# TAB 4: READ DOCUMENTS (ORIGINAL FORM VIEWER & MEMORY)
# ==============================================================================
with tab_read:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please select or create an academic subject in the sidebar to read its documents.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]

        st.subheader(f"📖 Read Documents — {sub_name}")
        st.caption("Read and study your course materials in their original form (PDF, DOCX, TXT). Uploaded documents and chunks are permanently preserved across refreshes.")

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
                    c_btn, _ = st.columns([2, 5])
                    with c_btn:
                        st.download_button(
                            label="📥 Download Original PDF",
                            data=raw_bytes,
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    st.write("")
                    
                    # Embedded PDF viewer using Base64 iframe
                    b64_pdf = base64.b64encode(raw_bytes).decode('utf-8')
                    pdf_display = f'''
                    <div style="border: 2px solid #cbd5e1; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                        <iframe src="data:application/pdf;base64,{b64_pdf}#toolbar=1" width="100%" height="800px" type="application/pdf">
                            <p>Your browser does not support inline PDF viewing. Please use the download button above.</p>
                        </iframe>
                    </div>
                    '''
                    st.markdown(pdf_display, unsafe_allow_html=True)

                    # Also provide expandable page-by-page view for convenience
                    with st.expander("📑 Structured Text View (Page by Page)", expanded=False):
                        chunks = load_document_chunks(sub_id, selected_doc_id)
                        if chunks:
                            current_page = None
                            page_text_acc = []
                            for c in chunks:
                                p = c.get('page')
                                if p != current_page and page_text_acc:
                                    p_label = f"Page {current_page}" if current_page else "General Content"
                                    st.markdown(f"#### 📄 {p_label}")
                                    st.markdown("\n\n".join(page_text_acc))
                                    st.divider()
                                    page_text_acc = []
                                current_page = p
                                page_text_acc.append(c.get('text', ''))
                            if page_text_acc:
                                p_label = f"Page {current_page}" if current_page else "General Content"
                                st.markdown(f"#### 📄 {p_label}")
                                st.markdown("\n\n".join(page_text_acc))
                        else:
                            st.caption("No chunks available.")
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
                        <div style="background-color: #ffffff; border: 2px solid #cbd5e1; border-radius: 8px; padding: 2rem; max-height: 800px; overflow-y: auto; font-family: 'Georgia', serif; font-size: 1.05rem; line-height: 1.8; color: #1e293b; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
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
