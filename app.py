"""
Subject-Aware AI University Learning Assistant
Main Streamlit Application.
Provides subject management, document processing, persistent FAISS indexing,
hybrid retrieval RAG chat, deep conceptual explanations, and a 10-question timed quiz.
"""

import os
import time
import streamlit as st
from typing import List, Dict, Any, Optional

st.set_page_config(
    page_title="AI University Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    submit_quiz,
)

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

if "selected_subject_id" not in st.session_state:
    st.session_state.selected_subject_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "last_context" not in st.session_state:
    st.session_state.last_context = ""

if "deep_explanation" not in st.session_state:
    st.session_state.deep_explanation = None

init_quiz_session_state()

@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def load_cached_embedding_model():
    return get_embedding_model()

try:
    load_cached_embedding_model()
except Exception as e:
    st.sidebar.error(f"Warning: Embedding model initialization error: {e}")

st.sidebar.title("📚 Subject Management")

api_key = get_groq_api_key()
if api_key:
    st.sidebar.success("🔑 Groq API Key: Active", icon="✅")
else:
    st.sidebar.warning(
        "⚠️ Groq API Key missing!\nAdd `GROQ_API_KEY` to `.streamlit/secrets.toml` or OS environment.",
        icon="⚠️"
    )

subjects = list_subjects()
subject_dict = {s["subject_id"]: s for s in subjects}

if subjects:
    subject_names = {s["subject_id"]: f"{s['display_name']} ({s['num_documents']} docs)" for s in subjects}
    current_id = st.session_state.selected_subject_id
    if current_id not in subject_names:
        current_id = subjects[0]["subject_id"]
        st.session_state.selected_subject_id = current_id

    selected_sub_id = st.sidebar.selectbox(
        "Current Academic Subject:",
        options=list(subject_names.keys()),
        format_func=lambda x: subject_names[x],
        index=list(subject_names.keys()).index(current_id) if current_id in subject_names else 0,
        key="subject_select_box"
    )

    if selected_sub_id != st.session_state.selected_subject_id:
        st.session_state.selected_subject_id = selected_sub_id
        st.session_state.chat_history = []
        st.session_state.last_query = ""
        st.session_state.last_answer = ""
        st.session_state.last_context = ""
        st.session_state.deep_explanation = None
        st.rerun()

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

st.sidebar.divider()

with st.sidebar.expander("➕ Create New Subject", expanded=(len(subjects) == 0)):
    with st.form("create_subject_form", clear_on_submit=True):
        new_subject_name = st.text_input("Subject Name", placeholder="e.g. Professional Practices")
        submitted = st.form_submit_button("Create Subject", use_container_width=True)
        if submitted:
            if not new_subject_name.strip():
                st.error("Please enter a subject name.")
            else:
                try:
                    new_id = create_subject(new_subject_name)
                    st.session_state.selected_subject_id = new_id
                    st.success(f"Subject '{new_subject_name}' created!")
                    st.rerun()
                except ValueError as ve:
                    st.error(str(ve))

if st.session_state.selected_subject_id:
    active_sub = subject_dict.get(st.session_state.selected_subject_id)
    with st.sidebar.expander("🗑️ Delete Current Subject"):
        if active_sub:
            st.warning(f"Delete **{active_sub['display_name']}** and all its documents and vectors?")
            if st.button("Confirm Delete Subject", type="primary", use_container_width=True):
                delete_subject(active_sub["subject_id"])
                st.session_state.selected_subject_id = None
                st.session_state.chat_history = []
                reset_quiz()
                st.success("Subject deleted successfully.")
                st.rerun()

st.markdown("""
<div class="main-header">
    <h1>🎓 AI University Learning Assistant</h1>
    <p>Your Course Material. Your Knowledge Base. Your AI Tutor.</p>
</div>
""", unsafe_allow_html=True)

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

tab_kb, tab_chat, tab_quiz, tab_settings = st.tabs([
    "📚 Knowledge Base & Upload",
    "🤖 AI Study Assistant",
    "📝 Conceptual Quiz (10 MCQs)",
    "⚙️ Settings & System"
])

# Tab 1: Knowledge Base
with tab_kb:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please create or select an academic subject in the sidebar to begin.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]

        st.subheader(f"📘 Subject: {sub_name}")
        st.caption("Upload lecture slides (PDF), notes (DOCX), or reading materials (TXT) to build this subject's private knowledge base.")

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

                        file_hash = compute_file_hash(file_bytes)
                        existing_doc = is_duplicate_document(sub_id, file_hash)
                        if existing_doc:
                            status.write(f"⚠️ `{filename}` is identical to already processed document `{existing_doc}`. Skipping.")
                            skipped_count += 1
                            continue

                        status.write(f"🔍 Extracting structured text from `{filename}`...")
                        try:
                            extracted = extract_document(file_bytes, filename)
                        except Exception as e:
                            st.error(f"Error extracting `{filename}`: {str(e)}")
                            continue

                        if not extracted:
                            st.warning(f"No readable text could be extracted from `{filename}`.")
                            continue

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
                        remaining_chunks = load_all_subject_chunks(sub_id)
                        build_and_save_index(sub_id, remaining_chunks)
                        st.success(f"Deleted {meta.get('filename')}")
                        st.rerun()

# Tab 2: AI Study Assistant
with tab_chat:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please create or select an academic subject in the sidebar to ask questions.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]

        st.subheader(f"🤖 AI Study Assistant — {sub_name}")
        st.caption("Ask questions about your uploaded materials. The assistant retrieves knowledge ONLY from this subject.")

        sub_stats = get_subject_stats(sub_id)
        if not sub_stats["has_index"] or sub_stats["num_chunks"] == 0:
            st.warning(
                f"The subject **{sub_name}** does not have any processed course documents yet. "
                "Please upload documents in the **Knowledge Base & Upload** tab first.",
                icon="⚠️"
            )

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

        user_query = st.chat_input(f"Ask a question about {sub_name}...")

        if user_query:
            if not api_key:
                st.error("Cannot query AI: GROQ_API_KEY is not configured.")
            elif not sub_stats["has_index"] or sub_stats["num_chunks"] == 0:
                st.error("Please upload and process course documents before asking questions.")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner(f"Searching {sub_name} knowledge base..."):
                        retrieved_chunks, is_confident = hybrid_search(
                            query=user_query,
                            subject_id=sub_id,
                            top_k=DEFAULT_TOP_K,
                            semantic_weight=DEFAULT_SEMANTIC_WEIGHT,
                            keyword_weight=DEFAULT_KEYWORD_WEIGHT,
                            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
                        )

                        context_str = build_context(retrieved_chunks)

                        st.session_state.last_query = user_query
                        st.session_state.last_context = context_str
                        st.session_state.deep_explanation = None

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
                        st.markdown(answer)

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

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks if is_confident else []
                        })

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

            if st.session_state.deep_explanation:
                st.markdown("### 🧠 In-Depth Conceptual Explanation")
                st.markdown(st.session_state.deep_explanation)

# Tab 3: Timed Conceptual Quiz (10 MCQs | 5-Minute Timer)
with tab_quiz:
    if not st.session_state.selected_subject_id:
        st.info("👈 Please create or select an academic subject in the sidebar to start a quiz.")
    else:
        current_sub = subject_dict.get(st.session_state.selected_subject_id)
        sub_name = current_sub["display_name"]
        sub_id = current_sub["subject_id"]
        sub_stats = get_subject_stats(sub_id)

        st.subheader(f"📝 Timed Conceptual Quiz — {sub_name}")
        st.markdown("""
        **Format:**
        * **10 Real-World & Conceptual Questions** (Multiple Choice)
        * **10 Marks** (1 mark per question — Score out of 10)
        * **5 Minutes** fixed countdown timer
        * **Deep Conceptual Understanding:** Scenario-based questions that test practical trade-offs, analytical problem-solving, and in-depth mastery of your course material.
        """)

        if not sub_stats["has_index"] or sub_stats["num_chunks"] == 0:
            st.warning("Please upload course documents before generating a quiz.", icon="⚠️")
        else:
            if not st.session_state.quiz_active and not st.session_state.quiz_submitted:
                if st.button("🚀 Start 5-Minute Quiz", type="primary"):
                    if not api_key:
                        st.error("Cannot generate quiz: GROQ_API_KEY is not configured.")
                    else:
                        with st.spinner(f"Synthesizing 10 real-world conceptual MCQs from {sub_name} materials..."):
                            all_chunks = load_all_subject_chunks(sub_id)
                            sample_text = "\n\n".join([f"Topic excerpt from {c['filename']}:\n{c['text']}" for c in all_chunks[:15]])
                            
                            try:
                                questions = generate_quiz(subject_name=sub_name, subject_context=sample_text)
                                start_quiz(sub_id, sub_name, questions)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating quiz: {str(e)}")

            elif st.session_state.quiz_active and not st.session_state.quiz_submitted:
                remaining_sec = get_remaining_seconds()

                if remaining_sec <= 0:
                    st.warning("⏰ Time is up! Automatically submitting your answers.")
                    submit_quiz()
                    st.rerun()
                else:
                    col_time, col_reset = st.columns([3, 1])
                    with col_time:
                        st.markdown(
                            f'<div class="quiz-timer-box">⏱ {format_remaining_time(remaining_sec)} remaining</div>',
                            unsafe_allow_html=True
                        )
                    with col_reset:
                        if st.button("Cancel Quiz"):
                            reset_quiz()
                            st.rerun()

                    questions = st.session_state.quiz_questions
                    
                    with st.form("quiz_form"):
                        for idx, q in enumerate(questions):
                            st.markdown(f"**Question {idx+1} of 10**")
                            st.markdown(f"**{q['question']}**")

                            opts = q["options"]
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
                                letter = selected_label[0]
                                st.session_state.quiz_user_answers[idx] = letter

                            st.write("")

                        submitted = st.form_submit_button("🏁 Submit Quiz", type="primary", use_container_width=True)
                        if submitted:
                            submit_quiz()
                            st.rerun()

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

# Tab 4: Settings & Diagnostics
with tab_settings:
    st.subheader("⚙️ System Configuration & Diagnostics")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("### 🤖 Groq LLM Configuration")
        curr_model = get_groq_model()
        st.text_input("Active Groq Model", value=curr_model, disabled=True)
        st.caption("Change by setting `GROQ_MODEL` in `.streamlit/secrets.toml` or OS environment.")
        
        has_key = bool(get_groq_api_key())
        st.markdown(f"**API Key Present:** {'✅ Yes' if has_key else '❌ No'}")

    with col_s2:
        st.markdown("### 🔍 Retrieval & Chunking Parameters")
        st.markdown(f"* **Embedding Model:** `{DEFAULT_EMBEDDING_MODEL}` (Sentence Transformers)")
        st.markdown(f"* **Chunk Size:** `{DEFAULT_CHUNK_SIZE}` characters")
        st.markdown(f"* **Chunk Overlap:** `{DEFAULT_CHUNK_OVERLAP}` characters")
        st.markdown(f"* **Hybrid Weights:** `{int(DEFAULT_SEMANTIC_WEIGHT*100)}% Semantic` / `{int(DEFAULT_KEYWORD_WEIGHT*100)}% Keyword`")
        st.markdown(f"* **Confidence Threshold:** `{DEFAULT_CONFIDENCE_THRESHOLD}`")

    st.divider()
    st.markdown("### 📁 Persistent Storage Architecture")
    st.code("""
data/
└── subjects/
    ├── <subject_id_1>/
    │   ├── subject_info.json
    │   ├── documents/
    │   ├── chunks/
    │   │   ├── doc_xxx_chunks.json
    │   │   └── doc_yyy_chunks.json
    │   ├── metadata/
    │   │   ├── documents_registry.json
    │   │   └── index_map.json
    │   └── index.faiss
    └── <subject_id_2>/
        └── ...
    """, language="text")
