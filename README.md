# 🎓 Subject-Aware AI University Learning Assistant

> **Your Course Material. Your Knowledge Base. Your AI Tutor.**

A modular, beginner-friendly **Subject-Aware RAG (Retrieval-Augmented Generation)** application built with Python and Streamlit. Designed specifically for university students to ingest their course documents (PDF, DOCX, TXT), organize them into isolated academic subject knowledge bases, and interact with a course-grounded AI tutor.

---

## 📌 Problem Statement

University students juggle multiple courses—from *Data Structures* and *Database Systems* to *Operating Systems*. Traditional general-purpose chatbots suffer from three major issues:
1. **Hallucinations & Generic Answers:** General LLMs answer from broad internet training data rather than the student's actual syllabus and exam expectations.
2. **Cross-Subject Knowledge Bleed:** Standard search tools mix up concepts between unrelated courses.
3. **Missing Citations:** Students need to know exact source documents and slide/page references when preparing for examinations.

This application enforces a **strict subject boundary**: retrieval happens *only* against the currently selected academic subject.

---

## ✨ Core Features

* **📚 Isolated Subject Management:** Create, view, inspect, and delete subject knowledge bases. Subject names are safely normalized on disk.
* **📄 Multi-Format Document Ingestion:** Supports **PDF** (extracts page-by-page preserving real page numbers), **DOCX** (preserves paragraphs), and **TXT** files.
* **🔒 Duplicate Prevention:** SHA-256 hash checks prevent duplicate processing and redundant vector calculations.
* **🧩 Chunking & Metadata Preservation:** Configurable sliding-window chunking (default ~800 characters with 150-character overlap) maintaining document ID, filename, page number, and chunk ID.
* **⚡ Persistent Vector Indexing:** Fast, local vector embeddings with `sentence-transformers` (`all-MiniLM-L6-v2`) and L2-normalized cosine similarity via **FAISS** (`IndexFlatIP`). Chunks and vectors are processed once and persisted locally.
* **🔍 Hybrid Semantic + Keyword Retrieval:** Blends vector similarity (70%) with transparent token matching (30%) to ensure exact academic terms and conceptual queries are both captured.
* **🛡️ Retrieval Confidence Gating:** If no relevant material is found, the assistant refuses to hallucinate, stating: *"I couldn't find enough information about this topic in your uploaded materials."*
* **🤖 Grounded AI Tutor:** Powered by the official **Groq API** (e.g. `llama-3.3-70b-versatile`) with strict academic grounding prompts.
* **📑 Transparent Citations:** Shows exact filenames, page numbers (or N/A for unpaged docs), similarity scores, and text excerpts.
* **🧠 Deep Conceptual Explanation:** One-click conceptual breakdown providing Simple Explanations, Detailed Explanations, How It Works, Examples, Important Concepts, Common Pitfalls, and Exam Takeaways.
* **📝 Timed Conceptual Quiz (10 MCQs):** Generates exactly 10 multiple-choice questions grounded in the subject's uploaded material, featuring a fixed 10-minute timer, automated scoring, and revision recommendations for missed topics.

---

## 🚀 Getting Started

### 1. Configure Groq API Key
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_groq_api_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
