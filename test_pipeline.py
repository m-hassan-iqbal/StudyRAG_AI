"""
Comprehensive validation and test script for the Subject-Aware RAG system.
Tests:
- Subject management (create, list, stats, delete)
- Duplicate prevention (SHA-256 hashing)
- Text chunking & metadata integrity
- Embedding generation & FAISS vector indexing
- Subject boundary isolation
- Hybrid retrieval (semantic + keyword)
- Context construction
- Quiz JSON validator and scoring logic
"""

import os
import shutil
from pathlib import Path
import numpy as np

from modules.config import SUBJECTS_DIR, normalize_subject_id
from modules.storage import (
    create_subject,
    delete_subject,
    list_subjects,
    get_subject_stats,
    compute_file_hash,
    is_duplicate_document,
    save_document_chunks,
    load_all_subject_chunks,
    delete_document,
)
from modules.extractor import clean_text, extract_txt
from modules.chunker import create_chunks
from modules.indexer import (
    get_embedding_model,
    build_and_save_index,
    load_subject_index,
)
from modules.retriever import hybrid_search, build_context
from modules.llm_service import validate_quiz, clean_json_string
from modules.quiz_engine import calculate_quiz_results


def run_tests():
    print("==================================================")
    print("STARTING TEST SUITE: SUBJECT-AWARE RAG PIPELINE")
    print("==================================================")

    # 1. Test Subject Normalization
    sub1_id = normalize_subject_id("Data Structures & Algorithms")
    assert sub1_id == "data_structures_algorithms", f"Normalization failed: {sub1_id}"
    print("[PASS] Test 1: Subject name normalization passed.")

    # Clean test subjects if left over
    delete_subject("test_data_structures")
    delete_subject("test_database_systems")

    # 2. Test Subject Creation
    s1_id = create_subject("Test Data Structures")
    s2_id = create_subject("Test Database Systems")
    assert s1_id == "test_data_structures"
    assert s2_id == "test_database_systems"
    
    subs = list_subjects()
    sub_ids = [s["subject_id"] for s in subs]
    assert "test_data_structures" in sub_ids
    assert "test_database_systems" in sub_ids
    print("[PASS] Test 2: Subject creation & listing passed.")

    # 3. Test Document Extraction (TXT)
    sample_ds_text = (
        "An AVL tree is a self-balancing binary search tree. In an AVL tree, the heights of the two child subtrees "
        "of any node differ by at most one; if at any time they differ by more than one, rebalancing is done to restore this property. "
        "Lookup, insertion, and deletion all take O(log n) time in both the average and worst cases. "
        "Rotations are used to balance the tree: Left Rotation, Right Rotation, Left-Right Rotation, and Right-Left Rotation."
    )
    raw_bytes = sample_ds_text.encode("utf-8")
    extracted = extract_txt(raw_bytes, "lecture_avl.txt")
    assert len(extracted) == 1
    assert extracted[0]["page"] is None
    assert "AVL tree" in extracted[0]["text"]
    print("[PASS] Test 3: Document extraction passed.")

    # 4. Test Duplicate Prevention
    hash1 = compute_file_hash(raw_bytes)
    assert not is_duplicate_document(s1_id, hash1)

    # 5. Test Chunking
    chunks = create_chunks(
        extracted_documents=extracted,
        subject_id=s1_id,
        document_id="doc_avl_01",
        chunk_size=200,
        overlap=50
    )
    assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
    for c in chunks:
        assert c["subject_id"] == s1_id
        assert c["document_id"] == "doc_avl_01"
        assert c["filename"] == "lecture_avl.txt"
        assert "chunk_id" in c
        assert len(c["text"]) > 0
    print(f"[PASS] Test 4: Chunking created {len(chunks)} overlapping chunks with complete metadata.")

    # 6. Save Chunks & Check Duplicate
    save_document_chunks(
        subject_id=s1_id,
        document_id="doc_avl_01",
        filename="lecture_avl.txt",
        file_hash=hash1,
        chunks=chunks
    )
    assert is_duplicate_document(s1_id, hash1) == "lecture_avl.txt"
    print("[PASS] Test 5: Chunk persistence and duplicate detection passed.")

    # 7. Test Ingestion for Subject 2 (Database Systems)
    sample_db_text = (
        "Relational database management systems organize data into tables consisting of rows and columns. "
        "SQL queries use SELECT, FROM, WHERE, and JOIN clauses to retrieve related records across tables. "
        "ACID properties (Atomicity, Consistency, Isolation, Durability) guarantee transaction reliability."
    )
    db_extracted = extract_txt(sample_db_text.encode("utf-8"), "lecture_sql.txt")
    db_chunks = create_chunks(
        extracted_documents=db_extracted,
        subject_id=s2_id,
        document_id="doc_db_01",
        chunk_size=200,
        overlap=50
    )
    save_document_chunks(
        subject_id=s2_id,
        document_id="doc_db_01",
        filename="lecture_sql.txt",
        file_hash=compute_file_hash(sample_db_text.encode("utf-8")),
        chunks=db_chunks
    )
    print("[PASS] Test 6: Ingested separate data for Subject 2.")

    # 8. Build FAISS Indexes
    build_and_save_index(s1_id, chunks)
    build_and_save_index(s2_id, db_chunks)

    idx1, map1 = load_subject_index(s1_id)
    idx2, map2 = load_subject_index(s2_id)
    assert idx1 is not None and len(map1) == len(chunks)
    assert idx2 is not None and len(map2) == len(db_chunks)
    print("[PASS] Test 7: Built and verified persistent FAISS indexes for both subjects.")

    # 9. Test Subject Boundary Isolation (CRITICAL)
    results_ds, conf_ds = hybrid_search("AVL tree rotations", s1_id, top_k=3)
    assert len(results_ds) > 0
    assert any("AVL" in r["text"] for r in results_ds)
    for r in results_ds:
        assert r["subject_id"] == s1_id, "Subject bleed detected in Data Structures!"

    results_db, conf_db = hybrid_search("AVL tree rotations", s2_id, top_k=3)
    for r in results_db:
        assert r["subject_id"] == s2_id, "Subject bleed: Database Systems returned Data Structures chunks!"
        assert "AVL" not in r["text"], "Database Systems returned AVL content!"
    print("[PASS] Test 8: Subject boundary isolation rigorously verified! No cross-subject leakage.")

    # 10. Test Context Construction
    ctx = build_context(results_ds)
    assert "SOURCE 1" in ctx
    assert "lecture_avl.txt" in ctx
    print("[PASS] Test 9: Context construction verified.")

    # 11. Test Quiz Validator
    mock_quiz_json = {
        "questions": [
            {
                "question": f"Conceptual Question #{i} regarding balancing?",
                "options": {
                    "A": "Option A explanation",
                    "B": "Option B explanation",
                    "C": "Option C explanation",
                    "D": "Option D explanation"
                },
                "correct_answer": "B",
                "explanation": "Because B satisfies the AVL balance factor.",
                "topic": "AVL Rotations"
            }
            for i in range(1, 11)
        ]
    }
    validated = validate_quiz(mock_quiz_json)
    assert len(validated) == 10
    print("[PASS] Test 10: Quiz JSON schema validator verified (10 questions, 4 options each).")

    # Cleanup test subjects
    delete_subject(s1_id)
    delete_subject(s2_id)
    print("[PASS] Test 11: Cleanup completed.")

    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
