"""
Subject Storage & Persistence Layer.
Manages isolated directories, document registries, SHA-256 duplicate checking,
and chunk JSON files under data/subjects/<subject_id>/.
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from modules.config import DATA_DIR, SUBJECTS_DIR, normalize_subject_id

ACTIVE_SESSION_FILE = DATA_DIR / "active_session.json"


def save_last_active_subject(subject_id: str) -> None:
    """Remembers the active subject across page reloads and browser sessions."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(ACTIVE_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_subject_id": subject_id, "updated_at": datetime.now().isoformat()}, f, indent=2)
    except Exception:
        pass


def get_last_active_subject() -> Optional[str]:
    """Retrieves the last active subject ID remembered across sessions."""
    if ACTIVE_SESSION_FILE.exists():
        try:
            with open(ACTIVE_SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                sub_id = data.get("last_subject_id")
                if sub_id and (SUBJECTS_DIR / sub_id).exists():
                    return sub_id
        except Exception:
            pass
    return None


def get_subject_path(subject_id: str) -> Path:
    """Returns the base directory path for a subject."""
    return SUBJECTS_DIR / subject_id


def ensure_subject_dirs(subject_id: str) -> Path:
    """Ensures that the directory hierarchy for a subject exists."""
    base = get_subject_path(subject_id)
    (base / "documents").mkdir(parents=True, exist_ok=True)
    (base / "chunks").mkdir(parents=True, exist_ok=True)
    (base / "metadata").mkdir(parents=True, exist_ok=True)
    return base


def save_original_document(subject_id: str, filename: str, file_bytes: bytes) -> Path:
    """
    Persists the original uploaded document in its raw unmodified form (PDF, DOCX, TXT)
    so it can be read and viewed permanently across page refreshes.
    """
    ensure_subject_dirs(subject_id)
    doc_path = get_subject_path(subject_id) / "documents" / filename
    with open(doc_path, "wb") as f:
        f.write(file_bytes)
    return doc_path


def get_original_document_path(subject_id: str, filename: str) -> Optional[Path]:
    """Returns path to the original document file if it exists on disk."""
    doc_path = get_subject_path(subject_id) / "documents" / filename
    return doc_path if doc_path.exists() else None


def load_original_document(subject_id: str, filename: str) -> Optional[bytes]:
    """Reads raw original file bytes from persistent disk storage."""
    doc_path = get_original_document_path(subject_id, filename)
    if doc_path and doc_path.exists():
        try:
            with open(doc_path, "rb") as f:
                return f.read()
        except Exception:
            return None
    return None


def compute_file_hash(file_bytes: bytes) -> str:
    """Computes a SHA-256 hash of file content to detect duplicate uploads."""
    return hashlib.sha256(file_bytes).hexdigest()


def get_registry_path(subject_id: str) -> Path:
    return get_subject_path(subject_id) / "metadata" / "documents_registry.json"


def load_registry(subject_id: str) -> Dict[str, Any]:
    """Loads the documents registry for a subject."""
    reg_path = get_registry_path(subject_id)
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"documents": {}}
    return {"documents": {}}


def save_registry(subject_id: str, registry_data: Dict[str, Any]) -> None:
    """Saves the documents registry for a subject."""
    reg_path = get_registry_path(subject_id)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry_data, f, indent=2)


def is_duplicate_document(subject_id: str, file_hash: str) -> Optional[str]:
    """
    Checks if a file with the given SHA-256 hash has already been processed for this subject.
    Returns existing filename if duplicate, else None.
    """
    registry = load_registry(subject_id)
    for doc_id, doc_meta in registry.get("documents", {}).items():
        if doc_meta.get("file_hash") == file_hash:
            return doc_meta.get("filename", doc_id)
    return None


def create_subject(display_name: str) -> str:
    """
    Creates a new isolated subject space.
    Returns subject_id. Raises ValueError if name is invalid or already exists.
    """
    display_name = display_name.strip()
    if not display_name:
        raise ValueError("Subject name cannot be empty.")

    subject_id = normalize_subject_id(display_name)
    subject_dir = get_subject_path(subject_id)

    if subject_dir.exists():
        info_file = subject_dir / "subject_info.json"
        if info_file.exists():
            raise ValueError(f"Subject '{display_name}' (ID: {subject_id}) already exists.")

    ensure_subject_dirs(subject_id)
    info = {
        "subject_id": subject_id,
        "display_name": display_name,
        "created_at": datetime.now().isoformat(),
    }
    with open(subject_dir / "subject_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    save_registry(subject_id, {"documents": {}})
    return subject_id


def delete_subject(subject_id: str) -> bool:
    """Safely removes an entire subject knowledge base and all associated files."""
    subject_dir = get_subject_path(subject_id)
    if subject_dir.exists() and subject_dir.is_dir():
        shutil.rmtree(subject_dir)
        return True
    return False


def list_subjects() -> List[Dict[str, Any]]:
    """
    Lists all existing subjects with their statistics.
    Returns a list of dicts with subject_id, display_name, num_documents, num_chunks, created_at.
    """
    subjects = []
    if not SUBJECTS_DIR.exists():
        return subjects

    for item in sorted(SUBJECTS_DIR.iterdir()):
        if item.is_dir():
            info_file = item / "subject_info.json"
            if info_file.exists():
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        info = json.load(f)
                except Exception:
                    info = {"subject_id": item.name, "display_name": item.name}
            else:
                info = {"subject_id": item.name, "display_name": item.name}

            stats = get_subject_stats(item.name)
            info["num_documents"] = stats["num_documents"]
            info["num_chunks"] = stats["num_chunks"]
            info["has_index"] = stats["has_index"]
            subjects.append(info)

    return subjects


def get_subject_info(subject_id: str) -> Optional[Dict[str, Any]]:
    """Returns metadata for a specific subject."""
    info_file = get_subject_path(subject_id) / "subject_info.json"
    if info_file.exists():
        with open(info_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_subject_stats(subject_id: str) -> Dict[str, Any]:
    """Computes document and chunk statistics for a single subject."""
    registry = load_registry(subject_id)
    num_docs = len(registry.get("documents", {}))
    
    # Count total chunks from chunk directory
    chunks_dir = get_subject_path(subject_id) / "chunks"
    total_chunks = 0
    if chunks_dir.exists():
        for chunk_file in chunks_dir.glob("*_chunks.json"):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_chunks += len(data)
            except Exception:
                pass

    index_exists = (get_subject_path(subject_id) / "index.faiss").exists()

    return {
        "num_documents": num_docs,
        "num_chunks": total_chunks,
        "has_index": index_exists
    }


def get_global_stats() -> Dict[str, int]:
    """Computes totals across all subjects for dashboard reporting."""
    subjects = list_subjects()
    total_subjects = len(subjects)
    total_docs = sum(s.get("num_documents", 0) for s in subjects)
    total_chunks = sum(s.get("num_chunks", 0) for s in subjects)
    return {
        "total_subjects": total_subjects,
        "total_documents": total_docs,
        "total_chunks": total_chunks
    }


def save_document_chunks(
    subject_id: str,
    document_id: str,
    filename: str,
    file_hash: str,
    chunks: List[Dict[str, Any]]
) -> None:
    """
    Saves document chunks to a JSON file and records the document in the registry.
    """
    ensure_subject_dirs(subject_id)
    chunks_path = get_subject_path(subject_id) / "chunks" / f"{document_id}_chunks.json"
    
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    # Update document registry
    registry = load_registry(subject_id)
    registry.setdefault("documents", {})[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "file_hash": file_hash,
        "num_chunks": len(chunks),
        "processed_at": datetime.now().isoformat()
    }
    save_registry(subject_id, registry)


def load_all_subject_chunks(subject_id: str) -> List[Dict[str, Any]]:
    """Loads all chunk records belonging to the subject across all documents."""
    chunks_dir = get_subject_path(subject_id) / "chunks"
    all_chunks = []
    if not chunks_dir.exists():
        return all_chunks

    for chunk_file in sorted(chunks_dir.glob("*_chunks.json")):
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_chunks.extend(data)
        except Exception:
            continue

    return all_chunks


def load_document_chunks(subject_id: str, document_id: str) -> List[Dict[str, Any]]:
    """Loads chunks specifically for a single document."""
    chunk_file = get_subject_path(subject_id) / "chunks" / f"{document_id}_chunks.json"
    if chunk_file.exists():
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []


def delete_document(subject_id: str, document_id: str) -> bool:
    """
    Permanently removes a document, its raw uploaded file, and its chunks
    from the subject's knowledge base.
    Caller is responsible for triggering FAISS re-indexing if needed.
    """
    chunk_file = get_subject_path(subject_id) / "chunks" / f"{document_id}_chunks.json"
    if chunk_file.exists():
        try:
            chunk_file.unlink()
        except Exception:
            pass

    registry = load_registry(subject_id)
    filename = None
    if document_id in registry.get("documents", {}):
        filename = registry["documents"][document_id].get("filename")
        del registry["documents"][document_id]
        save_registry(subject_id, registry)

    # Also delete raw file from documents/ directory
    if filename:
        raw_path = get_subject_path(subject_id) / "documents" / filename
        if raw_path.exists():
            try:
                raw_path.unlink()
            except Exception:
                pass

    return True
