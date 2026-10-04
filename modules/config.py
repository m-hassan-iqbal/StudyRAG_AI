"""
Configuration settings for the Subject-Aware AI University Learning Assistant.
Provides centralized paths, defaults, and configuration variables.
"""

import os
from pathlib import Path
import re

# Base directory for runtime persistence
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUBJECTS_DIR = DATA_DIR / "subjects"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)

# LLM & Embedding Settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Document Processing & Chunking Defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Hybrid Retrieval Settings
DEFAULT_SEMANTIC_WEIGHT = 0.70
DEFAULT_KEYWORD_WEIGHT = 0.30
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_THRESHOLD = 0.28  # Cosine similarity threshold for normalized vectors

# Quiz Settings
QUIZ_QUESTION_COUNT = 10
QUIZ_TOTAL_MARKS = 10
QUIZ_TIME_LIMIT_SECONDS = 600  # 10 minutes


def normalize_subject_id(display_name: str) -> str:
    """
    Safely normalizes user-provided subject name for filesystem directory storage.
    Prevents path traversal and unsafe characters.
    e.g., 'Data Structures & Algorithms' -> 'data_structures_algorithms'
    """
    cleaned = display_name.strip().lower()
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"[\s-]+", "_", cleaned)
    normalized = cleaned.strip("_")
    return normalized if normalized else "unnamed_subject"
