"""
Quiz Engine & Assessment Module.
Manages quiz state, session persistence, resilient 5-minute timer calculation,
auto-submission, scoring, and post-quiz conceptual review recommendations.
"""

import time
from typing import List, Dict, Any, Tuple
import streamlit as st

from modules.config import QUIZ_TIME_LIMIT_SECONDS, QUIZ_QUESTION_COUNT


def init_quiz_session_state() -> None:
    """Initializes Streamlit session state keys required for the quiz workflow."""
    if "quiz_active" not in st.session_state:
        st.session_state.quiz_active = False
    if "quiz_subject_id" not in st.session_state:
        st.session_state.quiz_subject_id = None
    if "quiz_subject_name" not in st.session_state:
        st.session_state.quiz_subject_name = ""
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "quiz_user_answers" not in st.session_state:
        st.session_state.quiz_user_answers = {}
    if "quiz_start_time" not in st.session_state:
        st.session_state.quiz_start_time = None
    if "quiz_duration" not in st.session_state:
        st.session_state.quiz_duration = QUIZ_TIME_LIMIT_SECONDS
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = []


def start_quiz(*args, **kwargs) -> None:
    """
    Initializes and starts a new timed quiz.
    Supports all call signatures:
      - start_quiz(quiz_data["questions"])
      - start_quiz(questions)
      - start_quiz(quiz_data)
      - start_quiz(subject_id, subject_name, questions)
      - start_quiz(questions=..., subject_id=..., subject_name=...)
    """
    sub_id = kwargs.get("subject_id") or st.session_state.get("selected_subject_id") or ""
    sub_name = kwargs.get("subject_name") or st.session_state.get("quiz_subject_name") or ""
    questions = kwargs.get("questions", [])

    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, dict) and "questions" in arg:
            questions = arg["questions"]
        else:
            questions = arg
    elif len(args) == 2:
        if isinstance(args[0], (list, tuple)):
            questions, sub_id = args[0], args[1]
        else:
            sub_id, questions = args[0], args[1]
    elif len(args) >= 3:
        if isinstance(args[0], (list, tuple)):
            questions, sub_id, sub_name = args[0], args[1], args[2]
        else:
            sub_id, sub_name, questions = args[0], args[1], args[2]

    # Handle if questions is a dict containing 'questions' key
    if isinstance(questions, dict) and "questions" in questions:
        questions = questions["questions"]

    if not isinstance(questions, (list, tuple)):
        questions = []

    # Infer subject display name if not yet populated
    if sub_id and not sub_name:
        try:
            from modules.storage import get_subject_info
            info = get_subject_info(sub_id)
            if info:
                sub_name = info.get("display_name", sub_id)
        except Exception:
            sub_name = str(sub_id)

    st.session_state.quiz_active = True
    st.session_state.quiz_subject_id = sub_id
    st.session_state.quiz_subject_name = sub_name
    st.session_state.quiz_questions = list(questions)[:QUIZ_QUESTION_COUNT]
    st.session_state.quiz_user_answers = {i: None for i in range(len(st.session_state.quiz_questions))}
    st.session_state.quiz_start_time = time.time()
    st.session_state.quiz_duration = QUIZ_TIME_LIMIT_SECONDS
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_feedback = []


def reset_quiz() -> None:
    """Resets quiz state to idle."""
    st.session_state.quiz_active = False
    st.session_state.quiz_questions = []
    st.session_state.quiz_user_answers = {}
    st.session_state.quiz_start_time = None
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_feedback = []


def get_remaining_seconds() -> int:
    """
    Computes seconds remaining in the quiz.
    Returns 0 if time has expired or quiz is inactive.
    """
    if not st.session_state.get("quiz_active") or not st.session_state.get("quiz_start_time"):
        return 0

    elapsed = time.time() - st.session_state.quiz_start_time
    remaining = int(st.session_state.quiz_duration - elapsed)
    return max(0, remaining)


def format_remaining_time(seconds: int) -> str:
    """Formats seconds into MM:SS string."""
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins:02d}:{secs:02d}"


def calculate_quiz_results() -> Tuple[int, int, List[str], List[Dict[str, Any]]]:
    """
    Computes final quiz score and builds targeted topic review feedback.
    Returns: (score, total, review_topics, detailed_results)
    """
    questions = st.session_state.get("quiz_questions", [])
    user_answers = st.session_state.get("quiz_user_answers", {})
    
    score = 0
    incorrect_topics = set()
    detailed_results = []

    for idx, q in enumerate(questions):
        user_choice = user_answers.get(idx)
        correct_choice = q["correct_answer"]
        is_correct = (user_choice == correct_choice)

        if is_correct:
            score += 1
        else:
            topic = q.get("topic", "General Concept")
            incorrect_topics.add(topic)

        detailed_results.append({
            "index": idx + 1,
            "question": q["question"],
            "options": q["options"],
            "user_choice": user_choice,
            "correct_choice": correct_choice,
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "topic": q.get("topic", "")
        })

    st.session_state.quiz_score = score
    st.session_state.quiz_feedback = sorted(list(incorrect_topics))
    return score, len(questions), sorted(list(incorrect_topics)), detailed_results


def submit_quiz() -> None:
    """Marks quiz as submitted and calculates final scores."""
    st.session_state.quiz_submitted = True
    calculate_quiz_results()
