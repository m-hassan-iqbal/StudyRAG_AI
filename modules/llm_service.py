"""
LLM Service Layer for Groq API.
Handles grounded RAG answering, deep conceptual explanations,
and structured 10-question quiz generation with strict JSON validation.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq

from modules.config import DEFAULT_GROQ_MODEL


def get_groq_api_key() -> Optional[str]:
    """Retrieves GROQ_API_KEY from Streamlit secrets or OS environment variable."""
    key = None
    try:
        import streamlit as st
        if "GROQ_API_KEY" in st.secrets:
            key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not key:
        key = os.environ.get("GROQ_API_KEY")

    return key.strip() if key else None


def get_groq_model() -> str:
    """Retrieves configured Groq model or falls back to default."""
    model = None
    try:
        import streamlit as st
        if "GROQ_MODEL" in st.secrets:
            model = st.secrets["GROQ_MODEL"]
    except Exception:
        pass

    if not model:
        model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    return model.strip()


def get_groq_client() -> Groq:
    """Instantiates the official Groq client or raises a clear user-facing error."""
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .streamlit/secrets.toml "
            "file or set the GROQ_API_KEY environment variable."
        )
    return Groq(api_key=api_key)


RAG_SYSTEM_PROMPT = """You are a Subject-Aware University AI Tutor.
The student has selected a specific academic subject: "{subject_name}".

Answer the student's question using ONLY the retrieved course material provided in the context.
The retrieved context is your primary source of truth.

Rules:
1. Do not invent facts, sources, page numbers, quotations, or course-specific information.
2. Do not claim that information came from a document if the retrieved context does not support that claim.
3. If the retrieved context contains insufficient information to answer the question, clearly tell the student:
   "I couldn't find enough information about this topic in your uploaded {subject_name} materials."
4. Do not pretend that unrelated retrieved content is relevant.
5. Give a clear, educational answer appropriate for a university student.
6. When referencing sources, refer only to the documents and pages explicitly listed in the retrieved context.
"""


def generate_rag_answer(
    question: str,
    subject_name: str,
    retrieved_context: str,
    is_confident: bool,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Generates a course-grounded answer to a student question using Groq LLM.
    If retrieval confidence is low, refuses to hallucinate and provides clear honest guidance.
    """
    if not is_confident or not retrieved_context.strip() or retrieved_context == "No relevant course material found.":
        return (
            f"I couldn't find enough information about this topic in your uploaded **{subject_name}** materials.\n\n"
            f"*Tip: Make sure lecture slides, notes, or readings covering this concept have been uploaded and processed under this subject.*"
        )

    client = get_groq_client()
    model = model_name or get_groq_model()

    system_content = RAG_SYSTEM_PROMPT.format(subject_name=subject_name)
    messages = [{"role": "system", "content": system_content}]

    if chat_history:
        for turn in chat_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_prompt = (
        f"STUDENT QUESTION:\n{question}\n\n"
        f"RETRIEVED COURSE MATERIAL FROM {subject_name.upper()}:\n"
        f"{retrieved_context}\n\n"
        f"Provide a clear, grounded academic answer based exclusively on the material above."
    )
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {str(e)}")


DEEP_EXPLANATION_PROMPT = """You are a University AI Tutor specializing in conceptual depth and pedagogical mastery.
The student has selected the academic subject: "{subject_name}".

The student asked: "{question}"
Initial answer provided:
{initial_answer}

Course Context from {subject_name}:
{retrieved_context}

Please provide a deep conceptual educational explanation grounded in the student's course material.
Structure your response clearly using the following markdown headers:

### Simple Explanation
(A clean, high-level analogy or intuition for a beginner)

### Detailed Explanation
(The formal academic definition, principles, and theoretical foundation)

### How It Works
(Step-by-step operational mechanics or algorithmic flow)

### Practical Example
(A concrete, realistic walk-through demonstrating the concept)

### Important Concepts & Terminology
(Bullet points defining critical keywords)

### Common Mistakes & Misconceptions
(Pitfalls students frequently encounter on this topic)

### Exam-Focused Points
(High-yield takeaway tips and questions typically tested in university exams)
"""


def generate_deep_explanation(
    question: str,
    subject_name: str,
    retrieved_context: str,
    initial_answer: str,
    model_name: Optional[str] = None
) -> str:
    """
    Generates an in-depth conceptual breakdown adapted for university students.
    """
    client = get_groq_client()
    model = model_name or get_groq_model()

    prompt = DEEP_EXPLANATION_PROMPT.format(
        subject_name=subject_name,
        question=question,
        initial_answer=initial_answer,
        retrieved_context=retrieved_context
    )

    messages = [
        {"role": "system", "content": f"You are a master academic tutor for {subject_name}. Strictly avoid hallucinating external course details."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=2200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to generate deep explanation: {str(e)}")


QUIZ_SYSTEM_PROMPT = """You are a University Assessment Specialist.
Your task is to generate a rigorous, 10-question conceptual multiple-choice quiz (MCQ)
based EXCLUSIVELY on the provided course material for the subject: "{subject_name}".

CRITICAL REQUIREMENTS:
1. Exactly 10 questions.
2. Each question must test conceptual understanding, application, reasoning, or comparison (avoid trivial memorization).
3. Each question must have exactly 4 choices: "A", "B", "C", and "D".
4. Exactly one choice is correct.
5. Provide a clear, educational explanation for why the correct option is right.
6. Provide the specific academic sub-topic for each question.
7. Return ONLY valid JSON in the exact schema below, with no markdown fences, no preface, and no trailing commentary.

JSON SCHEMA:
{
  "questions": [
    {
      "question": "Question text here?",
      "options": {
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text"
      },
      "correct_answer": "A",
      "explanation": "Why A is correct and others are not.",
      "topic": "Concept / Topic Name"
    }
  ]
}
"""


def clean_json_string(raw: str) -> str:
    """Strips markdown code blocks, backticks, and extraneous text around JSON."""
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return raw[first_brace:last_brace + 1].strip()
    return raw


def validate_quiz(quiz_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validates that the generated quiz conforms to all MVP specifications:
    - Contains 'questions' list
    - Has at least 10 valid questions
    - Each question has options A, B, C, D and valid correct_answer.
    """
    if not isinstance(quiz_data, dict) or "questions" not in quiz_data:
        raise ValueError("Quiz output missing 'questions' root key.")

    questions = quiz_data["questions"]
    if not isinstance(questions, list) or len(questions) < 10:
        raise ValueError(f"Quiz must contain at least 10 questions, got {len(questions) if isinstance(questions, list) else 0}.")

    validated_questions = []
    for idx, q in enumerate(questions[:10]):
        if not isinstance(q, dict):
            raise ValueError(f"Question #{idx+1} is not a valid dictionary.")

        q_text = q.get("question", "").strip()
        if not q_text:
            raise ValueError(f"Question #{idx+1} has empty question text.")

        options = q.get("options", {})
        if not isinstance(options, dict) or not all(k in options for k in ["A", "B", "C", "D"]):
            raise ValueError(f"Question #{idx+1} must contain options A, B, C, and D.")

        correct = str(q.get("correct_answer", "")).strip().upper()
        if correct not in ["A", "B", "C", "D"]:
            raise ValueError(f"Question #{idx+1} has invalid correct_answer '{correct}'. Must be A, B, C, or D.")

        explanation = q.get("explanation", "Correct based on course material.").strip()
        topic = q.get("topic", "General Course Concepts").strip()

        validated_questions.append({
            "question": q_text,
            "options": {
                "A": str(options["A"]).strip(),
                "B": str(options["B"]).strip(),
                "C": str(options["C"]).strip(),
                "D": str(options["D"]).strip(),
            },
            "correct_answer": correct,
            "explanation": explanation,
            "topic": topic
        })

    return validated_questions


def generate_quiz(
    subject_name: str,
    subject_context: str,
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates a 10-question conceptual MCQ quiz grounded in the subject's uploaded material.
    Parses and validates the structured output.
    """
    if not subject_context or len(subject_context.strip()) < 100:
        raise ValueError(
            f"Not enough course material in {subject_name} to generate a comprehensive 10-question quiz. "
            "Please upload more course documents first."
        )

    client = get_groq_client()
    model = model_name or get_groq_model()

    system_prompt = QUIZ_SYSTEM_PROMPT.format(subject_name=subject_name)
    user_prompt = (
        f"Generate a 10-question conceptual MCQ quiz based on the following course material from {subject_name}:\n\n"
        f"{subject_context}\n\n"
        f"Return ONLY valid JSON matching the schema."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=3000
            )
            raw_text = response.choices[0].message.content.strip()
        except Exception as e2:
            raise RuntimeError(f"Groq API call for quiz generation failed: {str(e2)}")

    cleaned = clean_json_string(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as jde:
        raise ValueError(f"AI response was not valid JSON: {str(jde)}\nResponse snippet: {raw_text[:200]}")

    return validate_quiz(parsed)
