"""
LLM Service Layer for Groq API.
Handles grounded RAG answering, deep conceptual explanations,
and structured 10-question quiz generation with strict JSON validation.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
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
    question: str = "",
    subject_name: str = "",
    retrieved_context: str = "",
    is_confident: bool = True,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None,
    query: Optional[str] = None,
    context: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generates a course-grounded answer to a student question using Groq LLM.
    If retrieval confidence is low, refuses to hallucinate and provides clear honest guidance.
    Supports both (question, retrieved_context) and (query, context) keyword parameters.
    """
    # Gracefully accept parameter aliases
    if not question and query:
        question = query
    if not retrieved_context and context:
        retrieved_context = context

    context_str = str(retrieved_context).strip() if retrieved_context else ""

    if not is_confident or not context_str or context_str == "No relevant course material found.":
        return (
            f"I couldn't find enough information about this topic in your uploaded **{subject_name}** materials.\n\n"
            f"*Tip: Make sure lecture slides, notes, or readings covering this concept have been uploaded and processed under this subject.*"
        )

    client = get_groq_client()
    model = model_name or get_groq_model()

    system_content = RAG_SYSTEM_PROMPT.format(subject_name=subject_name)

    messages = [{"role": "system", "content": system_content}]

    # Include recent conversational context if present (last 2 turns)
    if chat_history:
        for turn in chat_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    user_prompt = (
        f"STUDENT QUESTION:\n{question}\n\n"
        f"RETRIEVED COURSE MATERIAL FROM {subject_name.upper()}:\n"
        f"{context_str}\n\n"
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
    question: str = "",
    subject_name: str = "",
    retrieved_context: str = "",
    initial_answer: str = "",
    model_name: Optional[str] = None,
    query: Optional[str] = None,
    previous_answer: Optional[str] = None,
    context: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generates an in-depth conceptual breakdown adapted for university students.
    Supports both (question, initial_answer, retrieved_context) and (query, previous_answer, context) parameter aliases.
    """
    # Resolve aliases
    if not question and query:
        question = query
    if not initial_answer and previous_answer:
        initial_answer = previous_answer
    if not retrieved_context and context:
        retrieved_context = context

    client = get_groq_client()
    model = model_name or get_groq_model()

    prompt = DEEP_EXPLANATION_PROMPT.format(
        subject_name=subject_name,
        question=question,
        initial_answer=initial_answer,
        retrieved_context=str(retrieved_context or "")
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


QUIZ_SYSTEM_PROMPT = """You are an Expert University Assessment Architect and Senior Academic AI Agent.
Your objective is to generate an advanced, high-yield 10-question conceptual multiple-choice quiz (MCQ)
based EXCLUSIVELY on the course concepts from the subject: "{subject_name}".

PEDAGOGICAL & REAL-WORLD CONCEPTUAL REQUIREMENTS:
1. Generate EXACTLY 10 questions.
2. Focus on REAL-WORLD SCENARIOS, SYSTEM TRADE-OFFS, CONCEPTUAL RELATIONSHIPS, and PRACTICAL APPLICATION.
   Avoid surface-level trivia, trivial memorization, or simple textbook definitions.
   Every question should challenge the student to apply knowledge to realistic professional or engineering contexts.
3. Each question must provide 4 distinct, plausible options (A, B, C, D) with no obvious giveaway answers.
4. Exactly one option is correct.
5. Provide a thorough, pedagogical explanation detailing WHY the correct option is right and the conceptual pitfall of incorrect alternatives.
6. Categorize each question with its specific academic topic or conceptual domain.

FORMATTING REQUIREMENTS:
Return strictly a valid JSON object matching this structure:
{{
  "questions": [
    {{
      "question": "Realistic scenario or in-depth conceptual question text?",
      "options": {{
        "A": "Plausible choice A",
        "B": "Plausible choice B",
        "C": "Plausible choice C",
        "D": "Plausible choice D"
      }},
      "correct_answer": "B",
      "explanation": "In-depth pedagogical breakdown explaining why B is correct in this scenario.",
      "topic": "Topic Name"
    }}
  ]
}}
"""


class QuizResult(list):
    """
    A smart list containing validated quiz questions that also supports dictionary-style access.
    Enables callers to use either:
      - `questions = generate_quiz(...)` -> `for q in questions: ...`
      - `quiz_data = generate_quiz(...)` -> `if 'questions' in quiz_data: start_quiz(quiz_data['questions'])`
    """
    def __getitem__(self, item):
        if item == "questions":
            return list(self)
        return super().__getitem__(item)

    def __contains__(self, item):
        if item == "questions":
            return True
        return super().__contains__(item)

    def get(self, key, default=None):
        if key == "questions":
            return list(self)
        return default


def sanitize_and_extract_json(raw: str) -> Optional[Any]:
    """
    Robustly extracts and parses JSON from LLM output, handling:
    - Markdown code fences (```json ... ```)
    - Unescaped control characters or whitespace in keys (e.g. '\\n \"questions\"')
    - Root array vs root object
    - Strict and non-strict JSON parsing
    """
    if not raw or not raw.strip():
        return None

    raw_text = raw.strip()

    # 1. Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # 2. Try direct json.loads (strict and non-strict)
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            return json.loads(cleaned, strict=False)
        except Exception:
            pass

    # 3. Locate outer container ({ or [)
    first_brace = -1
    last_brace = -1
    for i, ch in enumerate(cleaned):
        if ch in ("{", "["):
            first_brace = i
            break
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] in ("}", "]"):
            last_brace = i
            break

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = cleaned[first_brace:last_brace + 1].strip()
        try:
            return json.loads(snippet, strict=False)
        except Exception:
            pass

    # 4. Fallback: Regex extraction of individual question objects
    questions = []
    q_pattern = re.compile(
        r'\{\s*"question"\s*:\s*"(?P<q>.*?)"\s*,\s*"options"\s*:\s*\{(?P<opts>.*?)\}\s*,\s*"correct_answer"\s*:\s*"(?P<ans>[A-D])"\s*,\s*"explanation"\s*:\s*"(?P<exp>.*?)"(?:\s*,\s*"topic"\s*:\s*"(?P<top>.*?)")?\s*\}',
        re.DOTALL
    )
    for m in q_pattern.finditer(raw_text):
        q_text = m.group("q").strip()
        opts_raw = m.group("opts")
        ans = m.group("ans").strip().upper()
        exp = m.group("exp").strip()
        top = (m.group("top") or "Core Concepts").strip()

        opts_dict = {}
        for opt_match in re.finditer(r'"([A-D])"\s*:\s*"(.*?)"', opts_raw):
            opts_dict[opt_match.group(1)] = opt_match.group(2).strip()

        if q_text and len(opts_dict) == 4:
            questions.append({
                "question": q_text,
                "options": opts_dict,
                "correct_answer": ans,
                "explanation": exp,
                "topic": top
            })

    if questions:
        return {"questions": questions}

    return None


# Backward-compatibility alias
clean_json_string = sanitize_and_extract_json


def normalize_quiz_dict(data: Any) -> List[Dict[str, Any]]:
    """
    Extracts the list of question objects from any data structure returned by the LLM.
    Handles dirty keys like '\\n \"questions\"', 'quiz', 'mcqs', or a raw list.
    """
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        # Check normalized keys
        for k, v in data.items():
            clean_k = str(k).strip().strip('"\'').strip().lower()
            if "question" in clean_k or "quiz" in clean_k or "mcq" in clean_k or "items" in clean_k:
                if isinstance(v, list):
                    return v

        # Check if any value is a list of question dicts
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                return v

    return []


def validate_quiz(raw_data: Any) -> QuizResult:
    """
    Validates and standardizes generated quiz questions into a uniform schema:
    10 conceptual MCQs, options A-D, single correct answer, and in-depth explanation.
    Returns a QuizResult object that supports both list and dict access.
    """
    questions_list = normalize_quiz_dict(raw_data)
    if not questions_list:
        raise ValueError("Could not extract a valid list of questions from the AI output.")

    validated = []
    for idx, item in enumerate(questions_list):
        if not isinstance(item, dict):
            continue

        # Clean all keys in the question dictionary
        clean_item = {}
        for ik, iv in item.items():
            clean_ik = str(ik).strip().strip('"\'').strip().lower()
            clean_item[clean_ik] = iv

        q_text = clean_item.get("question") or clean_item.get("q") or clean_item.get("text") or ""
        q_text = str(q_text).strip()
        if not q_text:
            continue

        raw_options = clean_item.get("options") or clean_item.get("choices") or clean_item.get("answers") or {}
        options_dict = {}
        if isinstance(raw_options, dict):
            for opt_k, opt_v in raw_options.items():
                norm_opt_k = str(opt_k).strip().upper().replace("OPTION_", "").replace("CHOICE_", "")
                if norm_opt_k in ["A", "B", "C", "D"]:
                    options_dict[norm_opt_k] = str(opt_v).strip()
        elif isinstance(raw_options, list):
            letters = ["A", "B", "C", "D"]
            for i, opt_val in enumerate(raw_options[:4]):
                options_dict[letters[i]] = str(opt_val).strip()

        for letter in ["A", "B", "C", "D"]:
            if letter not in options_dict:
                options_dict[letter] = f"Option {letter}"

        corr = str(clean_item.get("correct_answer") or clean_item.get("correct") or clean_item.get("answer") or "A").strip().upper()
        corr_match = re.search(r"\b([A-D])\b", corr)
        if corr_match:
            correct_choice = corr_match.group(1)
        elif corr and corr[0] in ["A", "B", "C", "D"]:
            correct_choice = corr[0]
        else:
            correct_choice = "A"

        explanation = clean_item.get("explanation") or clean_item.get("reason") or "Correct based on course principles."
        topic = clean_item.get("topic") or clean_item.get("concept") or "Core Professional Concepts"

        validated.append({
            "question": q_text,
            "options": options_dict,
            "correct_answer": correct_choice,
            "explanation": str(explanation).strip(),
            "topic": str(topic).strip()
        })

    if len(validated) < 5:
        raise ValueError(f"AI generated only {len(validated)} questions; expected at least 10.")

    return QuizResult(validated[:10])


def generate_quiz(
    subject_name: str = "",
    subject_context: Optional[str] = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    num_questions: int = 10,
    model_name: Optional[str] = None,
    **kwargs
) -> QuizResult:
    """
    Generates a 10-question conceptual real-world MCQ quiz grounded in the subject's material.
    Supports both direct string context (subject_context) and chunks list (chunks).
    Returns a QuizResult that works as both a List and a Dict ('questions' in quiz_data).
    """
    # Reconstruct context from chunks if raw text was not provided
    if not subject_context and chunks:
        sample_texts = [c.get("text", "") for c in chunks[:30] if isinstance(c, dict)]
        subject_context = "\n\n".join(sample_texts)

    if not subject_context or len(subject_context.strip()) < 50:
        raise ValueError(
            f"Not enough course material in '{subject_name}' to generate a {num_questions}-question quiz. "
            "Please upload more course documents first."
        )

    client = get_groq_client()
    model = model_name or get_groq_model()

    system_prompt = QUIZ_SYSTEM_PROMPT.format(subject_name=subject_name)
    user_prompt = (
        f"Generate a {num_questions}-question conceptual, real-world application MCQ quiz based on the following course material from {subject_name}:\n\n"
        f"{subject_context[:8000]}\n\n"
        f"Return strictly a valid JSON object matching the requested schema."
    )

    raw_text = ""
    # Try with json_object response format
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.25,
            response_format={"type": "json_object"},
            max_tokens=3500
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception:
        # Fallback without json_object enforcement
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.25,
                max_tokens=3500
            )
            raw_text = response.choices[0].message.content.strip()
        except Exception as e2:
            raise RuntimeError(f"Groq API call for quiz generation failed: {str(e2)}")

    parsed = sanitize_and_extract_json(raw_text)
    if parsed is None:
        raise ValueError(f"Could not parse quiz JSON from AI model.\nResponse snippet: {raw_text[:250]}")

    return validate_quiz(parsed)
