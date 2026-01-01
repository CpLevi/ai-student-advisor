import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from collections import defaultdict

# Initialize OpenAI client (API key via environment variable)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# Intent Classification Setup
# ----------------------------
training_data = [
    ("How can I prepare for exams?", "study_guidance"),
    ("How do I study better?", "study_guidance"),
    ("I am confused about my career", "career_guidance"),
    ("AI or Data Science which is better", "career_guidance"),
    ("Which career should I choose", "career_guidance"),
    ("Explain machine learning", "concept_explanation"),
    ("What is artificial intelligence", "concept_explanation"),
    ("Difference between ML and DL", "concept_explanation"),
    ("How to learn Python", "skill_recommendation"),
    ("Skills needed for AI jobs", "skill_recommendation"),
]

X_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]

intent_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

intent_classifier.fit(X_train, y_train)

# ----------------------------
# Session Memory (per user)
# ----------------------------
user_sessions = defaultdict(list)

# ----------------------------
# Utility Functions
# ----------------------------
def detect_topic_lock(context_text):
    text = " ".join(context_text).lower()
    if "ai" in text and "data science" in text:
        return "ai_vs_data_science"
    return None

def predict_intent(user_input):
    probabilities = intent_classifier.predict_proba([user_input])[0]
    max_prob = max(probabilities)

    if max_prob < 0.45:
        return "general_query"

    return intent_classifier.predict([user_input])[0]

# ----------------------------
# Core AI Agent Function
# ----------------------------
def ai_student_agent(user_input, session_id="default_user"):
    session = user_sessions[session_id]
    session.append(user_input)

    # Keep memory limited
    session = session[-5:]
    user_sessions[session_id] = session

    intent = predict_intent(user_input)
    topic_lock = detect_topic_lock(session)

    context_summary = "\n".join(session)

    system_prompt = """
You are a professional AI Student Advisor.

Your role:
- Help students with career decisions, studies, and AI concepts
- Be honest, calm, and practical
- Do NOT overpromise
- Ask for clarification if the question is vague
- Keep explanations simple but insightful
"""

    user_prompt = f"""
Conversation context:
{context_summary}

Detected intent: {intent}
Locked topic: {topic_lock}

Current user question:
{user_input}

Guidelines:
- If topic is locked, stay within it
- Give structured answers (bullets when helpful)
- Personalize advice for a student audience
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=250,
        temperature=0.6
    )

    return response.choices[0].message.content
