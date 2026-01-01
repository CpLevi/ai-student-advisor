import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from collections import defaultdict

# Initialize OpenAI clientimport os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from collections import defaultdict

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# ENHANCED Intent Classification (60+ examples)
# ----------------------------
training_data = [
    # Study Guidance (15 examples)
    ("How can I prepare for exams?", "study_guidance"),
    ("How do I study better?", "study_guidance"),
    ("Best study techniques", "study_guidance"),
    ("Time management for students", "study_guidance"),
    ("How to focus while studying", "study_guidance"),
    ("Study schedule tips", "study_guidance"),
    ("Memory improvement techniques", "study_guidance"),
    ("How to take better notes", "study_guidance"),
    ("Effective study methods", "study_guidance"),
    ("How to avoid procrastination", "study_guidance"),
    ("I can't concentrate on studies", "study_guidance"),
    ("How to prepare for competitive exams", "study_guidance"),
    ("Best apps for studying", "study_guidance"),
    ("How to revise effectively", "study_guidance"),
    ("Tips for last minute exam prep", "study_guidance"),
    
    # Career Guidance (15 examples)
    ("I am confused about my career", "career_guidance"),
    ("AI or Data Science which is better", "career_guidance"),
    ("Which career should I choose", "career_guidance"),
    ("Career paths in technology", "career_guidance"),
    ("Job opportunities in AI", "career_guidance"),
    ("Should I pursue ML engineering", "career_guidance"),
    ("Fresher job advice", "career_guidance"),
    ("Career roadmap for developers", "career_guidance"),
    ("Software engineer vs data scientist", "career_guidance"),
    ("Best tech career for beginners", "career_guidance"),
    ("How to switch to tech career", "career_guidance"),
    ("Is cybersecurity a good career", "career_guidance"),
    ("Career prospects in cloud computing", "career_guidance"),
    ("Should I do MBA or MTech", "career_guidance"),
    ("How to get into FAANG companies", "career_guidance"),
    
    # Concept Explanation (15 examples)
    ("Explain machine learning", "concept_explanation"),
    ("What is artificial intelligence", "concept_explanation"),
    ("Difference between ML and DL", "concept_explanation"),
    ("How does neural network work", "concept_explanation"),
    ("What is deep learning", "concept_explanation"),
    ("Explain supervised learning", "concept_explanation"),
    ("What is NLP", "concept_explanation"),
    ("Computer vision basics", "concept_explanation"),
    ("What are transformers in AI", "concept_explanation"),
    ("Explain reinforcement learning", "concept_explanation"),
    ("What is LLM", "concept_explanation"),
    ("Difference between AI and ML", "concept_explanation"),
    ("How does ChatGPT work", "concept_explanation"),
    ("What is prompt engineering", "concept_explanation"),
    ("Explain cloud computing", "concept_explanation"),
    
    # Skill Recommendation (10 examples)
    ("How to learn Python", "skill_recommendation"),
    ("Skills needed for AI jobs", "skill_recommendation"),
    ("Best programming language to learn", "skill_recommendation"),
    ("Learning path for data science", "skill_recommendation"),
    ("Should I learn React or Angular", "skill_recommendation"),
    ("Essential skills for developers", "skill_recommendation"),
    ("How to become an AI engineer", "skill_recommendation"),
    ("Roadmap for machine learning", "skill_recommendation"),
    ("What skills for data analyst", "skill_recommendation"),
    ("Must learn tools for AI", "skill_recommendation"),
    
    # Project Guidance (10 examples)
    ("Project ideas for resume", "project_guidance"),
    ("How to build portfolio", "project_guidance"),
    ("Beginner AI projects", "project_guidance"),
    ("Good projects for students", "project_guidance"),
    ("ML project for beginners", "project_guidance"),
    ("Portfolio projects for data science", "project_guidance"),
    ("Real world AI project ideas", "project_guidance"),
    ("Projects to showcase skills", "project_guidance"),
    ("Easy Python projects for beginners", "project_guidance"),
    ("Web development project ideas", "project_guidance"),
]

X_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]

intent_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=600)),
    ("classifier", LogisticRegression(max_iter=1500, C=1.2))
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
    """Detect if conversation is focused on a specific topic"""
    text = " ".join(context_text).lower()
    
    if text.count("ai") + text.count("data science") >= 3:
        return "ai_vs_data_science"
    if text.count("career") >= 2:
        return "career_focus"
    if text.count("study") + text.count("exam") >= 2:
        return "study_focus"
    if text.count("project") >= 2:
        return "project_focus"
    
    return None

def predict_intent(user_input):
    """Predict user intent with confidence threshold"""
    try:
        probabilities = intent_classifier.predict_proba([user_input])[0]
        max_prob = max(probabilities)
        
        # Lower threshold for better coverage
        if max_prob < 0.30:
            return "general_query"
        
        return intent_classifier.predict([user_input])[0]
    except:
        return "general_query"

# ----------------------------
# Core AI Agent Function
# ----------------------------
def ai_student_agent(user_input, session_id="default_user"):
    """Main AI agent with enhanced formatting and structure"""
    session = user_sessions[session_id]
    session.append(user_input)
    
    # Keep last 6 messages for context
    session = session[-6:]
    user_sessions[session_id] = session
    
    intent = predict_intent(user_input)
    topic_lock = detect_topic_lock(session)
    context_summary = "\n".join(session[-4:])  # Use last 4 for context
    
    system_prompt = """You are an expert AI Student Advisor with 10+ years of experience mentoring students in technology, AI, career development, and academic success.

YOUR CORE RESPONSIBILITIES:
- Provide honest career guidance for students (tech/AI/data science/software)
- Explain complex technical concepts in simple, relatable terms with real examples
- Give actionable study advice based on cognitive science and proven techniques
- Recommend skills, learning paths, and resources with realistic timelines
- Suggest practical projects that demonstrate real skills to employers
- Be encouraging but realistic about challenges and time commitments
- Help students make informed decisions, not just feel good

CRITICAL FORMATTING RULES (MUST FOLLOW EVERY TIME):
1. **Use bold (double asterisks) for ALL key points, headings, and important terms**
2. Structure with SHORT paragraphs (2-3 lines maximum each)
3. Use bullet points (single • character) for ALL lists, steps, and options
4. Add blank lines between sections for breathing room
5. Keep total response 200-280 words (concise but complete)
6. Start with direct answer, then elaborate
7. End with an encouraging but honest closing line

TONE & PERSONALITY:
- Friendly senior mentor, not a salesperson or motivational speaker
- Honest about tradeoffs and realistic timelines
- Use "you" to make it personal and engaging
- Explain jargon when used, avoid unnecessary complexity
- Use 1-2 emojis maximum (only if genuinely appropriate)
- Balance optimism with pragmatism

ANSWER STRUCTURE (FOLLOW THIS):
**[Direct Answer]**: Clear 1-2 sentence response to their question.

**[Key Section 1 Heading]**:
Brief explanation (2-3 lines max) with **important terms** in bold.

**[Key Section 2 Heading]**:
• Bullet point with actionable detail
• Another bullet point with specific info
• Third bullet point if needed

**[Optional Section 3]**:
Additional context or recommendations.

[Encouraging but realistic closing line.]

EXAMPLE OF PERFECT FORMAT:
**Career Choice**: Both AI and Data Science are excellent for 2026, and they overlap significantly!

**Key Differences**:
• **AI Engineering**: Build intelligent systems, work with LLMs, neural networks, and deployment at scale
• **Data Science**: Extract insights from data, create predictive models, communicate findings to stakeholders

**My Recommendation**:
Start with **Python fundamentals** and **basic statistics** (2-3 months). Then try a small project in each field—you'll naturally gravitate toward one. Many professionals blend both skills anyway.

**Reality Check**: Entry roles need 6-12 months of focused learning. Start building projects now, they matter more than certificates.

You're asking the right questions—keep that curiosity! 🚀

IMPORTANT NOTES:
- Always format bullet points with single • character (not multiple characters)
- Always use **double asterisks** for bold
- Never skip the structure above
- Be specific with numbers, timelines, and resources when relevant"""

    # Intent-specific guidance
    intent_guidance = {
        "study_guidance": """Give evidence-based study techniques (spaced repetition, active recall, pomodoro).
Be specific about implementation. Mention realistic time management. Avoid generic advice.""",
        
        "career_guidance": """Provide honest career advice with required skills AND realistic timelines.
Mention current job market realities. Compare options fairly. Give actionable next steps.
Mention salary ranges if relevant. Be encouraging but don't sugarcoat challenges.""",
        
        "concept_explanation": """Explain using simple analogies and real-world examples.
Break complex ideas into digestible chunks. Bold all technical terms. Use 2-3 bullet points for key aspects.
Relate to things students already understand.""",
        
        "skill_recommendation": """Give clear learning roadmap with time estimates.
Prioritize essential skills first. Mention 2-3 specific free resources if helpful.
Realistic time commitment (hours/weeks). Balance depth vs breadth.""",
        
        "project_guidance": """Suggest 2-3 specific, practical projects with difficulty levels.
Explain what skills each teaches and how it looks on resume. Mention typical time to build.
Focus on projects that demonstrate real competency, not just tutorials.""",
        
        "general_query": """Answer helpfully with clear structure. Use bold headings and bullets.
Be specific rather than vague. Give actionable takeaways."""
    }
    
    user_prompt = f"""Previous conversation context (last 4 messages):
{context_summary}

Current question intent: {intent}
Conversation topic focus: {topic_lock if topic_lock else "None - fresh topic"}

Guidance for this intent type:
{intent_guidance.get(intent, "")}

User's current question:
"{user_input}"

CRITICAL REMINDERS:
- Use **bold** (double asterisks) for all key points and headings
- Use single • character for bullet points
- Keep paragraphs SHORT (2-3 lines maximum)
- Be specific with numbers, timelines, resources
- Add encouraging but realistic tone
- Follow the structure in your system prompt exactly"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.7,
            top_p=0.92,
            frequency_penalty=0.3,
            presence_penalty=0.1
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            return "**Error**: Invalid or missing OpenAI API key. Please check your .env file!\n\nMake sure you have:\n`OPENAI_API_KEY=sk-proj-your-key-here`"
        elif "rate_limit" in error_msg.lower():
            return "**Rate Limited**: Too many requests. Please wait a moment and try again."
        else:
            return f"**Error**: I encountered a technical issue: {error_msg}\n\nPlease try again or rephrase your question!"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# ENHANCED Intent Classification (60+ examples)
# ----------------------------
training_data = [
    # Study Guidance (15 examples)
    ("How can I prepare for exams?", "study_guidance"),
    ("How do I study better?", "study_guidance"),
    ("Best study techniques", "study_guidance"),
    ("Time management for students", "study_guidance"),
    ("How to focus while studying", "study_guidance"),
    ("Study schedule tips", "study_guidance"),
    ("Memory improvement techniques", "study_guidance"),
    ("How to take better notes", "study_guidance"),
    ("Effective study methods", "study_guidance"),
    ("How to avoid procrastination", "study_guidance"),
    ("I can't concentrate on studies", "study_guidance"),
    ("How to prepare for competitive exams", "study_guidance"),
    ("Best apps for studying", "study_guidance"),
    ("How to revise effectively", "study_guidance"),
    ("Tips for last minute exam prep", "study_guidance"),
    
    # Career Guidance (15 examples)
    ("I am confused about my career", "career_guidance"),
    ("AI or Data Science which is better", "career_guidance"),
    ("Which career should I choose", "career_guidance"),
    ("Career paths in technology", "career_guidance"),
    ("Job opportunities in AI", "career_guidance"),
    ("Should I pursue ML engineering", "career_guidance"),
    ("Fresher job advice", "career_guidance"),
    ("Career roadmap for developers", "career_guidance"),
    ("Software engineer vs data scientist", "career_guidance"),
    ("Best tech career for beginners", "career_guidance"),
    ("How to switch to tech career", "career_guidance"),
    ("Is cybersecurity a good career", "career_guidance"),
    ("Career prospects in cloud computing", "career_guidance"),
    ("Should I do MBA or MTech", "career_guidance"),
    ("How to get into FAANG companies", "career_guidance"),
    
    # Concept Explanation (15 examples)
    ("Explain machine learning", "concept_explanation"),
    ("What is artificial intelligence", "concept_explanation"),
    ("Difference between ML and DL", "concept_explanation"),
    ("How does neural network work", "concept_explanation"),
    ("What is deep learning", "concept_explanation"),
    ("Explain supervised learning", "concept_explanation"),
    ("What is NLP", "concept_explanation"),
    ("Computer vision basics", "concept_explanation"),
    ("What are transformers in AI", "concept_explanation"),
    ("Explain reinforcement learning", "concept_explanation"),
    ("What is LLM", "concept_explanation"),
    ("Difference between AI and ML", "concept_explanation"),
    ("How does ChatGPT work", "concept_explanation"),
    ("What is prompt engineering", "concept_explanation"),
    ("Explain cloud computing", "concept_explanation"),
    
    # Skill Recommendation (10 examples)
    ("How to learn Python", "skill_recommendation"),
    ("Skills needed for AI jobs", "skill_recommendation"),
    ("Best programming language to learn", "skill_recommendation"),
    ("Learning path for data science", "skill_recommendation"),
    ("Should I learn React or Angular", "skill_recommendation"),
    ("Essential skills for developers", "skill_recommendation"),
    ("How to become an AI engineer", "skill_recommendation"),
    ("Roadmap for machine learning", "skill_recommendation"),
    ("What skills for data analyst", "skill_recommendation"),
    ("Must learn tools for AI", "skill_recommendation"),
    
    # Project Guidance (10 examples)
    ("Project ideas for resume", "project_guidance"),
    ("How to build portfolio", "project_guidance"),
    ("Beginner AI projects", "project_guidance"),
    ("Good projects for students", "project_guidance"),
    ("ML project for beginners", "project_guidance"),
    ("Portfolio projects for data science", "project_guidance"),
    ("Real world AI project ideas", "project_guidance"),
    ("Projects to showcase skills", "project_guidance"),
    ("Easy Python projects for beginners", "project_guidance"),
    ("Web development project ideas", "project_guidance"),
]

X_train = [x[0] for x in training_data]
y_train = [x[1] for x in training_data]

intent_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=600)),
    ("classifier", LogisticRegression(max_iter=1500, C=1.2))
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
    """Detect if conversation is focused on a specific topic"""
    text = " ".join(context_text).lower()
    
    if text.count("ai") + text.count("data science") >= 3:
        return "ai_vs_data_science"
    if text.count("career") >= 2:
        return "career_focus"
    if text.count("study") + text.count("exam") >= 2:
        return "study_focus"
    if text.count("project") >= 2:
        return "project_focus"
    
    return None

def predict_intent(user_input):
    """Predict user intent with confidence threshold"""
    try:
        probabilities = intent_classifier.predict_proba([user_input])[0]
        max_prob = max(probabilities)
        
        # Lower threshold for better coverage
        if max_prob < 0.30:
            return "general_query"
        
        return intent_classifier.predict([user_input])[0]
    except:
        return "general_query"

# ----------------------------
# Core AI Agent Function
# ----------------------------
def ai_student_agent(user_input, session_id="default_user"):
    """Main AI agent with enhanced formatting and structure"""
    session = user_sessions[session_id]
    session.append(user_input)
    
    # Keep last 6 messages for context
    session = session[-6:]
    user_sessions[session_id] = session
    
    intent = predict_intent(user_input)
    topic_lock = detect_topic_lock(session)
    context_summary = "\n".join(session[-4:])  # Use last 4 for context
    
    system_prompt = """You are an expert AI Student Advisor with 10+ years of experience mentoring students in technology, AI, career development, and academic success.

YOUR CORE RESPONSIBILITIES:
- Provide honest career guidance for students (tech/AI/data science/software)
- Explain complex technical concepts in simple, relatable terms with real examples
- Give actionable study advice based on cognitive science and proven techniques
- Recommend skills, learning paths, and resources with realistic timelines
- Suggest practical projects that demonstrate real skills to employers
- Be encouraging but realistic about challenges and time commitments
- Help students make informed decisions, not just feel good

CRITICAL FORMATTING RULES (MUST FOLLOW EVERY TIME):
1. **Use bold (double asterisks) for ALL key points, headings, and important terms**
2. Structure with SHORT paragraphs (2-3 lines maximum each)
3. Use bullet points (single • character) for ALL lists, steps, and options
4. Add blank lines between sections for breathing room
5. Keep total response 200-280 words (concise but complete)
6. Start with direct answer, then elaborate
7. End with an encouraging but honest closing line

TONE & PERSONALITY:
- Friendly senior mentor, not a salesperson or motivational speaker
- Honest about tradeoffs and realistic timelines
- Use "you" to make it personal and engaging
- Explain jargon when used, avoid unnecessary complexity
- Use 1-2 emojis maximum (only if genuinely appropriate)
- Balance optimism with pragmatism

ANSWER STRUCTURE (FOLLOW THIS):
**[Direct Answer]**: Clear 1-2 sentence response to their question.

**[Key Section 1 Heading]**:
Brief explanation (2-3 lines max) with **important terms** in bold.

**[Key Section 2 Heading]**:
• Bullet point with actionable detail
• Another bullet point with specific info
• Third bullet point if needed

**[Optional Section 3]**:
Additional context or recommendations.

[Encouraging but realistic closing line.]

EXAMPLE OF PERFECT FORMAT:
**Career Choice**: Both AI and Data Science are excellent for 2026, and they overlap significantly!

**Key Differences**:
• **AI Engineering**: Build intelligent systems, work with LLMs, neural networks, and deployment at scale
• **Data Science**: Extract insights from data, create predictive models, communicate findings to stakeholders

**My Recommendation**:
Start with **Python fundamentals** and **basic statistics** (2-3 months). Then try a small project in each field—you'll naturally gravitate toward one. Many professionals blend both skills anyway.

**Reality Check**: Entry roles need 6-12 months of focused learning. Start building projects now, they matter more than certificates.

You're asking the right questions—keep that curiosity! 🚀

IMPORTANT NOTES:
- Always format bullet points with single • character (not multiple characters)
- Always use **double asterisks** for bold
- Never skip the structure above
- Be specific with numbers, timelines, and resources when relevant"""

    # Intent-specific guidance
    intent_guidance = {
        "study_guidance": """Give evidence-based study techniques (spaced repetition, active recall, pomodoro).
Be specific about implementation. Mention realistic time management. Avoid generic advice.""",
        
        "career_guidance": """Provide honest career advice with required skills AND realistic timelines.
Mention current job market realities. Compare options fairly. Give actionable next steps.
Mention salary ranges if relevant. Be encouraging but don't sugarcoat challenges.""",
        
        "concept_explanation": """Explain using simple analogies and real-world examples.
Break complex ideas into digestible chunks. Bold all technical terms. Use 2-3 bullet points for key aspects.
Relate to things students already understand.""",
        
        "skill_recommendation": """Give clear learning roadmap with time estimates.
Prioritize essential skills first. Mention 2-3 specific free resources if helpful.
Realistic time commitment (hours/weeks). Balance depth vs breadth.""",
        
        "project_guidance": """Suggest 2-3 specific, practical projects with difficulty levels.
Explain what skills each teaches and how it looks on resume. Mention typical time to build.
Focus on projects that demonstrate real competency, not just tutorials.""",
        
        "general_query": """Answer helpfully with clear structure. Use bold headings and bullets.
Be specific rather than vague. Give actionable takeaways."""
    }
    
    user_prompt = f"""Previous conversation context (last 4 messages):
{context_summary}

Current question intent: {intent}
Conversation topic focus: {topic_lock if topic_lock else "None - fresh topic"}

Guidance for this intent type:
{intent_guidance.get(intent, "")}

User's current question:
"{user_input}"

CRITICAL REMINDERS:
- Use **bold** (double asterisks) for all key points and headings
- Use single • character for bullet points
- Keep paragraphs SHORT (2-3 lines maximum)
- Be specific with numbers, timelines, resources
- Add encouraging but realistic tone
- Follow the structure in your system prompt exactly"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=400,
            temperature=0.7,
            top_p=0.92,
            frequency_penalty=0.3,
            presence_penalty=0.1
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            return "**Error**: Invalid or missing OpenAI API key. Please check your .env file!\n\nMake sure you have:\n`OPENAI_API_KEY=sk-proj-your-key-here`"
        elif "rate_limit" in error_msg.lower():
            return "**Rate Limited**: Too many requests. Please wait a moment and try again."
        else:
            return f"**Error**: I encountered a technical issue: {error_msg}\n\nPlease try again or rephrase your question!"