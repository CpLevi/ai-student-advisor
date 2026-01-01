# 🎓 AI Student Advisor

An intelligent, AI-powered chatbot designed to help students with **career guidance**, **study techniques**, **tech concepts**, **skill roadmaps**, and **project recommendations**.

## ✨ Features

### 🤖 **Advanced AI Capabilities**
- **60+ training examples** for accurate intent classification
- **Context-aware conversations** (remembers last 6 messages)
- **Topic lock detection** for focused discussions
- **Session management** with unique user IDs
- **Smart error handling** with specific error messages

### 🎨 **Premium UI/UX**
- **Glassmorphism design** with 40px backdrop blur
- **Digital lavender & teal gradient** color palette
- **Spring physics animations** for message bubbles
- **Thermal glow effect** for typing indicator
- **Word-reveal animations** for AI responses
- **Dynamic Island style** input with focus states
- **Custom scrollbar** (hidden until hover)
- **6 quick action chips** for common queries

### 📚 **Student-Focused Content**
- Career guidance (AI, Data Science, Software Engineering, etc.)
- Study techniques (memory, focus, time management)
- Tech concept explanations (ML, AI, NLP, etc.)
- Skill recommendations with learning roadmaps
- Project ideas for portfolios
- Realistic timelines and honest advice

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- pip package manager

### Installation

1. **Clone or create project folder**
```bash
mkdir ai-student-advisor
cd ai-student-advisor
```

2. **Create the required files**
- `app.py` (FastAPI server)
- `ai_student_advisor.py` (AI brain)
- `requirements.txt` (dependencies)
- `templates/` folder
- `templates/index.html` (UI)
- `.env` (API key)

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create `.env` file with your OpenAI key**
```bash
echo "OPENAI_API_KEY=sk-proj-your-actual-key-here" > .env
```

5. **Run the server**
```bash
python app.py
```

6. **Open in browser**
```
http://localhost:8000
```

## 📁 Project Structure

```
ai-student-advisor/
├── app.py                      # FastAPI server with endpoints
├── ai_student_advisor.py       # AI agent with ML classification
├── requirements.txt            # Python dependencies
├── .env                        # OpenAI API key (create this)
├── templates/
│   └── index.html             # Premium glassmorphic UI
└── README.md                  # This file
```

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern, fast web framework
- **OpenAI GPT-4o-mini** - Latest AI model for responses
- **scikit-learn** - Intent classification with Logistic Regression
- **TF-IDF Vectorizer** - Text feature extraction

### Frontend
- **Vanilla JavaScript** - No frameworks, pure performance
- **CSS3** - Glassmorphism, gradients, animations
- **Inter Font** - Modern typography
- **Responsive Design** - Works on all devices

## 🎯 Key Features Explained

### Intent Classification
The system classifies user queries into 5 categories:
1. **study_guidance** - Study techniques, time management, exam prep
2. **career_guidance** - Career paths, job advice, industry insights
3. **concept_explanation** - Technical concepts, AI/ML terms
4. **skill_recommendation** - Learning paths, roadmaps, tools
5. **project_guidance** - Portfolio projects, resume builders

### Session Management
- Each user gets a unique session ID
- Remembers last 6 messages for context
- Detects topic locks (career_focus, study_focus, etc.)
- Can be reset via `/reset` endpoint

### Formatting System
AI responses automatically format:
- **Bold text** for key points and headings
- Bullet points (•) for lists
- Short paragraphs (2-3 lines max)
- Structured sections
- Encouraging closing lines

## 🔧 Configuration

### Model Parameters
```python
model="gpt-4o-mini"           # Latest OpenAI model
max_tokens=400                # Response length
temperature=0.7               # Creativity (0-1)
top_p=0.92                    # Nucleus sampling
frequency_penalty=0.3         # Reduce repetition
presence_penalty=0.1          # Encourage variety
```

### Intent Classification Threshold
```python
if max_prob < 0.30:
    return "general_query"
```
Lower = More sensitive, Higher = More strict

## 📊 API Endpoints

### `GET /`
Serves the main chat interface

### `POST /chat`
**Request:**
```json
{
  "message": "How do I learn Python?",
  "session_id": "user_abc123"
}
```

**Response:**
```json
{
  "response": "**Learning Python**: Great choice! Here's your roadmap...",
  "session_id": "user_abc123"
}
```

### `POST /reset`
Reset user session history
```json
{
  "session_id": "user_abc123"
}
```

### `GET /health`
Health check endpoint
```json
{
  "status": "healthy",
  "service": "AI Student Advisor",
  "version": "2.0.0"
}
```

### `GET /docs`
Auto-generated FastAPI documentation

## 🎨 Customization

### Change Color Scheme
Edit CSS variables in `templates/index.html`:
```css
:root {
  --lavender-deep: #9333ea;  /* Primary color */
  --teal: #14b8a6;          /* Accent color */
  --glass-bg: rgba(255, 255, 255, 0.65);  /* Background opacity */
}
```

### Add More Training Data
Edit `ai_student_advisor.py`:
```python
training_data = [
    ("Your new example", "category_name"),
    # Add more examples here
]
```

### Adjust AI Personality
Edit the `system_prompt` in `ai_student_advisor.py` to change:
- Tone (friendly, formal, casual)
- Response structure
- Level of detail
- Encouragement style

## ⚡ Performance Tips

1. **Use environment variables for production**
```python
port = int(os.environ.get("PORT", 8000))
```

2. **Enable caching for repeated queries**
3. **Rate limit API calls** (built-in via OpenAI)
4. **Monitor token usage** (currently ~300-400 tokens per response)

## 🐛 Troubleshooting

### "Module not found: ai_student_advisor"
**Solution:** Make sure `ai_student_advisor.py` is in the same directory as `app.py`

### "OpenAI API key error"
**Solution:** Check your `.env` file:
```bash
cat .env  # Should show: OPENAI_API_KEY=sk-proj-...
```

### "Connection refused" or "Load failed"
**Solution:** Make sure the server is running:
```bash
python app.py  # Should show server startup message
```

### Bullet points showing as weird characters
**Solution:** File encoding issue. Save `ai_student_advisor.py` as UTF-8

### UI not loading
**Solution:** Make sure `index.html` is inside `templates/` folder

## 📈 Future Enhancements

- [ ] Add voice input/output
- [ ] Save conversation history to database
- [ ] Add multi-language support
- [ ] Integrate with calendar for study scheduling
- [ ] Add user authentication
- [ ] Export chat history as PDF
- [ ] Add code execution for programming help
- [ ] Integrate with learning platforms (Coursera, Udemy)

## 📄 License

MIT License - Free to use for personal and educational projects

## 🤝 Contributing

Feel free to:
- Add more training examples
- Improve UI/UX
- Add new features
- Fix bugs
- Improve documentation

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review FastAPI logs for errors
3. Verify OpenAI API key is valid
4. Check that all files are in correct locations

## 🎓 Educational Use

Perfect for:
- Learning FastAPI and modern web development
- Understanding AI integration in applications
- Studying UI/UX design patterns
- Exploring ML classification with scikit-learn
- Building portfolio projects

---

**Made with ❤️ for students, by students**

*Last updated: January 2026*
