from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_student_advisor import ai_student_agent
import os
import base64
from datetime import datetime
import json
from typing import Optional

app = FastAPI(
    title="AI Student Advisor v2.1",
    description="Advanced AI advisor with image analysis, resume review, and quiz generation",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# In-memory session storage (replace with database in production)
sessions_db = {}

class UserQuery(BaseModel):
    message: str
    session_id: str = "default_user"
    mode: Optional[str] = "chat"  # chat, resume, quiz

class SessionData(BaseModel):
    session_id: str
    title: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(query: UserQuery):
    """Handle regular chat requests"""
    try:
        if not query.message or len(query.message.strip()) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        
        if len(query.message) > 2000:
            return JSONResponse(
                status_code=400,
                content={"error": "Message too long (max 2000 characters)"}
            )
        
        # Get AI response
        reply = ai_student_agent(query.message, query.session_id)
        
        # Store in session
        if query.session_id not in sessions_db:
            sessions_db[query.session_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "title": query.message[:60]
            }
        
        sessions_db[query.session_id]["messages"].append({
            "role": "user",
            "content": query.message,
            "timestamp": datetime.now().isoformat()
        })
        sessions_db[query.session_id]["messages"].append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "response": reply,
            "session_id": query.session_id,
            "message_count": len(sessions_db[query.session_id]["messages"])
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Server error: {str(e)}",
                "response": "**Error**: I'm having trouble right now. Please try again!"
            }
        )

@app.post("/upload-image")
async def upload_image(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    prompt: str = Form("Analyze this image")
):
    """Handle image uploads and analysis"""
    try:
        # Read image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Analyze with GPT-4o (vision model)
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an AI Student Advisor analyzing an image.

Context: {prompt}

Provide helpful analysis for students:
- If it's a diagram/whiteboard: Explain concepts clearly
- If it's homework: Guide thinking, don't give direct answers
- If it's code: Review and suggest improvements
- If it's notes: Summarize key points

Use **bold** for key terms and • bullets for lists."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "response": analysis,
            "image_processed": True,
            "filename": file.filename
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Image analysis error: {str(e)}"}
        )

@app.post("/analyze-resume")
async def analyze_resume(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    target_role: str = Form("Software Engineer")
):
    """Analyze resume and provide feedback"""
    try:
        contents = await file.read()
        text_content = contents.decode('utf-8', errors='ignore')
        
        # Analyze resume with specialized prompt
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert career advisor and resume reviewer with 10+ years of experience.

Provide structured feedback on resumes:

**Overall Score**: X/10

**Strengths**:
• Point 1
• Point 2

**Areas for Improvement**:
• Specific issue and how to fix
• Another improvement

**Action Items**:
• Concrete step 1
• Concrete step 2

Be honest but encouraging. Focus on actionable improvements."""
                },
                {
                    "role": "user",
                    "content": f"""Target Role: {target_role}

Resume Content:
{text_content[:3000]}

Provide comprehensive feedback."""
                }
            ],
            max_tokens=600,
            temperature=0.7
        )
        
        feedback = response.choices[0].message.content
        
        return {
            "response": feedback,
            "resume_analyzed": True,
            "target_role": target_role,
            "filename": file.filename
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Resume analysis error: {str(e)}"}
        )

@app.post("/generate-quiz")
async def generate_quiz(query: UserQuery):
    """Generate a quiz on any topic"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a quiz generator for students. Create engaging, educational quizzes.

Format EXACTLY like this:

**Quiz: [Topic]** (5 questions)

**Q1**: Question text here?
A) Option A
B) Option B
C) Option C
D) Option D
*Correct: A*
*Explanation*: Brief explanation why.

**Q2**: Next question?
...

Make questions:
- Clear and unambiguous
- Educational and relevant
- Progressive difficulty
- Include explanations"""
                },
                {
                    "role": "user",
                    "content": f"Create a 5-question quiz about: {query.message}"
                }
            ],
            max_tokens=800,
            temperature=0.8
        )
        
        quiz = response.choices[0].message.content
        
        return {
            "response": quiz,
            "quiz_generated": True,
            "topic": query.message
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Quiz generation error: {str(e)}"}
        )

@app.get("/sessions")
async def get_sessions(session_id: str):
    """Get all sessions for a user"""
    try:
        user_sessions = {
            sid: {
                "title": data["title"],
                "created_at": data["created_at"],
                "message_count": len(data["messages"])
            }
            for sid, data in sessions_db.items()
            if sid.startswith(session_id.split('_')[0])
        }
        
        return {"sessions": user_sessions}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get specific session data"""
    try:
        if session_id not in sessions_db:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found"}
            )
        
        return {
            "session": sessions_db[session_id]
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    try:
        if session_id in sessions_db:
            del sessions_db[session_id]
            return {"status": "deleted", "session_id": session_id}
        return JSONResponse(
            status_code=404,
            content={"error": "Session not found"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/reset")
async def reset_session(data: SessionData):
    """Reset user session"""
    try:
        from ai_student_advisor import user_sessions
        user_sessions[data.session_id] = []
        if data.session_id in sessions_db:
            sessions_db[data.session_id]["messages"] = []
        return {"status": "success", "message": "Session reset"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Student Advisor",
        "version": "2.1.0",
        "features": [
            "chat",
            "image_analysis",
            "resume_review",
            "quiz_generation",
            "session_history"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         🎓 AI Student Advisor v2.1 Server                ║
║                                                          ║
║  🚀 New Features:                                        ║
║     • Image Upload & Analysis                            ║
║     • Resume Analyzer                                    ║
║     • Quiz Generator                                     ║
║     • Session History                                    ║
║                                                          ║
║  Server: http://localhost:{port}                         ║
║  Docs:   http://localhost:{port}/docs                    ║
║  Health: http://localhost:{port}/health                  ║
║                                                          ║
║  Press CTRL+C to stop                                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )