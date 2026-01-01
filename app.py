from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_student_advisor import ai_student_agent
import os

app = FastAPI(
    title="AI Student Advisor",
    description="Intelligent AI-powered student advisor for career, study, and tech guidance",
    version="2.0.0"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class UserQuery(BaseModel):
    message: str
    session_id: str = "default_user"

class ResetRequest(BaseModel):
    session_id: str = "default_user"

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(query: UserQuery):
    """Handle chat requests with AI agent"""
    try:
        if not query.message or len(query.message.strip()) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )
        
        # Limit message length to prevent abuse
        if len(query.message) > 1000:
            return JSONResponse(
                status_code=400,
                content={"error": "Message too long (max 1000 characters)"}
            )
        
        reply = ai_student_agent(query.message, query.session_id)
        return {"response": reply, "session_id": query.session_id}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Server error: {str(e)}",
                "response": "**Error**: I'm having trouble right now. Please try again in a moment!"
            }
        )

@app.post("/reset")
async def reset_session(data: ResetRequest):
    """Reset user session history"""
    try:
        from ai_student_advisor import user_sessions
        user_sessions[data.session_id] = []
        return {
            "status": "success",
            "message": "Session history cleared successfully"
        }
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
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         🎓 AI Student Advisor Server                     ║
║                                                          ║
║  Server running on: http://localhost:{port}              ║
║  Open in browser:   http://localhost:{port}              ║
║                                                          ║
║  Health check:      http://localhost:{port}/health       ║
║  API docs:          http://localhost:{port}/docs         ║
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