from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from ai_student_advisor import ai_student_agent
import os

app = FastAPI(title="AI Student Advisor")
templates = Jinja2Templates(directory="templates")

class UserQuery(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
def chat(query: UserQuery):
    reply = ai_student_agent(query.message)
    return {"response": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
