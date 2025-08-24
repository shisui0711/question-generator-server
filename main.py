import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
import uvicorn
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import Session, select, SQLModel
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import tempfile
import os
from pydantic import BaseModel

from database import engine, get_db
from models import User, UserCreate, UserResponse, UserLogin
from auth import get_password_hash, verify_password, create_access_token, verify_token
from services import QuestionGenerationService

SQLModel.metadata.create_all(engine)

app = FastAPI(
    title="Question Generator API",
    description="API for Question Generator",
    version="1.0.0"
)

security = HTTPBearer()

# Initialize Question Generation Service
question_service = QuestionGenerationService()

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class UrlInput(BaseModel):
    url: str
    metadata: Optional[Dict[str, Any]] = None

class QuestionGenerationRequest(BaseModel):
    query: str = ""
    num_questions: int = 5
    question_types: Optional[List[str]] = None
    difficulty_level: str = "medium"
    context_limit: int = 3

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    username = verify_token(credentials.credentials)
    user = db.exec(select(User).where(User.username == username)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if username exists
    db_user = db.exec(select(User).where(User.username == user.username)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email exists
    db_user = db.exec(select(User).where(User.email == user.email)).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        role='user'
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login")
def login(credentials:UserLogin, db: Session = Depends(get_db)):
    user = db.exec(select(User).where(User.username == credentials.username)).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/my-info")
def my_info(current_user: User = Depends(get_current_user)):
    return current_user

# QUESTION GENERATION ENDPOINTS

@app.post("/process-text")
async def process_text(
    text_input: TextInput,
    current_user: User = Depends(get_current_user)
):
    """Process text input and store in vector database"""
    try:
        documents = question_service.process_text_input(
            text_input.text, 
            text_input.metadata
        )
        return {
            "message": "Text processed successfully",
            "documents_count": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-url")
async def process_url(
    url_input: UrlInput,
    current_user: User = Depends(get_current_user)
):
    """Process URL input using web crawler"""
    try:
        documents = await question_service.process_url_input(
            url_input.url,
            url_input.metadata
        )
        return {
            "message": "URL processed successfully",
            "documents_count": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload and process document files (PDF, DOCX, TXT)"""
    try:
        # Check file type
        allowed_types = ["pdf", "docx", "doc", "txt"]
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Allowed: {allowed_types}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            documents = question_service.process_document_file(
                tmp_file_path,
                file_extension,
                {"filename": file.filename, "user_id": current_user.id}
            )
            
            return {
                "message": "Document processed successfully",
                "filename": file.filename,
                "documents_count": len(documents),
                "documents": documents
            }
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Upload and process audio files using Whisper"""
    try:
        # Check file type
        allowed_types = ["mp3", "wav", "m4a", "flac", "ogg"]
        file_extension = file.filename.split(".")[-1].lower()
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio type: {file_extension}. Allowed: {allowed_types}"
            )
        
        content = await file.read()
        documents = question_service.process_audio_file(
            content,
            file_extension,
            {"filename": file.filename, "user_id": current_user.id}
        )
        
        return {
            "message": "Audio processed successfully",
            "filename": file.filename,
            "documents_count": len(documents),
            "documents": documents
        }
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-questions")
async def generate_questions(
    request: QuestionGenerationRequest,
    current_user: User = Depends(get_current_user)
):
    """Generate questions based on processed content"""
    try:
        result = question_service.generate_questions(
            query=request.query,
            num_questions=request.num_questions,
            question_types=request.question_types,
            difficulty_level=request.difficulty_level,
            context_limit=request.context_limit
        )
        
        return {
            "message": "Questions generated successfully",
            "user_id": current_user.id,
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear-content")
async def clear_content(
    current_user: User = Depends(get_current_user)
):
    """Clear all processed content from vector store"""
    try:
        question_service.clear_vector_store()
        return {"message": "All content cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Question Generator API"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
