import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    app_name: str = "Question Generator API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database Settings
    database_url: str = Field(
        default="postgresql://user:password@localhost/question_generator",
        env="DATABASE_URL"
    )
    
    # Qdrant Settings
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field(
        default="question_generator_docs", 
        env="QDRANT_COLLECTION_NAME"
    )
    
    # Model Settings
    embedding_model_name: str = Field(
        default="intfloat/multilingual-e5-large",
        env="EMBEDDING_MODEL_NAME"
    )
    language_model_name: str = Field(
        default="google/gemma-3-270m",  # Can change to gemma-3-270m if needed
        env="LANGUAGE_MODEL_NAME"
    )
    whisper_model_name: str = Field(
        default="small",
        env="WHISPER_MODEL_NAME"
    )
    
    # Text Processing Settings
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Question Generation Settings
    default_num_questions: int = Field(default=5, env="DEFAULT_NUM_QUESTIONS")
    default_difficulty: str = Field(default="medium", env="DEFAULT_DIFFICULTY")
    default_context_limit: int = Field(default=3, env="DEFAULT_CONTEXT_LIMIT")
    
    supported_question_types: List[str] = [
        "multiple_choice",
        "short_answer", 
        "essay",
        "true_false",
        "fill_blank"
    ]
    
    supported_document_types: List[str] = ["pdf", "docx", "doc", "txt"]
    supported_audio_types: List[str] = ["mp3", "wav", "m4a", "flac", "ogg"]
    
    # File Upload Settings
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    
    # Security Settings
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Model configuration for different environments
MODEL_CONFIGS = {
    "development": {
        "language_model": "google/gemma-3-270m",
        "device_map": "auto",
        "torch_dtype": "auto",
        "max_new_tokens": 512,
        "temperature": 0.7
    },
    "production": {
        "language_model": "google/gemma-3-270m",  # Can upgrade to this
        "device_map": "auto", 
        "torch_dtype": "float16",
        "max_new_tokens": 1024,
        "temperature": 0.6
    }
}

# Question generation prompts templates
QUESTION_PROMPTS = {
    "multiple_choice_vi": """
Dựa trên nội dung sau, hãy tạo {count} câu hỏi trắc nghiệm ({difficulty}) với 4 lựa chọn A, B, C, D và đáp án đúng.

Nội dung:
{context}

Định dạng trả về:
Câu hỏi: [câu hỏi]
A. [lựa chọn A]
B. [lựa chọn B] 
C. [lựa chọn C]
D. [lựa chọn D]
Đáp án: [A/B/C/D]
Giải thích: [giải thích ngắn gọn]
---
""",
    
    "short_answer_vi": """
Dựa trên nội dung sau, hãy tạo {count} câu hỏi ngắn ({difficulty}) yêu cầu trả lời trong 1-2 câu.

Nội dung:
{context}

Định dạng trả về:
Câu hỏi: [câu hỏi]
Đáp án mẫu: [đáp án mẫu]
---
""",
    
    "essay_vi": """
Dựa trên nội dung sau, hãy tạo {count} câu hỏi tự luận ({difficulty}) yêu cầu trả lời chi tiết.

Nội dung:
{context}

Định dạng trả về:
Câu hỏi: [câu hỏi]
Gợi ý trả lời: [các điểm chính cần đề cập]
---
""",
    
    "true_false_vi": """
Dựa trên nội dung sau, hãy tạo {count} câu hỏi đúng/sai ({difficulty}).

Nội dung:
{context}

Định dạng trả về:
Câu hỏi: [câu hỏi]
Đáp án: [Đúng/Sai]
Giải thích: [giải thích]
---
""",
    
    "fill_blank_vi": """
Dựa trên nội dung sau, hãy tạo {count} câu hỏi điền vào chỗ trống ({difficulty}).

Nội dung:
{context}

Định dạng trả về:
Câu hỏi: [câu có chỗ trống với ___]
Đáp án: [từ/cụm từ cần điền]
---
"""
}

# Export commonly used settings
__all__ = [
    "settings", 
    "MODEL_CONFIGS", 
    "QUESTION_PROMPTS"
]
