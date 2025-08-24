# Question Generator API

Ứng dụng tạo câu hỏi tự động sử dụng AI, hỗ trợ nhiều loại đầu vào khác nhau như văn bản, tài liệu, âm thanh và URL.

## Tính năng chính

### Xử lý đầu vào đa dạng
- **Văn bản**: Nhập trực tiếp văn bản để tạo câu hỏi
- **URL**: Crawl nội dung từ trang web
- **Tài liệu**: PDF, DOCX, TXT
- **Âm thanh**: MP3, WAV, M4A, FLAC, OGG (sử dụng Whisper)

### Loại câu hỏi được hỗ trợ
- Trắc nghiệm (Multiple Choice)
- Câu hỏi ngắn (Short Answer)
- Tự luận (Essay)
- Đúng/Sai (True/False)
- Điền vào chỗ trống (Fill in the Blank)

### Công nghệ sử dụng
- **LangChain**: Framework xử lý dữ liệu và tích hợp AI
- **Google Gemma 2-2B**: Model sinh ngôn ngữ tự nhiên
- **multilingual-E5-large**: Model embedding đa ngôn ngữ
- **Qdrant**: Vector database cho tìm kiếm ngữ nghĩa
- **OpenAI Whisper**: Chuyển đổi giọng nói thành văn bản
- **MegaParse**: Xử lý tài liệu PDF nâng cao
- **Crawl4AI**: Thu thập dữ liệu web

## Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- PostgreSQL
- Qdrant Vector Database

### Cài đặt dependencies

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Cài đặt packages
pip install -r requirements.txt
```

### Cấu hình môi trường

Tạo file `.env` và cấu hình:

```env
# Database
DATABASE_URL=postgresql://username:password@localhost/question_generator

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Security
SECRET_KEY=your-super-secret-key-here

# Models (tùy chọn)
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
LANGUAGE_MODEL_NAME=google/gemma-3-270m
WHISPER_MODEL_NAME=small

# Logging
LOG_LEVEL=INFO
```

### Khởi động Qdrant

```bash
# Sử dụng Docker
docker run -p 6333:6333 qdrant/qdrant
```

### Khởi động ứng dụng

```bash
python main.py
```

API sẽ chạy tại `http://localhost:8000`

## Sử dụng API

### Authentication

Tất cả endpoints (trừ `/register`, `/login`, `/health`) yêu cầu Bearer token:

```bash
Authorization: Bearer <your-access-token>
```

### Đăng ký và đăng nhập

#### Đăng ký
```bash
POST /register
Content-Type: application/json

{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123",
    "full_name": "Test User"
}
```

#### Đăng nhập
```bash
POST /login
Content-Type: application/x-www-form-urlencoded

username=testuser&password=password123
```

### Xử lý nội dung

#### Xử lý văn bản
```bash
POST /process-text
Content-Type: application/json
Authorization: Bearer <token>

{
    "text": "Nội dung văn bản cần tạo câu hỏi...",
    "metadata": {
        "title": "Tiêu đề tài liệu",
        "subject": "Môn học"
    }
}
```

#### Xử lý URL
```bash
POST /process-url
Content-Type: application/json
Authorization: Bearer <token>

{
    "url": "https://example.com/article",
    "metadata": {
        "category": "news"
    }
}
```

#### Upload tài liệu
```bash
POST /upload-document
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: [PDF/DOCX/TXT file]
```

#### Upload audio
```bash
POST /upload-audio
Content-Type: multipart/form-data
Authorization: Bearer <token>

file: [MP3/WAV/M4A file]
```

### Tạo câu hỏi

```bash
POST /generate-questions
Content-Type: application/json
Authorization: Bearer <token>

{
    "query": "từ khóa tìm kiếm (tùy chọn)",
    "num_questions": 5,
    "question_types": ["multiple_choice", "short_answer", "essay"],
    "difficulty_level": "medium",
    "context_limit": 3
}
```

#### Response format:
```json
{
    "message": "Questions generated successfully",
    "user_id": "user-uuid",
    "questions": [
        {
            "id": 1,
            "question": "Câu hỏi được tạo",
            "type": "multiple_choice",
            "options": ["A. Lựa chọn A", "B. Lựa chọn B", "C. Lựa chọn C", "D. Lựa chọn D"],
            "correct_answer": "A",
            "explanation": "Giải thích đáp án",
            "created_at": "2025-01-23T10:30:00"
        }
    ],
    "context_sources": [
        {
            "content": "Nội dung ngữ cảnh...",
            "metadata": {"source": "document", "filename": "test.pdf"}
        }
    ],
    "total_generated": 5
}
```

### Quản lý nội dung

#### Xóa tất cả nội dung
```bash
DELETE /clear-content
Authorization: Bearer <token>
```

#### Health check
```bash
GET /health
```

## Cấu trúc dự án

```
├── main.py              # FastAPI application và endpoints
├── services.py          # Business logic và AI services
├── models.py            # Database models
├── database.py          # Database configuration
├── auth.py              # Authentication utilities
├── config.py            # Application configuration
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
└── README.md           # Documentation
```

## Tùy chỉnh

### Thay đổi models

Trong file `config.py`, bạn có thể thay đổi các model:

```python
# Embedding model
embedding_model_name: str = "intfloat/multilingual-e5-large"

# Language model (có thể thay thành gemma-3-270m nếu có đủ tài nguyên)
language_model_name: str = "google/gemma-3-270m

# Whisper model (tiny, base, small, medium, large)
whisper_model_name: str = "small"
```

### Tùy chỉnh prompts

Prompts cho sinh câu hỏi có thể được tùy chỉnh trong `config.py` tại `QUESTION_PROMPTS`.

### Cấu hình chunk size

```python
chunk_size: int = 1000      # Kích thước mỗi chunk văn bản
chunk_overlap: int = 200    # Độ chồng lấn giữa các chunk
```

## Troubleshooting

### Lỗi thường gặp

1. **Model tải quá chậm**: Giảm kích thước model hoặc sử dụng GPU
2. **Qdrant connection error**: Kiểm tra Qdrant service đã chạy chưa
3. **Out of memory**: Giảm batch size hoặc sử dụng model nhỏ hơn
4. **Whisper model lỗi**: Kiểm tra file audio format có được hỗ trợ không

### Performance tips

- Sử dụng GPU để tăng tốc inference
- Tăng `chunk_size` nếu có đủ RAM
- Cache models để tránh reload
- Sử dụng async endpoints để xử lý concurrent requests

## Contributing

Khi đóng góp vào dự án:

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License

MIT License
