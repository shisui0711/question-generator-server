import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import shutil

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension from filename"""
        return Path(filename).suffix.lower().lstrip('.')
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        """Get MIME type from filename"""
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"
    
    @staticmethod
    def is_supported_document(filename: str, supported_types: List[str]) -> bool:
        """Check if document type is supported"""
        extension = FileUtils.get_file_extension(filename)
        return extension in supported_types
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    @staticmethod
    def create_safe_filename(filename: str) -> str:
        """Create a safe filename by removing/replacing dangerous characters"""
        # Remove dangerous characters
        safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        safe_filename = ''.join(c for c in filename if c in safe_chars)
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(safe_filename)
        return f"{name}_{timestamp}{ext}"
    
    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    @staticmethod
    def ensure_directory_exists(directory: str) -> None:
        """Ensure directory exists, create if it doesn't"""
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def clean_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
        """Clean temporary files older than max_age_hours"""
        cleaned_count = 0
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        try:
            for file_path in Path(temp_dir).glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned temp file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
        
        return cleaned_count

class TextUtils:
    """Utility functions for text processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove control characters but keep newlines and tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
        """Truncate text to specified length"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text (simple implementation)"""
        # Remove common stop words (Vietnamese and English)
        stop_words = {
            'và', 'hoặc', 'nhưng', 'vì', 'nên', 'để', 'của', 'trong', 'trên', 'dưới',
            'với', 'từ', 'đến', 'về', 'cho', 'bằng', 'theo', 'như', 'khi', 'nếu',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        # Simple word extraction
        words = text.lower().split()
        keywords = [word.strip('.,!?;:"()[]{}') for word in words 
                   if len(word) > 3 and word.lower() not in stop_words]
        
        # Count frequency and return top keywords
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:max_keywords]]
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences (simple implementation)"""
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

class ValidationUtils:
    """Utility functions for validation"""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid"""
        import re
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+'  # domain...
            r'(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # host...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(url))
    
    @staticmethod
    def validate_question_types(question_types: List[str], supported_types: List[str]) -> bool:
        """Validate question types against supported types"""
        return all(qtype in supported_types for qtype in question_types)
    
    @staticmethod
    def validate_difficulty_level(difficulty: str) -> bool:
        """Validate difficulty level"""
        valid_levels = ['easy', 'medium', 'hard', 'expert']
        return difficulty.lower() in valid_levels
    
    @staticmethod
    def validate_file_size(file_size: int, max_size: int) -> bool:
        """Validate file size"""
        return file_size <= max_size

class QuestionUtils:
    """Utility functions for question processing"""
    
    @staticmethod
    def format_question_response(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format question response for API"""
        formatted_questions = []
        
        for i, question in enumerate(questions):
            formatted_q = {
                "id": i + 1,
                "question": question.get("question", ""),
                "type": question.get("type", "unknown"),
                "created_at": datetime.now().isoformat()
            }
            
            # Add type-specific fields
            if question.get("type") == "multiple_choice":
                formatted_q.update({
                    "options": question.get("options", []),
                    "correct_answer": question.get("correct_answer", ""),
                    "explanation": question.get("explanation", "")
                })
            elif question.get("type") == "short_answer":
                formatted_q["sample_answer"] = question.get("sample_answer", "")
            elif question.get("type") == "essay":
                formatted_q["answer_hints"] = question.get("answer_hints", "")
            elif question.get("type") == "true_false":
                formatted_q.update({
                    "correct_answer": question.get("correct_answer", ""),
                    "explanation": question.get("explanation", "")
                })
            elif question.get("type") == "fill_blank":
                formatted_q["answer"] = question.get("answer", "")
            
            formatted_questions.append(formatted_q)
        
        return formatted_questions
    
    @staticmethod
    def calculate_question_difficulty_score(question: Dict[str, Any]) -> float:
        """Calculate difficulty score for a question (0-1 scale)"""
        # Simple heuristic based on question length and complexity
        question_text = question.get("question", "")
        
        # Base score
        score = 0.5
        
        # Adjust based on question length
        if len(question_text) > 100:
            score += 0.2
        elif len(question_text) < 50:
            score -= 0.1
        
        # Adjust based on question type
        qtype = question.get("type", "")
        if qtype == "essay":
            score += 0.3
        elif qtype == "multiple_choice":
            score += 0.1
        elif qtype == "true_false":
            score -= 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

class ResponseUtils:
    """Utility functions for API responses"""
    
    @staticmethod
    def create_success_response(
        message: str, 
        data: Any = None, 
        status_code: int = 200
    ) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code
        }
        
        if data is not None:
            response["data"] = data
        
        return response
    
    @staticmethod
    def create_error_response(
        message: str, 
        error_code: str = None,
        details: Any = None,
        status_code: int = 400
    ) -> Dict[str, Any]:
        """Create standardized error response"""
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code
        }
        
        if error_code:
            response["error_code"] = error_code
        
        if details:
            response["details"] = details
        
        return response

# Export utility classes
__all__ = [
    "FileUtils",
    "TextUtils", 
    "ValidationUtils",
    "QuestionUtils",
    "ResponseUtils"
]
