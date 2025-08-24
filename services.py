import os
import tempfile
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
import logging
import numpy as np
import shutil

# Core libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, Docx2txtLoader
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Specific libraries
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, WhisperProcessor, WhisperForConditionalGeneration
from crawl4ai import AsyncWebCrawler
from docling.document_converter import DocumentConverter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerationService:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333):
        """Initialize the Question Generation Service"""
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection_name = "question_generator_docs"
        
        # Initialize components
        self._init_embedding_model()
        self._init_qdrant_client()
        self._init_language_model()
        self._init_whisper_model()
        self._init_docling_converter()
        self._init_text_splitter()
        self._init_web_crawler()
        
        logger.info("QuestionGenerationService initialized successfully")
    
    def _init_embedding_model(self):
        """Initialize multilingual E5 large embedding model"""
        try:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
            self.langchain_embeddings = HuggingFaceEmbeddings(
                model_name='intfloat/multilingual-e5-large',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model initialized")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def _init_qdrant_client(self):
        """Initialize Qdrant vector database client"""
        try:
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            
            # Create collection if it doesn't exist
            try:
                self.qdrant_client.get_collection(self.collection_name)
            except:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1024,  # multilingual-e5-large dimension
                        distance=models.Distance.COSINE
                    )
                )
            
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.langchain_embeddings
            )
            logger.info("Qdrant client initialized")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise
    
    def _init_language_model(self):
        """Initialize Gemma 2-27B model for question generation"""
        try:
            # Note: Using a smaller model first, you can upgrade to gemma-3-270m if you have enough resources
            model_name = "google/gemma-3-270m"  # Using 2B model instead of 270M as it's more capable
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,token ="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")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto",
                token = "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"
            )
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )
            
            self.llm = HuggingFacePipeline(pipeline=self.llm_pipeline)

            logger.info("Language model initialized")
        except Exception as e:
            logger.error(f"Error initializing language model: {e}")
            raise
    
    def _init_whisper_model(self):
        """Initialize Whisper model for audio processing"""
        try:
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            self.whisper_model.config.forced_decoder_ids = None
            logger.info("Whisper model initialized")
        except Exception as e:
            logger.error(f"Error initializing Whisper model: {e}")
            raise
    def _init_docling_converter(self):
        """Initialize Docling converter"""
        try:
            self.docling_converter = DocumentConverter()
            logger.info("Docling converter initialized")
        except Exception as e:
            logger.error(f"Error initializing Docling converter: {e}")
            raise
    def _init_text_splitter(self):
        """Initialize text splitter for document processing"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def _init_web_crawler(self):
        """Initialize web crawler"""
        try:
            self.web_crawler = AsyncWebCrawler()
            logger.info("Web crawler initialized")
        except Exception as e:
            logger.error(f"Error initializing web crawler: {e}")
            raise
    
    
    # INPUT PROCESSING METHODS
    
    def process_text_input(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process text input and store in vector database"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = metadata or {}
                doc_metadata.update({
                    'chunk_id': i,
                    'source': 'text_input',
                    'total_chunks': len(chunks)
                })
                documents.append({
                    'content': chunk,
                    'metadata': doc_metadata
                })
            
            # Store in vector database
            self._store_documents(documents)
            
            logger.info(f"Processed text input into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            raise
    
    async def process_url_input(self, url: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process URL input using web crawler"""
        try:
            # Crawl the URL
            async with AsyncWebCrawler(
                headless=True,
                verbose=False
            ) as crawler:
                result = await crawler.arun(url)
                
                if not result or not result.success:
                    raise Exception(f"Failed to crawl URL: {url}")
                
                # Extract text content
                text_content = result.cleaned_html or result.markdown or ""
                
                if not text_content:
                    raise Exception(f"No content extracted from URL: {url}")
                
                # Add URL to metadata
                url_metadata = metadata or {}
                url_metadata.update({
                    'source': 'url',
                    'url': url,
                    'title': getattr(result, 'title', ''),
                })
                
                # Process the extracted text
                return self.process_text_input(text_content, url_metadata)
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            raise
    
    def process_document_file(self, file_path: str, file_type: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process document files (PDF, DOCX, etc.)"""
        try:
            result = self.docling_converter.convert_single(file_path)
            text_content = result.render_as_markdown()
            
            # Add file metadata
            file_metadata = metadata or {}
            file_metadata.update({
                'source': 'document',
                'file_path': file_path,
                'file_type': file_type
            })
            
            # Process the extracted text
            return self.process_text_input(text_content, file_metadata)
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def process_audio_file(self, audio_content: bytes, file_extension: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process audio file using Whisper"""
        try:
            import librosa
            
            # Save audio content to temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_audio_path = temp_file.name
            
            try:
                # Load audio using librosa
                audio_array, sampling_rate = librosa.load(temp_audio_path, sr=16000)
                
                # Process audio for Whisper
                input_features = self.whisper_processor(
                    audio_array, 
                    sampling_rate=sampling_rate, 
                    return_tensors="pt"
                ).input_features
                
                # Generate token ids
                predicted_ids = self.whisper_model.generate(input_features)
                
                # Decode token ids to text
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )
                
                # Extract transcribed text
                transcribed_text = transcription[0] if transcription else ""
                
                if not transcribed_text:
                    raise Exception("No text could be transcribed from audio")
                
                # Add audio metadata
                audio_metadata = metadata or {}
                audio_metadata.update({
                    'source': 'audio',
                    'file_extension': file_extension,
                    'transcription_method': 'whisper-small',
                    'sampling_rate': sampling_rate,
                    'audio_duration': len(audio_array) / sampling_rate
                })
                
                # Process the transcribed text
                logger.info(f"Successfully transcribed audio file: {len(transcribed_text)} characters")
                return self.process_text_input(transcribed_text, audio_metadata)
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise
    
    def _store_documents(self, documents: List[Dict[str, Any]]):
        """Store documents in vector database"""
        try:
            texts = [doc['content'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # Add to vector store
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            
            logger.info(f"Stored {len(documents)} documents in vector database")
            
        except Exception as e:
            logger.error(f"Error storing documents: {e}")
            raise
    
    # QUESTION GENERATION METHODS
    
    def generate_questions(
        self, 
        query: str = "", 
        num_questions: int = 5,
        question_types: List[str] = None,
        difficulty_level: str = "medium",
        context_limit: int = 3
    ) -> Dict[str, Any]:
        """Generate questions based on stored content"""
        try:
            # Default question types
            if question_types is None:
                question_types = ["multiple_choice", "short_answer", "essay", "true_false", "fill_blank"]
            
            # Retrieve relevant context
            if query:
                relevant_docs = self.vector_store.similarity_search(
                    query, k=context_limit
                )
            else:
                # Get recent documents if no specific query
                relevant_docs = self.vector_store.similarity_search(
                    "information knowledge content", k=context_limit
                )
            
            if not relevant_docs:
                raise Exception("No relevant documents found for question generation")
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate questions for each type
            generated_questions = []
            
            for question_type in question_types[:num_questions]:
                questions = self._generate_questions_by_type(
                    context, question_type, difficulty_level, 1
                )
                generated_questions.extend(questions)
            
            # If we need more questions, generate additional ones
            remaining = num_questions - len(generated_questions)
            if remaining > 0:
                additional_questions = self._generate_questions_by_type(
                    context, "mixed", difficulty_level, remaining
                )
                generated_questions.extend(additional_questions)
            
            result = {
                "questions": generated_questions[:num_questions],
                "context_sources": [{
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                } for doc in relevant_docs],
                "total_generated": len(generated_questions[:num_questions])
            }
            
            logger.info(f"Generated {len(result['questions'])} questions")
            return result
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            raise
    
    def _generate_questions_by_type(
        self, 
        context: str, 
        question_type: str, 
        difficulty: str, 
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate questions of specific type"""
        
        prompts = {
            "multiple_choice": f"""
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
            
            "short_answer": f"""
            Dựa trên nội dung sau, hãy tạo {count} câu hỏi ngắn ({difficulty}) yêu cầu trả lời trong 1-2 câu.
            
            Nội dung:
            {context}
            
            Định dạng trả về:
            Câu hỏi: [câu hỏi]
            Đáp án mẫu: [đáp án mẫu]
            ---
            """,
            
            "essay": f"""
            Dựa trên nội dung sau, hãy tạo {count} câu hỏi tự luận ({difficulty}) yêu cầu trả lời chi tiết.
            
            Nội dung:
            {context}
            
            Định dạng trả về:
            Câu hỏi: [câu hỏi]
            Gợi ý trả lời: [các điểm chính cần đề cập]
            ---
            """,
            
            "true_false": f"""
            Dựa trên nội dung sau, hãy tạo {count} câu hỏi đúng/sai ({difficulty}).
            
            Nội dung:
            {context}
            
            Định dạng trả về:
            Câu hỏi: [câu hỏi]
            Đáp án: [Đúng/Sai]
            Giải thích: [giải thích]
            ---
            """,
            
            "fill_blank": f"""
            Dựa trên nội dung sau, hãy tạo {count} câu hỏi điền vào chỗ trống ({difficulty}).
            
            Nội dung:
            {context}
            
            Định dạng trả về:
            Câu hỏi: [câu có chỗ trống với ___]
            Đáp án: [từ/cụm từ cần điền]
            ---
            """,
            
            "mixed": f"""
            Dựa trên nội dung sau, hãy tạo {count} câu hỏi hỗn hợp ({difficulty}) bao gồm trắc nghiệm, ngắn, tự luận.
            
            Nội dung:
            {context}
            
            Sử dụng định dạng phù hợp cho từng loại câu hỏi.
            ---
            """
        }
        
        try:
            prompt = prompts.get(question_type, prompts["mixed"])
            
            # Generate using LLM
            response = self.llm.invoke(prompt)
            
            # Parse response into structured format
            questions = self._parse_generated_questions(response, question_type)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating {question_type} questions: {e}")
            return []
    
    def _parse_generated_questions(self, response: str, question_type: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured questions"""
        questions = []
        
        try:
            # Split by separator
            parts = response.split("---")
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                    
                question_data = {
                    "type": question_type,
                    "raw_content": part
                }
                
                # Extract question
                if "Câu hỏi:" in part:
                    lines = part.split("\n")
                    for line in lines:
                        if line.strip().startswith("Câu hỏi:"):
                            question_data["question"] = line.replace("Câu hỏi:", "").strip()
                            break
                
                # Type-specific parsing
                if question_type == "multiple_choice":
                    options = []
                    correct_answer = ""
                    explanation = ""
                    
                    lines = part.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith(("A.", "B.", "C.", "D.")):
                            options.append(line)
                        elif line.startswith("Đáp án:"):
                            correct_answer = line.replace("Đáp án:", "").strip()
                        elif line.startswith("Giải thích:"):
                            explanation = line.replace("Giải thích:", "").strip()
                    
                    question_data.update({
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": explanation
                    })
                
                elif question_type == "short_answer":
                    lines = part.split("\n")
                    for line in lines:
                        if line.strip().startswith("Đáp án mẫu:"):
                            question_data["sample_answer"] = line.replace("Đáp án mẫu:", "").strip()
                            break
                
                elif question_type == "essay":
                    lines = part.split("\n")
                    for line in lines:
                        if line.strip().startswith("Gợi ý trả lời:"):
                            question_data["answer_hints"] = line.replace("Gợi ý trả lời:", "").strip()
                            break
                
                elif question_type == "true_false":
                    correct_answer = ""
                    explanation = ""
                    
                    lines = part.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Đáp án:"):
                            correct_answer = line.replace("Đáp án:", "").strip()
                        elif line.startswith("Giải thích:"):
                            explanation = line.replace("Giải thích:", "").strip()
                    
                    question_data.update({
                        "correct_answer": correct_answer,
                        "explanation": explanation
                    })
                
                elif question_type == "fill_blank":
                    lines = part.split("\n")
                    for line in lines:
                        if line.strip().startswith("Đáp án:"):
                            question_data["answer"] = line.replace("Đáp án:", "").strip()
                            break
                
                if "question" in question_data:
                    questions.append(question_data)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
            return []
    
    def clear_vector_store(self):
        """Clear all documents from vector store"""
        try:
            # Delete and recreate collection
            self.qdrant_client.delete_collection(self.collection_name)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE
                )
            )
            
            # Reinitialize vector store
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embeddings=self.langchain_embeddings
            )
            
            logger.info("Vector store cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            raise
