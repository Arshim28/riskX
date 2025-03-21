import os
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import tiktoken
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from mistralai import Mistral
from base.base_tools import BaseTool, ToolResult
from utils.logging import get_logger


class ImageData(BaseModel):
    id: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    image_base64: str


class Dimensions(BaseModel):
    dpi: int
    height: int
    width: int


class Page(BaseModel):
    index: int
    markdown: str
    images: List[ImageData]
    dimensions: Dimensions


class UsageInfo(BaseModel):
    pages_processed: int
    doc_size_bytes: int


class OcrResponse(BaseModel):
    pages: List[Page]
    model: str
    usage_info: UsageInfo


class TokenInfo(BaseModel):
    total_tokens: int
    tokens_per_page: Dict[int, int]
    encoding_name: str


class OcrTool(BaseTool):
    name = "ocr_tool"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(self.name)
        self.api_key = config.get("api_key", os.environ.get("MISTRAL_API_KEY"))
        if not self.api_key:
            raise ValueError("Mistral API key not provided")
        self.client = Mistral(api_key=self.api_key)
        self.pdf_id: Optional[str] = None
        self.pdf_url: Optional[str] = None
        self.pages: Optional[List[Page]] = None
        self.usage_info: Optional[UsageInfo] = None
        self.model: Optional[str] = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _upload_pdf(self, pdf_path: str) -> None:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        try:
            loop = asyncio.get_event_loop()
            
            with open(pdf_path, "rb") as f:
                file_content = f.read()
                
            uploaded_pdf = await loop.run_in_executor(
                None,
                lambda: self.client.files.upload(
                    file={
                        "file_name": os.path.basename(pdf_path),
                        "content": file_content,
                    },
                    purpose="ocr"
                )
            )
            
            self.logger.info(f"Successfully uploaded PDF: {os.path.basename(pdf_path)}")
            
            self.pdf_id = uploaded_pdf.id
            signed_url_response = await loop.run_in_executor(
                None,
                lambda: self.client.files.get_signed_url(file_id=self.pdf_id)
            )
            self.pdf_url = signed_url_response.url  
        except Exception as e:
            self.logger.error(f"Error uploading PDF: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _get_ocr_response(self, pages=None, image_limit=None, image_min_size=None) -> OcrResponse:
        if not self.pdf_url:
            raise ValueError("No PDF URL found. Please upload a PDF first.")
            
        try:
            loop = asyncio.get_event_loop()
            
            payload = {
                "model": "mistral-ocr-latest",
                "document": {
                    "type": "document_url",
                    "document_url": self.pdf_url,
                },
                "include_image_base64": True
            }
            
            if pages is not None:
                payload["pages"] = pages
            
            if image_limit is not None:
                payload["image_limit"] = image_limit
                
            if image_min_size is not None:
                payload["image_min_size"] = image_min_size
            
            response = await loop.run_in_executor(
                None,
                lambda: self.client.ocr.process(**payload)
            )
            
            self.logger.info("Successfully processed document with OCR")
            
            self.pages = response.pages
            self.usage_info = response.usage_info
            self.model = response.model
            
            return OcrResponse(
                pages=self.pages,
                model=self.model,
                usage_info=self.usage_info
            )
        except Exception as e:
            self.logger.error(f"Error processing OCR: {e}")
            raise

    async def execute(self, pdf_path: str, pages=None, image_limit=None, image_min_size=None) -> OcrResponse:
        await self._upload_pdf(pdf_path)
        return await self._get_ocr_response(pages=pages, image_limit=image_limit, image_min_size=image_min_size)

    async def get_text(self) -> str:
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
            
        return "".join(page.markdown for page in self.pages)
    
    async def get_text_by_page(self) -> Dict[int, str]:
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
        
        return {page.index: page.markdown for page in self.pages}

    async def get_images(self) -> List[ImageData]:
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
            
        return [image for page in self.pages for image in page.images]

    async def get_usage_info(self) -> UsageInfo:
        if not self.usage_info:
            raise ValueError("No OCR response found. Please run execute() method first.")
            
        return self.usage_info
    
    async def count_tokens(self, encoding_name="cl100k_base") -> TokenInfo:
        if not self.pages:
            raise ValueError("No OCR response found. Please run execute() method first.")
        
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            
            tokens_per_page = {}
            total_tokens = 0
            
            for page in self.pages:
                page_tokens = len(encoding.encode(page.markdown))
                tokens_per_page[page.index] = page_tokens
                total_tokens += page_tokens
            
            return TokenInfo(
                total_tokens=total_tokens,
                tokens_per_page=tokens_per_page,
                encoding_name=encoding_name
            )
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            raise
            
    async def analyze_for_embedding(self, text, encoding_name="cl100k_base") -> Tuple[int, Dict[str, str]]:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = len(encoding.encode(text))
        
        embedding_services = {
            "OpenAI Ada 002": 8191,
            "OpenAI text-embedding-3-small": 8191,
            "OpenAI text-embedding-3-large": 8191,
            "Cohere embed-english-v3.0": 512,
            "Cohere embed-multilingual-v3.0": 512,
            "Azure OpenAI Embeddings": 8191,
            "Mistral embed": 8192,
            "Vertex AI Embeddings": 3072,
            "Anthropic Embed": 9000,
            "Gemini embedding": 8000,
        }
        
        compatibility = {}
        for service, limit in embedding_services.items():
            if tokens <= limit:
                compatibility[service] = "Compatible"
            else:
                compatibility[service] = f"Exceeds limit by {tokens - limit} tokens"
        
        return tokens, compatibility

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def run(self, pdf_path: str, pages=None, image_limit=None, image_min_size=None, **kwargs) -> ToolResult[Dict[str, Any]]:
        try:
            response = await self.execute(pdf_path, pages, image_limit, image_min_size)
            
            text = await self.get_text()
            text_by_page = await self.get_text_by_page()
            images = await self.get_images()
            usage_info = await self.get_usage_info()
            token_info = await self.count_tokens()
            token_count, compatibility = await self.analyze_for_embedding(text)
            
            result = {
                "text": text,
                "text_by_page": text_by_page,
                "images": [img.model_dump() for img in images],
                "usage_info": usage_info.model_dump(),
                "token_info": token_info.model_dump(),
                "embedding_compatibility": compatibility,
                "model": self.model,
                "token_count": token_count
            }
            
            self.logger.info(f"Successfully processed PDF with {len(self.pages)} pages and {token_count} tokens")
            
            return ToolResult(success=True, data=result)
        except Exception as e:
            self.logger.error(f"OCR processing error: {str(e)}")
            return await self._handle_error(e)