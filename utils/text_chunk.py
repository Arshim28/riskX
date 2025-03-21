from typing import Dict, Any, Optional, List
from utils.logging import get_logger


class TextChunk:
    def __init__(self, text: str, metadata: Dict[str, Any]):
        self.text = text
        self.metadata = metadata
        self.embedding: Optional[List[float]] = None
        self._log_creation()
    
    def _log_creation(self):
        try:
            logger = get_logger("text_chunk")
            logger.debug(f"Created TextChunk with length {len(self.text)} bytes")
        except RuntimeError:
            pass

    def __repr__(self):
        return f"TextChunk(len={len(self.text)}, metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
        chunk = cls(text=data["text"], metadata=data["metadata"])
        if "embedding" in data and data["embedding"] is not None:
            chunk.embedding = data["embedding"]
        return chunk

    def __del__(self):
        try:
            self.text = None
            self.metadata = None
            self.embedding = None
        except:
            pass