import re
from typing import List, Dict, Any

class TextProcessor:
    """
    Process and chunk text data for embedding generation.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 min_chunk_length: int = 100, remove_html_tags: bool = True,
                 remove_urls: bool = True, remove_emails: bool = True):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            min_chunk_length: Minimum length of a chunk
            remove_html_tags: Whether to remove HTML tags
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.remove_html_tags = remove_html_tags
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
    
    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted elements."""
        # Remove HTML tags
        if self.remove_html_tags:
            text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\S+@\S+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        # Clean the text first
        text = self.clean_text(text)
        
        # If text is shorter than chunk size, return it as is
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_length else []
        
        # Split text into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence exceeds chunk size, save current chunk and start a new one
            if len(current_chunk) + len(sentence) > self.chunk_size and len(current_chunk) >= self.min_chunk_length:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                words = current_chunk.split()
                if len(words) > self.chunk_overlap // 10:  # Approximate number of words for overlap
                    overlap_text = " ".join(words[-self.chunk_overlap // 10:])
                    current_chunk = overlap_text + " "
                else:
                    current_chunk = ""
            
            current_chunk += sentence + " "
        
        # Add the last chunk if it's long enough
        if len(current_chunk) >= self.min_chunk_length:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_and_chunk(self, text: str) -> List[str]:
        """Process and chunk text."""
        return self.split_into_chunks(text)