import torch
import logging
from typing import List, Dict, Any, Union
import numpy as np

from transformers import AutoTokenizer, AutoModel

class InstructorEmbeddings:
    """
    Generate embeddings using the Instructor model.
    """
    
    def __init__(self, model_name: str = "hkunlp/instructor-large",
                 instruction: str = "Represent the University of Texas at Dallas content for retrieval:",
                 device: Union[str, torch.device] = None,
                 max_length: int = 512,
                 batch_size: int = 32,
                 cache_dir: str = None):
        """
        Initialize the embeddings model.
        
        Args:
            model_name: Name of the model to use
            instruction: Instruction for the model
            device: Device to use (cpu or cuda)
            max_length: Maximum length of input
            batch_size: Batch size for processing
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading model {model_name} on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        
        self.logger.info(f"Model loaded successfully")
    
    def _prepare_inputs(self, texts: List[str]) -> List[List[str]]:
        """Prepare inputs for the model with instruction."""
        inputs = []
        for text in texts:
            inputs.append([self.instruction, text])
        return inputs
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embeddings
        """
        self.model.eval()
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_inputs = self._prepare_inputs(batch_texts)
            
            # Tokenize
            encoded_inputs = self.tokenizer(
                batch_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded_inputs)
                embeddings = self._mean_pooling(outputs, encoded_inputs['attention_mask'])
                
                # Convert to list of lists
                embeddings_np = embeddings.cpu().numpy()
                all_embeddings.extend(embeddings_np.tolist())
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
        
        Returns:
            Query embedding
        """
        return self.embed_documents([query])[0]
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling to get sentence embeddings.
        
        Args:
            model_output: Output from the model
            attention_mask: Attention mask
        
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)