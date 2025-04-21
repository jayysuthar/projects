import torch
import logging
from typing import List, Dict, Any, Union, Optional
from transformers import T5ForConditionalGeneration, AutoTokenizer

class FlanT5Model:
    """
    Generate responses using Google's Flan T5 model.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-xxl",
                 device: Union[str, torch.device] = None,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 precision: str = "float16",
                 cache_dir: Optional[str] = None):
        """
        Initialize the language model.
        
        Args:
            model_name: Name of the model to use
            device: Device to use (cpu or cuda)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling
            top_k: Top-k sampling
            precision: Precision to use (float32, float16, int8)
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading model {model_name} on {self.device}")
        
        # Set precision
        self.precision = precision
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        self.dtype = dtype_map.get(precision, torch.float32)
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        if self.precision == "int8":
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, 
                device_map="auto", 
                load_in_8bit=True,
                cache_dir=cache_dir
            )
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, 
                torch_dtype=self.dtype,
                cache_dir=cache_dir
            )
            self.model.to(self.device)
        
        self.logger.info(f"Model loaded successfully")
    
    def _create_prompt(self, question: str, context: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Create a prompt for the model.
        
        Args:
            question: Question to answer
            context: Context to use for answering
            chat_history: Chat history
        
        Returns:
            Prompt for the model
        """
        # Format chat history
        history_text = ""
        if chat_history:
            for exchange in chat_history[-3:]:  # Only use the last 3 exchanges to avoid context length issues
                history_text += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n"
        
        # Create the prompt
        prompt = f"Answer the question based on the following context:\n\nContext: {context}\n\n"
        if history_text:
            prompt += f"Chat History:\n{history_text}\n"
        prompt += f"Question: {question}\n\nAnswer:"
        
        return prompt
    
    def generate_response(self, question: str, context: str, 
                          chat_history: List[Dict[str, str]] = None) -> str:
        """
        Generate a response to a question.
        
        Args:
            question: Question to answer
            context: Context to use for answering
            chat_history: Chat history
        
        Returns:
            Generated response
        """
        if chat_history is None:
            chat_history = []
        
        # Create prompt
        prompt = self._create_prompt(question, context, chat_history)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.temperature > 0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response
        response = response.replace(prompt, "").strip()
        
        return response