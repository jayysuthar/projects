import os
import gradio as gr
import logging
import yaml
import torch
import json
from typing import List, Dict, Any, Tuple
import numpy as np

from langchain_community.vectorstores import FAISS

# Import OmniAgent modules
from src.data_collection.web_scraper import WebScraper
from src.embeddings.instructor_embeddings import InstructorEmbeddings
from src.language_model.flan_t5_model import FlanT5Model

class OmniAgentChatbot:
    """
    Chatbot for the University of Texas at Dallas.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the chatbot.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize embeddings model
        self.embeddings_model = InstructorEmbeddings(
            model_name=self.config["embeddings"]["model_name"],
            instruction=self.config["embeddings"]["instruction"],
            device=self.device,
            max_length=self.config["embeddings"]["max_length"],
            batch_size=self.config["embeddings"]["batch_size"],
            cache_dir=self.config["embeddings"]["cache_dir"]
        )
        
        # Initialize language model
        self.language_model = FlanT5Model(
            model_name=self.config["language_model"]["model_name"],
            device=self.device,
            max_new_tokens=self.config["language_model"]["max_new_tokens"],
            temperature=self.config["language_model"]["temperature"],
            top_p=self.config["language_model"]["top_p"],
            top_k=self.config["language_model"]["top_k"],
            precision=self.config["language_model"]["precision"],
            cache_dir=self.config["language_model"]["cache_dir"]
        )
        
        # Load vector store
        self.vector_store = self._load_vector_store()
        
        # Initialize chat history
        self.chat_history = []
    
    def _load_vector_store(self):
        """Load the vector store from disk."""
        vector_store_path = self.config["faiss"]["storage_path"]
        
        if os.path.exists(os.path.dirname(vector_store_path)):
            try:
                self.logger.info(f"Loading vector store from {vector_store_path}")
                
                # Create a wrapper around embeddings model for compatibility with FAISS
                class EmbeddingsWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_documents(self, texts):
                        return self.model.embed_documents(texts)
                    
                    def embed_query(self, text):
                        return self.model.embed_query(text)
                
                embeddings_wrapper = EmbeddingsWrapper(self.embeddings_model)
                
                # Load FAISS index
                vector_store = FAISS.load_local(
                    os.path.dirname(vector_store_path),
                    embeddings_wrapper,
                    allow_dangerous_deserialization=True
                )
                
                self.logger.info(f"Vector store loaded successfully")
                return vector_store
                
            except Exception as e:
                self.logger.error(f"Error loading vector store: {str(e)}")
                self.logger.info("Creating a new vector store")
                
                # Create a simple dummy vector store
                dummy_texts = ["University of Texas at Dallas information"]
                dummy_embeddings = [self.embeddings_model.embed_documents(dummy_texts)[0]]
                
                vector_store = FAISS.from_embeddings(
                    text_embeddings=list(zip(dummy_texts, dummy_embeddings)),
                    embedding=None
                )
                
                return vector_store
        else:
            self.logger.warning(f"Vector store path {vector_store_path} does not exist")
            
            # Create a simple dummy vector store
            dummy_texts = ["University of Texas at Dallas information"]
            dummy_embeddings = [self.embeddings_model.embed_documents(dummy_texts)[0]]
            
            vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(dummy_texts, dummy_embeddings)),
                embedding=None
            )
            
            return vector_store
    
    def retrieve_context(self, query: str) -> str:
        """
        Retrieve context for a query.
        
        Args:
            query: Query to retrieve context for
        
        Returns:
            Context
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings_model.embed_query(query)
            
            # Search vector store
            docs_and_scores = self.vector_store.similarity_search_by_vector_with_score(
                query_embedding,
                k=self.config["retrieval"]["top_k"]
            )
            
            # Extract contexts
            contexts = []
            for doc, score in docs_and_scores:
                if score < self.config["retrieval"]["similarity_threshold"]:
                    contexts.append(doc.page_content)
            
            # Join contexts
            context = "\n\n".join(contexts)
            
            return context
        
        except Exception as e:
            self.logger.error(f"Error retrieving context: {str(e)}")
            return "Information about the University of Texas at Dallas."
    
    def chat(self, message: str, history: List[List[str]]) -> str:
        """
        Chat with the user.
        
        Args:
            message: User message
            history: Chat history
        
        Returns:
            Response
        """
        try:
            # Convert Gradio history to our format
            chat_history = []
            for user_msg, assistant_msg in history:
                chat_history.append({"user": user_msg, "assistant": assistant_msg})
            
            # Retrieve context
            context = self.retrieve_context(message)
            
            # Generate response
            response = self.language_model.generate_response(message, context, chat_history)
            
            # Update chat history
            chat_history.append({"user": message, "assistant": response})
            self.chat_history = chat_history
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, I encountered an error. Please try again."

def create_gradio_interface(config_path: str = "config/config.yaml"):
    """
    Create a Gradio interface for the chatbot.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Gradio interface
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    ui_config = config["ui"]
    
    # Initialize chatbot
    chatbot = OmniAgentChatbot(config_path)
    
    # Set up the theme
    theme = gr.themes.Default()
    
    # Create the chat interface
    chat_interface = gr.ChatInterface(
        fn=chatbot.chat,
        title=ui_config["title"],
        description=ui_config["description"],
        theme=theme,
        examples=[
            ["What is UT Dallas?"],
            ["How many schools does UTD have?"],
            ["Tell me about the Computer Science program at UTD."],
            ["What admission requirements are there for international students?"],
            ["Where is the university located?"]
        ],
        cache_examples=False
    )
    
    return chat_interface

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )