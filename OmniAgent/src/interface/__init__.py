"""
Interface module for OmniAgent.
Contains Gradio app implementation.
"""

from .gradio_app import OmniAgentChatbot, create_gradio_interface

__all__ = ["OmniAgentChatbot", "create_gradio_interface"]