"""
ConvAI Innovations - Interactive LLM Training Academy

A comprehensive educational platform for learning to build Large Language Models
from scratch through hands-on coding sessions.

Author: ConvAI Innovations
License: MIT
"""

__version__ = "1.0.0"
__author__ = "ConvAI Innovations"
__email__ = "support@convai-innovations.com"
__license__ = "GPL-3.0"

# Import main components
from .convai import (
    SessionBasedLLMLearningDashboard,
    SessionManager,
    LLMAIFeedbackSystem,
    EnhancedKokoroTTSSystem,
    ModernCodeEditor,
    ModelDownloader,
)

# Public API
__all__ = [
    "SessionBasedLLMLearningDashboard",
    "SessionManager", 
    "LLMAIFeedbackSystem",
    "EnhancedKokoroTTSSystem",
    "ModernCodeEditor",
    "ModelDownloader",
    "main",
    "run_convai",
]

def main():
    """Main entry point for the ConvAI Innovations application."""
    from .cli import main as cli_main
    cli_main()

def run_convai():
    """Alternative entry point for programmatic use."""
    main()

# Package metadata
__package_info__ = {
    "name": "convai-innovations",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": "Interactive LLM Training Academy - Learn to build language models from scratch",
    "url": "https://github.com/convai-innovations/convai-innovations",
    "license": __license__,
}