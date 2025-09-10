"""
ConvAI Innovations - Interactive LLM Training Academy

A comprehensive educational platform for learning to build Large Language Models
from scratch through hands-on coding sessions with AI mentor Sandra!

Enhanced Features:
- 🌍 Multi-language support (English, Spanish, French, Hindi, Italian, Portuguese)
- 📊 Interactive visualizations for ML concepts
- 🤖 AI-powered code generation
- 🔊 Text-to-speech guidance in multiple languages
- 🎯 Progressive learning sessions from Python basics to LLM deployment
- 🌐 Offline translation support for better accessibility

Author: ConvAI Innovations
License: GPL-3.0
"""

__version__ = "1.1.1"
__author__ = "ConvAI Innovations"
__email__ = "support@convai-innovations.com"
__license__ = "GPL-3.0"

# Import core components
from .convai import (
    SessionBasedLLMLearningDashboard,
    main
)

from .session_manager import SessionManager
from .ai_systems import LLMAIFeedbackSystem, EnhancedKokoroTTSSystem
from .visualizations import VisualizationManager
from .ui_components import (
    ModernCodeEditor,
    CodeGenerationPanel,
    LanguageSelector,
    ModelDownloader
)
from .models import (
    Language,
    Session,
    LearningProgress,
    VisualizationConfig,
    CodeGenRequest,
    CodeGenResponse
)
from .utils import (
    check_dependencies,
    get_system_info,
    setup_data_directory,
    validate_model_file,
    format_file_size
)

# Public API
__all__ = [
    # Main application
    "SessionBasedLLMLearningDashboard",
    "main",
    "run_convai",
    
    # Core managers
    "SessionManager",
    "LLMAIFeedbackSystem", 
    "EnhancedKokoroTTSSystem",
    "VisualizationManager",
    
    # UI Components
    "ModernCodeEditor",
    "CodeGenerationPanel",
    "LanguageSelector",
    "ModelDownloader",
    
    # Data models
    "Language",
    "Session",
    "LearningProgress",
    "VisualizationConfig",
    "CodeGenRequest",
    "CodeGenResponse",
    
    # Utilities
    "check_dependencies",
    "get_system_info",
    "setup_data_directory",
    "validate_model_file",
    "format_file_size"
]


def run_convai():
    """Alternative entry point for programmatic use."""
    main()


def get_version():
    """Get the current version."""
    return __version__


def get_supported_languages():
    """Get list of supported languages."""
    return [lang.display_name for lang in Language]


def check_system_requirements():
    """
    Check system requirements and return status report.
    
    Returns:
        Dict with system status information
    """
    from .utils import check_dependencies, get_system_info, get_gpu_info
    
    all_deps_ok, missing_req, missing_opt = check_dependencies()
    system_info = get_system_info()
    gpu_info = get_gpu_info()
    
    return {
        "dependencies": {
            "all_required_available": all_deps_ok,
            "missing_required": missing_req,
            "missing_optional": missing_opt
        },
        "system": system_info,
        "gpu": gpu_info,
        "version": __version__
    }


def create_learning_session(title, description, code, objectives, hints, 
                          visualization_type=None):
    """
    Create a custom learning session.
    
    Args:
        title: Session title
        description: Session description
        code: Reference code
        objectives: List of learning objectives
        hints: List of hints
        visualization_type: Optional visualization type
        
    Returns:
        Session object
    """
    import uuid
    session_id = str(uuid.uuid4())
    
    return Session(
        id=session_id,
        title=title,
        description=description,
        reference_code=code,
        learning_objectives=objectives,
        hints=hints,
        visualization_type=visualization_type
    )


# Package metadata for introspection
__package_info__ = {
    "name": "convai-innovations",
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "description": "Interactive LLM Training Academy - Learn to build language models from scratch",
    "url": "https://github.com/ConvAI-Innovations/ailearning",
    "license": __license__,
    "features": [
        "Multi-language TTS support",
        "Interactive ML visualizations", 
        "AI-powered code generation",
        "Progressive learning curriculum",
        "Real-time feedback system"
    ],
    "supported_languages": get_supported_languages()
}


# Welcome message
def print_welcome():
    """Print welcome message with system info."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    🧠 ConvAI Innovations v{__version__}                    ║
║              Interactive LLM Training Academy                ║
║                                                              ║
║  🌍 Multi-language support  📊 Interactive visualizations   ║
║  🤖 AI code generation      🔊 Text-to-speech guidance      ║
║  🎯 Progressive curriculum  ⚡ Real-time feedback           ║
║                                                              ║
║        Learn to build Language Models from scratch!         ║
║              With AI Mentor Sandra by your side             ║
╚══════════════════════════════════════════════════════════════╝

Supported Languages: {', '.join(get_supported_languages())}
""")


# Auto-check dependencies on import
if __name__ != "__main__":
    try:
        status = check_system_requirements()
        if not status["dependencies"]["all_required_available"]:
            import warnings
            missing = ", ".join(status["dependencies"]["missing_required"])
            warnings.warn(
                f"Missing required dependencies: {missing}. "
                "Some features may not work properly.",
                ImportWarning
            )
    except Exception:
        pass  # Don't fail on import