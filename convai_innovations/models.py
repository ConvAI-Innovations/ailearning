"""
Data classes and models for ConvAI Innovations platform.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Language(Enum):
    """Supported languages - FIXED with both ISO codes (for translation) and TTS codes (for Kokoro)"""
    ENGLISH = ('en', 'a', 'af_bella', 'English')         # ISO: en, TTS: a
    SPANISH = ('es', 'e', 'ef_dora', 'Español')          # ISO: es, TTS: e
    FRENCH = ('fr', 'f', 'ff_siwis', 'Français')         # ISO: fr, TTS: f
    HINDI = ('hi', 'h', 'hf_alpha', 'हिन्दी')             # ISO: hi, TTS: h
    ITALIAN = ('it', 'i', 'if_sara', 'Italiano')         # ISO: it, TTS: i
    PORTUGUESE = ('pt', 'p', 'pf_dora', 'Português')     # ISO: pt, TTS: p
    
    def __init__(self, iso_code, tts_code, voice, display_name):
        self.code = iso_code      # For translation (ISO 639-1)
        self.tts_code = tts_code  # For Kokoro TTS
        self.voice = voice
        self.display_name = display_name


@dataclass
class Session:
    """Represents a learning session"""
    id: str
    title: str
    description: str
    reference_code: str
    learning_objectives: List[str]
    hints: List[str]
    visualization_type: Optional[str] = None
    completed: bool = False
    

@dataclass
class LearningProgress:
    """Tracks overall learning progress"""
    current_session_id: str = "python_fundamentals"
    completed_sessions: List[str] = field(default_factory=list)
    session_scores: Dict[str, float] = field(default_factory=dict)
    total_sessions: int = 0
    preferred_language: Language = Language.ENGLISH
    
    def get_completion_percentage(self) -> float:
        if self.total_sessions == 0:
            return 0.0
        return (len(self.completed_sessions) / self.total_sessions) * 100


@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    width: int = 800
    height: int = 600
    background_color: str = "#1e1e1e"
    primary_color: str = "#00aaff"
    secondary_color: str = "#ffc107"
    success_color: str = "#28a745"
    error_color: str = "#dc3545"
    animate: bool = True
    animation_speed: int = 1000


@dataclass
class CodeGenRequest:
    """Request for code generation"""
    session_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    language: str = "python"


@dataclass
class CodeGenResponse:
    """Response from code generation"""
    generated_code: str
    explanation: str
    success: bool
    error_message: Optional[str] = None