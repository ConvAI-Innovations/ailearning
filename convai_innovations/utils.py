"""
Utility functions for ConvAI Innovations platform.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib.util


def check_dependencies() -> Tuple[bool, List[str], List[str]]:
    """
    Check if required and optional dependencies are available.
    
    Returns:
        Tuple of (all_required_available, missing_required, missing_optional)
    """
    required_deps = [
        ("tkinter", "tkinter (usually comes with Python)"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("numpy", "numpy")
    ]
    
    optional_deps = [
        ("kokoro", "kokoro-tts"),
        ("sounddevice", "sounddevice")
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required dependencies
    for module_name, display_name in required_deps:
        if not _is_module_available(module_name):
            missing_required.append(display_name)
    
    # Check optional dependencies
    for module_name, display_name in optional_deps:
        if not _is_module_available(module_name):
            missing_optional.append(display_name)
    
    all_required_available = len(missing_required) == 0
    
    return all_required_available, missing_required, missing_optional


def _is_module_available(module_name: str) -> bool:
    """Check if a module is available for import."""
    try:
        if module_name == "tkinter":
            import tkinter
            return True
        else:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
    except ImportError:
        return False


def get_system_info() -> Dict[str, str]:
    """Get system information for debugging."""
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "working_directory": os.getcwd(),
        "home_directory": str(Path.home()),
        "architecture": platform.architecture()[0]
    }


def setup_data_directory(custom_path: Optional[str] = None) -> Path:
    """
    Setup data directory for models and user data.
    
    Args:
        custom_path: Custom path for data directory
        
    Returns:
        Path to data directory
    """
    if custom_path:
        data_dir = Path(custom_path)
    else:
        # Use environment variable or default
        env_path = os.environ.get("CONVAI_DATA_DIR")
        if env_path:
            data_dir = Path(env_path)
        else:
            # Default to user's home directory
            data_dir = Path.home() / ".convai_innovations"
    
    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "models").mkdir(exist_ok=True)
    (data_dir / "sessions").mkdir(exist_ok=True)
    (data_dir / "user_code").mkdir(exist_ok=True)
    
    return data_dir


def validate_model_file(model_path: str) -> Tuple[bool, str]:
    """
    Validate that a model file exists and appears to be valid.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(model_path)
    
    if not path.exists():
        return False, f"Model file does not exist: {model_path}"
    
    if not path.is_file():
        return False, f"Path is not a file: {model_path}"
    
    # Check file size (should be > 100MB for a valid model)
    file_size = path.stat().st_size
    if file_size < 100_000_000:  # 100MB
        return False, f"Model file appears too small: {file_size / 1_000_000:.1f}MB"
    
    # Check file extension (transformers supports various formats)
    if not model_path.lower().endswith(('.bin', '.safetensors', '.pt', '.pth')):
        return False, f"Unsupported model file format: {path.suffix}"
    
    return True, "Model file appears valid"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def safe_import(module_name: str, package=None):
    """
    Safely import a module without raising ImportError.
    
    Args:
        module_name: Name of module to import
        package: Package name for relative imports
        
    Returns:
        Module object if successful, None otherwise
    """
    try:
        return importlib.import_module(module_name, package)
    except ImportError:
        return None


def get_available_languages() -> List[str]:
    """Get list of available language codes for TTS."""
    return ['en', 'es', 'fr', 'hi', 'it', 'pt']


def create_desktop_shortcut(install_path: str) -> bool:
    """
    Create desktop shortcut for the application.
    
    Args:
        install_path: Path to the installation directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        system = platform.system()
        
        if system == "Windows":
            return _create_windows_shortcut(install_path)
        elif system == "Darwin":  # macOS
            return _create_macos_shortcut(install_path)
        elif system == "Linux":
            return _create_linux_shortcut(install_path)
        else:
            return False
    except Exception:
        return False


def _create_windows_shortcut(install_path: str) -> bool:
    """Create Windows desktop shortcut."""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "ConvAI Innovations.lnk")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = "-m convai_innovations"
        shortcut.WorkingDirectory = install_path
        shortcut.IconLocation = sys.executable
        shortcut.save()
        
        return True
    except ImportError:
        return False


def _create_macos_shortcut(install_path: str) -> bool:
    """Create macOS desktop shortcut."""
    # macOS shortcuts are more complex, would need .app bundle
    return False


def _create_linux_shortcut(install_path: str) -> bool:
    """Create Linux desktop shortcut."""
    try:
        desktop_dir = Path.home() / "Desktop"
        if not desktop_dir.exists():
            desktop_dir = Path.home() / ".local" / "share" / "applications"
        
        shortcut_path = desktop_dir / "convai-innovations.desktop"
        
        shortcut_content = f"""[Desktop Entry]
Name=ConvAI Innovations
Comment=Interactive LLM Training Academy
Exec={sys.executable} -m convai_innovations
Icon=applications-science
Terminal=false
Type=Application
Categories=Education;Science;
"""
        
        shortcut_path.write_text(shortcut_content)
        shortcut_path.chmod(0o755)
        
        return True
    except Exception:
        return False


def log_session_progress(session_id: str, completion_time: float, score: float = None):
    """
    Log session progress for analytics.
    
    Args:
        session_id: ID of completed session
        completion_time: Time taken to complete session (seconds)
        score: Optional score (0.0 to 1.0)
    """
    try:
        data_dir = setup_data_directory()
        log_file = data_dir / "session_log.txt"
        
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{session_id},{completion_time:.1f},{score or 'N/A'}\n")
    except Exception:
        pass  # Don't fail the application if logging fails


def cleanup_temp_files():
    """Clean up temporary files created by the application."""
    try:
        data_dir = setup_data_directory()
        temp_dir = data_dir / "temp"
        
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
    except Exception:
        pass


def get_gpu_info() -> Dict[str, str]:
    """Get GPU information for optimization."""
    gpu_info = {"available": False, "name": "None", "memory": "Unknown"}
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    except ImportError:
        pass
    
    return gpu_info


def optimize_for_system() -> Dict[str, any]:
    """Get optimization settings based on system capabilities."""
    settings = {
        "n_gpu_layers": -1,  # Use all GPU layers if available
        "n_ctx": 4096,       # Context size
        "n_threads": None,   # Auto-detect
        "use_mlock": False,  # Memory locking
        "use_mmap": True     # Memory mapping
    }
    
    # Adjust based on available memory
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_ram_gb < 4:
            settings["n_ctx"] = 2048  # Reduce context for low RAM
            settings["use_mmap"] = False
        elif available_ram_gb > 16:
            settings["use_mlock"] = True  # Use memory locking for high RAM
            
    except ImportError:
        pass
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        settings["n_gpu_layers"] = 0  # CPU only
    
    return settings


def generate_session_report(session_data: Dict) -> str:
    """Generate a formatted session report."""
    report = f"""
# ConvAI Innovations Session Report

## Session Information
- **Session ID**: {session_data.get('id', 'Unknown')}
- **Title**: {session_data.get('title', 'Unknown')}
- **Completion Status**: {'✅ Completed' if session_data.get('completed') else '⏳ In Progress'}

## Learning Objectives
"""
    
    objectives = session_data.get('learning_objectives', [])
    for i, objective in enumerate(objectives, 1):
        report += f"{i}. {objective}\n"
    
    report += f"""
## Progress Summary
- **Code Attempts**: {session_data.get('code_attempts', 0)}
- **Successful Runs**: {session_data.get('successful_runs', 0)}
- **AI Hints Used**: {session_data.get('hints_used', 0)}
- **Time Spent**: {session_data.get('time_spent', 'Unknown')}

Generated by ConvAI Innovations - Interactive LLM Training Academy
"""
    
    return report


def export_user_progress(progress_data: Dict, output_path: str) -> bool:
    """
    Export user progress to a file.
    
    Args:
        progress_data: User progress dictionary
        output_path: Path to output file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception:
        return False


def import_user_progress(input_path: str) -> Optional[Dict]:
    """
    Import user progress from a file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        Progress dictionary if successful, None otherwise
    """
    try:
        import json
        
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None