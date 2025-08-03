#!/usr/bin/env python3
"""
ConvAI Innovations CLI - Command Line Interface

This module provides the command line interface for launching the
LLM Training Learning Dashboard.
"""

import argparse
import sys
import os
from pathlib import Path

def print_banner():
    """Print the ConvAI Innovations banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🧠 ConvAI Innovations                     ║
║              Interactive LLM Training Academy                ║
║                                                              ║
║        Learn to build Language Models from scratch!         ║
║              With AI Mentor Sandra by your side             ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import tkinter
    except ImportError:
        missing_deps.append("tkinter (usually comes with Python)")
    
    try:
        import llama_cpp
    except ImportError:
        missing_deps.append("llama-cpp-python")
    
    # Optional dependencies
    optional_missing = []
    try:
        import kokoro
        import torch
        import sounddevice
    except ImportError:
        optional_missing.append("audio features (kokoro-tts, torch, sounddevice)")
    
    if missing_deps:
        print("❌ Missing required dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies and try again.")
        return False
    
    if optional_missing:
        print("⚠️  Optional features not available:")
        for dep in optional_missing:
            print(f"   - {dep}")
        print("Install with: pip install convai-innovations[audio]")
        print()
    
    return True

def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="ConvAI Innovations - Interactive LLM Training Academy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  convai                    # Launch the application
  convai --no-banner       # Launch without banner
  convai --model-path /path/to/model.gguf  # Use custom model
  convai --version         # Show version information
  convai --check-deps      # Check dependencies

For more information, visit: https://github.com/ConvAI-Innovations/ailearning
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"ConvAI Innovations v{get_version()}"
    )
    
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the startup banner"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a custom LLM model file (.gguf)"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if all dependencies are installed"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom directory for storing models and data"
    )
    
    return parser

def get_version():
    """Get the package version."""
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "unknown"

def setup_environment(args):
    """Setup the environment based on command line arguments."""
    if args.debug:
        os.environ["CONVAI_DEBUG"] = "1"
        print("🐛 Debug mode enabled")
    
    if args.data_dir:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            try:
                data_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created data directory: {data_path}")
            except Exception as e:
                print(f"❌ Failed to create data directory: {e}")
                return False
        os.environ["CONVAI_DATA_DIR"] = str(data_path)
    
    return True

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle special commands first
    if args.check_deps:
        if not args.no_banner:
            print_banner()
        print("🔍 Checking dependencies...\n")
        if check_dependencies():
            print("✅ All required dependencies are available!")
        sys.exit(0)
    
    # Print banner unless disabled
    if not args.no_banner:
        print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    if not setup_environment(args):
        sys.exit(1)
    
    # Import and run the main application
    try:
        print("🚀 Starting ConvAI Innovations...")
        print("📚 Loading learning sessions...")
        
        # Import the main application components
        from .convai import main as app_main
        
        # Override model path if specified
        if args.model_path:
            if not Path(args.model_path).exists():
                print(f"❌ Model file not found: {args.model_path}")
                sys.exit(1)
            print(f"🤖 Using custom model: {args.model_path}")
            # You might want to pass this to the app somehow
            os.environ["CONVAI_MODEL_PATH"] = args.model_path
        
        print("✅ Launching application...")
        
        # Run the main application
        app_main()
        
    except KeyboardInterrupt:
        print("\n👋 ConvAI Innovations stopped by user.")
        sys.exit(0)
    except Exception as e:
        if args.debug:
            import traceback
            print("\n❌ An error occurred:")
            traceback.print_exc()
        else:
            print(f"\n❌ An error occurred: {e}")
            print("Use --debug for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()