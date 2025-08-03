#!/usr/bin/env python3
"""
Build script for ConvAI Innovations PyPI package.

This script handles building, testing, and preparing the package for distribution.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse


def run_command(command, check=True, capture_output=False):
    """Run a shell command and handle errors."""
    print(f"🔧 Running: {command}")
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=True, check=check)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        raise


def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning previous build artifacts...")
    
    # Directories to clean
    dirs_to_clean = [
        "build",
        "dist", 
        "*.egg-info",
        "convai_innovations/__pycache__",
        "convai_innovations/*.pyc",
        "tests/__pycache__",
        "tests/*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov"
    ]
    
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                print(f"  Removing directory: {path}")
                shutil.rmtree(path)
            elif path.is_file():
                print(f"  Removing file: {path}")
                path.unlink()


def install_build_deps():
    """Install build dependencies."""
    print("📦 Installing build dependencies...")
    
    build_deps = [
        "build>=0.10.0",
        "twine>=4.0.0", 
        "wheel>=0.38.0",
        "setuptools>=61.0"
    ]
    
    for dep in build_deps:
        # Remove quotes for cross-platform compatibility
        run_command(f"pip install {dep}")


def format_code():
    """Format code with black."""
    print("🎨 Formatting code with black...")
    try:
        run_command("black convai_innovations/ scripts/ tests/ --line-length 88")
        print("✅ Code formatting completed")
    except subprocess.CalledProcessError:
        print("⚠️  Black formatting failed - continuing anyway")


def lint_code():
    """Lint code with flake8."""
    print("🔍 Linting code with flake8...")
    try:
        run_command("flake8 convai_innovations/ --max-line-length=88 --ignore=E203,W503")
        print("✅ Linting passed")
    except subprocess.CalledProcessError:
        print("⚠️  Linting found issues - continuing anyway")


def type_check():
    """Type check with mypy."""
    print("🔬 Type checking with mypy...")
    try:
        run_command("mypy convai_innovations/ --ignore-missing-imports")
        print("✅ Type checking passed")
    except subprocess.CalledProcessError:
        print("⚠️  Type checking found issues - continuing anyway")


def run_tests():
    """Run tests with pytest."""
    print("🧪 Running tests...")
    try:
        run_command("pytest tests/ -v --cov=convai_innovations --cov-report=html")
        print("✅ All tests passed")
    except subprocess.CalledProcessError:
        print("❌ Tests failed")
        return False
    return True


def check_package_structure():
    """Check that all required files exist."""
    print("📋 Checking package structure...")
    
    required_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "convai_innovations/__init__.py",
        "convai_innovations/cli.py",
        "convai_innovations/convai.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print("✅ Package structure is valid")
    return True


def build_package():
    """Build the package."""
    print("🏗️  Building package...")
    
    # Build source distribution and wheel
    run_command("python -m build")
    
    # Check if build artifacts exist
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ Build failed - no dist directory created")
        return False
    
    files = list(dist_dir.glob("*"))
    if not files:
        print("❌ Build failed - no files in dist directory")
        return False
    
    print("✅ Package built successfully:")
    for file_path in files:
        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {file_path.name} ({file_size:.2f} MB)")
    
    return True


def check_package():
    """Check the built package with twine."""
    print("🔍 Checking package with twine...")
    
    try:
        run_command("twine check dist/*")
        print("✅ Package check passed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Package check failed")
        return False


def show_build_summary():
    """Show build summary."""
    print("\n" + "="*60)
    print("📊 BUILD SUMMARY")
    print("="*60)
    
    dist_dir = Path("dist")
    if dist_dir.exists():
        files = list(dist_dir.glob("*"))
        if files:
            print(f"✅ Built {len(files)} package(s):")
            for file_path in files:
                file_size = file_path.stat().st_size / (1024 * 1024)
                print(f"   📦 {file_path.name} ({file_size:.2f} MB)")
        else:
            print("❌ No packages built")
    else:
        print("❌ No dist directory found")
    
    print("\n🚀 Ready for deployment!")
    print("Next steps:")
    print("  1. Test install: pip install dist/*.whl")
    print("  2. Deploy to PyPI: python scripts/deploy.py")
    print("  3. Deploy to Test PyPI: python scripts/deploy.py --test")


def main():
    """Main build function."""
    parser = argparse.ArgumentParser(description="Build ConvAI Innovations package")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-lint", action="store_true", help="Skip code linting")
    parser.add_argument("--skip-format", action="store_true", help="Skip code formatting")
    parser.add_argument("--clean-only", action="store_true", help="Only clean, don't build")
    parser.add_argument("--fast", action="store_true", help="Fast build (skip tests, lint, format)")
    
    args = parser.parse_args()
    
    print("🚀 Starting ConvAI Innovations build process...")
    print("="*60)
    
    # Always clean first
    clean_build()
    
    if args.clean_only:
        print("✅ Clean completed")
        return
    
    # Install dependencies
    install_build_deps()
    
    # Check package structure
    if not check_package_structure():
        sys.exit(1)
    
    # Quality checks (unless fast build)
    if not args.fast:
        if not args.skip_format:
            format_code()
        
        if not args.skip_lint:
            lint_code()
        
        type_check()
        
        if not args.skip_tests:
            if not run_tests():
                print("❌ Build failed due to test failures")
                sys.exit(1)
    
    # Build the package
    if not build_package():
        sys.exit(1)
    
    # Check the package
    if not check_package():
        sys.exit(1)
    
    # Show summary
    show_build_summary()
    
    print("\n✅ Build completed successfully!")


if __name__ == "__main__":
    main()