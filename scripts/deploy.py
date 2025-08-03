#!/usr/bin/env python3
"""
Deployment script for ConvAI Innovations PyPI package.

This script handles uploading the package to PyPI (or Test PyPI) using twine.
Supports token-based authentication.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import getpass


def run_command(command, check=True, env=None):
    """Run a shell command and handle errors."""
    print(f"🔧 Running: {command}")
    try:
        subprocess.run(command, shell=True, check=check, env=env)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        raise


def check_dist_files():
    """Check that distribution files exist."""
    print("📋 Checking distribution files...")
    
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("❌ No dist directory found. Run build first: python scripts/build.py")
        return False
    
    files = list(dist_dir.glob("*.tar.gz")) + list(dist_dir.glob("*.whl"))
    if not files:
        print("❌ No distribution files found. Run build first: python scripts/build.py")
        return False
    
    print("✅ Found distribution files:")
    for file_path in files:
        file_size = file_path.stat().st_size / (1024 * 1024)
        print(f"  - {file_path.name} ({file_size:.2f} MB)")
    
    return True


def get_pypi_token(test_pypi=False):
    """Get PyPI token from environment or user input."""
    env_var = "PYPI_TEST_TOKEN" if test_pypi else "PYPI_TOKEN"
    repo_name = "Test PyPI" if test_pypi else "PyPI"
    
    # Try to get token from environment
    token = os.environ.get(env_var)
    
    if token:
        print(f"✅ Using {repo_name} token from environment variable {env_var}")
        return token
    
    # Ask user for token
    print(f"🔑 {repo_name} token not found in environment variable {env_var}")
    print(f"Please enter your {repo_name} API token:")
    print("(You can get one from https://pypi.org/manage/account/token/)")
    if test_pypi:
        print("(For Test PyPI: https://test.pypi.org/manage/account/token/)")
    
    token = getpass.getpass("Token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return None
    
    return token


def check_package_version(test_pypi=False):
    """Check if package version already exists on PyPI."""
    print("🔍 Checking package version...")
    
    try:
        # Extract version from pyproject.toml
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Fallback for older Python
        except ImportError:
            print("⚠️  Cannot check version - tomllib/tomli not available")
            return True
    
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        
        package_name = data["project"]["name"]
        version = data["project"]["version"]
        
        print(f"📦 Package: {package_name} v{version}")
        
        # Check if version exists (simplified check)
        import urllib.request
        import json
        
        base_url = "https://test.pypi.org" if test_pypi else "https://pypi.org"
        url = f"{base_url}/pypi/{package_name}/json"
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read())
                existing_versions = list(data["releases"].keys())
                
                if version in existing_versions:
                    repo_name = "Test PyPI" if test_pypi else "PyPI"
                    print(f"❌ Version {version} already exists on {repo_name}")
                    print("Please update the version in pyproject.toml")
                    return False
                else:
                    print(f"✅ Version {version} is new")
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("✅ Package is new (doesn't exist on PyPI yet)")
                return True
            else:
                print(f"⚠️  Could not check existing versions: {e}")
                return True
    
    except Exception as e:
        print(f"⚠️  Could not check version: {e}")
        return True


def upload_to_pypi(token, test_pypi=False, skip_existing=False):
    """Upload package to PyPI using twine."""
    repo_name = "Test PyPI" if test_pypi else "PyPI"
    print(f"🚀 Uploading to {repo_name}...")
    
    # Prepare twine command
    if test_pypi:
        repository_url = "https://test.pypi.org/legacy/"
    else:
        repository_url = "https://upload.pypi.org/legacy/"
    
    # Build twine command
    cmd = f"twine upload --repository-url {repository_url} --username __token__ --password {token}"
    
    if skip_existing:
        cmd += " --skip-existing"
    
    cmd += " dist/*"
    
    # Set up environment
    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    
    try:
        run_command(cmd, env=env)
        print(f"✅ Successfully uploaded to {repo_name}!")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Upload to {repo_name} failed")
        return False


def show_post_upload_info(test_pypi=False):
    """Show information after successful upload."""
    repo_name = "Test PyPI" if test_pypi else "PyPI"
    base_url = "https://test.pypi.org" if test_pypi else "https://pypi.org"
    
    print("\n" + "="*60)
    print(f"🎉 DEPLOYMENT TO {repo_name.upper()} SUCCESSFUL!")
    print("="*60)
    
    print(f"📦 Package uploaded to {repo_name}")
    print(f"🔗 View at: {base_url}/project/convai-innovations/")
    
    if test_pypi:
        print("\n🧪 Testing installation from Test PyPI:")
        print("pip install --index-url https://test.pypi.org/simple/ convai-innovations")
    else:
        print("\n🎯 Installing from PyPI:")
        print("pip install convai-innovations")
    
    print("\n🚀 CLI Usage:")
    print("convai")
    print("convai --help")
    print("convai --version")
    
    print("\n📚 Import in Python:")
    print("from convai_innovations import convai")
    print("convai.main()")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy ConvAI Innovations to PyPI")
    parser.add_argument("--test", action="store_true", help="Deploy to Test PyPI instead of PyPI")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist")
    parser.add_argument("--force", action="store_true", help="Skip version check")
    parser.add_argument("--token", type=str, help="PyPI token (alternatively set PYPI_TOKEN env var)")
    
    args = parser.parse_args()
    
    repo_name = "Test PyPI" if args.test else "PyPI"
    print(f"🚀 Starting deployment to {repo_name}...")
    print("="*60)
    
    # Check distribution files exist
    if not check_dist_files():
        sys.exit(1)
    
    # Check package version (unless forced)
    if not args.force:
        if not check_package_version(args.test):
            print("\n💡 Tip: Use --force to skip version check")
            sys.exit(1)
    
    # Get PyPI token
    if args.token:
        token = args.token
        print(f"✅ Using provided {repo_name} token")
    else:
        token = get_pypi_token(args.test)
    
    if not token:
        sys.exit(1)
    
    # Confirm deployment
    if not args.force:
        print(f"\n⚠️  Ready to deploy to {repo_name}")
        print("This action cannot be undone!")
        response = input("Continue? [y/N]: ").strip().lower()
        
        if response not in ['y', 'yes']:
            print("❌ Deployment cancelled")
            sys.exit(0)
    
    # Upload to PyPI
    if upload_to_pypi(token, args.test, args.skip_existing):
        show_post_upload_info(args.test)
    else:
        print(f"\n❌ Deployment to {repo_name} failed")
        
        if not args.test:
            print("💡 Tip: Try deploying to Test PyPI first: python scripts/deploy.py --test")
        
        sys.exit(1)


if __name__ == "__main__":
    main()