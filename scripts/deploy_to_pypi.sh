# 🚀 ConvAI Innovations - Simple PowerShell Deployment

# Set your tokens (get from https://pypi.org/manage/account/token/)
$env:PYPI_TEST_TOKEN = "pypi-your-test-token-here"
$env:PYPI_TOKEN = "pypi-your-production-token-here"

# Install build tools
pip install build twine wheel

# Navigate to project
cd convai-innovations

# ============================================================================
# 3-STEP DEPLOYMENT
# ============================================================================

# STEP 1: Build
python scripts/build.py

# STEP 2: Deploy to Test PyPI
python scripts/deploy.py --test

# STEP 3: Deploy to Production PyPI
python scripts/deploy.py

# ============================================================================
# ONE-LINER ALTERNATIVES
# ============================================================================

# Build and test deploy
python scripts/build.py; if ($LASTEXITCODE -eq 0) { python scripts/deploy.py --test }

# Build and production deploy
python scripts/build.py; if ($LASTEXITCODE -eq 0) { python scripts/deploy.py }