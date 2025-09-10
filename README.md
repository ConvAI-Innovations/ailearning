# 🧠 ConvAI Innovations Dashboard - Interactive LLM Training Academy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Learn to build Large Language Models from scratch through hands-on coding sessions with AI mentor Sandra!**

ConvAI Innovations dashboard is a comprehensive educational platform that takes you from Python fundamentals to training your own LLMs. Experience interactive learning with real-time AI feedback powered by Hugging Face Transformers, offline multi-language support via Argostranslate, text-to-speech guidance, and a structured curriculum covering everything from neural networks to transformer architecture.

## ✨ Features

### 🎓 **Comprehensive Learning Path**
- **13 Interactive Sessions**: From Python basics to LLM inference
- **Hands-on Coding**: Type code manually to build muscle memory
- **Progressive Difficulty**: Each session builds on previous knowledge
- **Real-world Applications**: Learn concepts used in ChatGPT, GPT-4, and other LLMs

### 🤖 **AI-Powered Learning (2025 Enhanced)**
- **Sandra, Your AI Mentor**: Powered by Qwen3-0.6B via Hugging Face Transformers
- **Multi-language Support**: Learn in 6+ languages with offline neural translation
- **Smart Hints**: Context-aware suggestions when you're stuck
- **Error Analysis**: Intelligent debugging assistance with real-time feedback
- **Advanced TTS**: Kokoro-powered text-to-speech in multiple languages
- **Offline Translation**: Argostranslate ensures privacy and fast responses

### 💻 **Advanced IDE Features**
- **Syntax Highlighting**: Professional code editor with line numbers
- **Auto-indentation**: Smart code formatting
- **Real-time Execution**: Run Python code instantly with output
- **Save/Load Projects**: Manage your learning progress
- **AI Code Generation**: Built-in code generator with editable output
- **Modern UI**: Responsive interface with real-time progress tracking

### 📚 **Complete Curriculum**

| Session | Topic | What You'll Learn |
|---------|-------|------------------|
| 🐍 | Python Fundamentals | Variables, functions, classes for ML |
| 🔢 | PyTorch & NumPy | Tensor operations, mathematical foundations |
| 🧠 | Neural Networks | Perceptrons, multi-layer networks, forward propagation |
| ⬅️ | Backpropagation | How neural networks learn, gradient computation |
| 🛡️ | Regularization | Preventing overfitting, dropout, batch norm |
| 📉 | Loss Functions & Optimizers | Cross-entropy, MSE, SGD, Adam, AdamW |
| 🏗️ | LLM Architecture | Transformers, attention mechanisms, embeddings |
| 🔤 | Tokenization & BPE | Text preprocessing, byte pair encoding |
| 🎯 | RoPE & Self-Attention | Rotary position encoding, modern attention |
| ⚖️ | RMS Normalization | Advanced normalization techniques |
| 🔄 | FFN & Activations | Feed-forward networks, GELU, SiLU |
| 🚂 | Training LLMs | Complete training pipeline, optimization |
| 🎯 | Inference & Generation | Text generation, sampling strategies |

## 🎯 What's New in 2025

### 🚀 **Modern AI Architecture**
- **Hugging Face Integration**: Direct access to state-of-the-art models
- **Qwen3-0.6B AI Mentor**: Fast, efficient, and highly capable language model
- **Background Loading**: Non-blocking AI initialization for smooth startup
- **Smart Caching**: Optimized model and translation caching

### 🌍 **Multi-Language Learning**
- **6 Supported Languages**: English, Spanish, French, Hindi, Italian, Portuguese
- **Offline Translation**: Argostranslate neural translation (no API keys needed)
- **Cultural Adaptation**: Learning content adapted for different regions
- **Voice Support**: Multi-language text-to-speech with Kokoro

### 🛡️ **Privacy & Performance**
- **Offline-First**: All AI processing happens on your machine
- **No Data Collection**: Your code and progress stay private
- **Fast Response**: Optimized caching for instant feedback
- **Lightweight**: Efficient resource usage with modern compression

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install convai-innovations

# With audio features (recommended)
pip install convai-innovations[audio]

# Full installation with development tools
pip install convai-innovations[all]
```

### Launch the Academy

```bash
# Start the interactive learning dashboard
convai

# Launch without banner
convai --no-banner

# Use custom Hugging Face model
convai --model-path "microsoft/DialoGPT-medium"

# Check dependencies
convai --check-deps

# Enable debug mode for troubleshooting
convai --debug
```

### Python API

```python
from convai_innovations import convai

# Launch the application
convai.main()

# Or use the alternative entry point
convai.run_convai()
```

## 📋 System Requirements

### Required Dependencies
- **Python 3.10+**
- **tkinter** (usually included with Python)
- **transformers** (for AI mentor - Hugging Face models)
- **torch** (for neural network operations and model inference)
- **requests** (for downloading models and translation packages)
- **argostranslate** (for offline multi-language support)

### Optional Dependencies
- **kokoro-tts** (for advanced text-to-speech features)
- **sounddevice** (for audio output)
- **argostranslate** (for offline translation - enables multi-language support)

### Hardware Requirements
- **Memory**: 6GB RAM minimum, 12GB recommended
- **Storage**: 3GB free space (for AI model and translation packages)
- **GPU**: Optional, but recommended for faster AI responses (CUDA support via PyTorch)

## 🛠️ Advanced Usage

### Custom Model Configuration

```bash
# Use your own Hugging Face model
convai --model-path "your-username/your-model-name"

# Custom data directory
convai --data-dir /path/to/custom/data

# Debug mode
convai --debug
```

### Environment Variables

```bash
# Set custom Hugging Face model
export CONVAI_MODEL_PATH="microsoft/DialoGPT-medium"

# Custom data directory
export CONVAI_DATA_DIR="/path/to/data"

# Enable debug mode
export CONVAI_DEBUG="1"
```

## 🎓 Learning Tips

### For Beginners
1. **Start with Python Fundamentals** - Even if you know Python, review ML-specific concepts
2. **Type Code Manually** - Don't copy-paste; typing builds muscle memory
3. **Use Sandra's Hints** - The AI mentor provides context-aware help
4. **Practice Regularly** - Consistency is key to mastering LLM concepts

### For Advanced Users
1. **Experiment with Code** - Modify examples to deepen understanding
2. **Ask Questions** - Use the AI mentor to explore advanced topics
3. **Build Projects** - Apply learned concepts to your own projects
4. **Contribute** - Share improvements and new sessions

## 🔧 Development

### Build from Source

```bash
# Clone the repository
git clone https://github.com/ConvAI-Innovations/ailearning.git
cd convai-innovations

# Install in development mode
pip install -e .[dev]

# Run tests
python scripts/build.py

# Build package
python scripts/build.py --skip-tests

# Deploy to Test PyPI
python scripts/deploy.py --test
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Write tests
5. Submit a pull request

## 📖 Documentation

- **Full Documentation**: [convai-innovations.readthedocs.io](https://convai-innovations.readthedocs.io/)
- **API Reference**: [API Documentation](https://convai-innovations.readthedocs.io/en/latest/api/)
- **Tutorials**: [Learning Guides](https://convai-innovations.readthedocs.io/en/latest/tutorials/)
- **Examples**: [Code Examples](https://github.com/ConvAI-Innovations/ailearning/tree/main/examples)

## 🆘 Support

### Getting Help
- **GitHub Issues**: [Report bugs or request features](https://github.com/ConvAI-Innovations/ailearning/issues)
- **Email**: support@convai-innovations.com

### Common Issues

**Q: The AI mentor isn't working**
A: Make sure you have `transformers` and `torch` installed and a stable internet connection for Hugging Face model download.

**Q: No audio from Sandra**
A: Install audio dependencies: `pip install convai-innovations[audio]`

**Q: Application crashes on startup**
A: Check dependencies with `convai --check-deps` and ensure Python 3.10+.

**Q: Model download is slow or fails**
A: The first launch downloads ~1.5GB from Hugging Face. Ensure stable internet and sufficient storage.

**Q: Translation features not working**
A: Install Argostranslate: `pip install argostranslate`. First launch will download translation packages.

**Q: UI becomes unresponsive during startup**
A: This has been fixed in v1.1.3+ with background loading. Update to the latest version.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

This is a copyleft license that requires derivative works to also be licensed under GPL-3.0.

## 🙏 Acknowledgments

- **Hugging Face** for model hosting and transformers library
- **Qwen Team** for the efficient Qwen3-0.6B model used as AI mentor
- **PyTorch** team for the deep learning framework
- **Argostranslate** for offline neural translation
- **Kokoro TTS** for natural-sounding text-to-speech
- **The open-source AI community** for inspiration and support

## 🌟 Star History

If you find ConvAI Innovations dashboard helpful, please consider giving it a star! ⭐

---

**Ready to become an LLM expert? Start your journey today!**

```bash
pip install convai-innovations[audio]
convai
```

*Happy Learning! 🚀*