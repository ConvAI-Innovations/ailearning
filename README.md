# 🧠 ConvAI Innovations Dashboard - Interactive LLM Training Academy

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Learn to build Large Language Models from scratch through hands-on coding sessions with AI mentor Sandra!**

ConvAI Innovations dashboard is a comprehensive educational platform that takes you from Python fundamentals to training your own LLMs. Experience interactive learning with real-time AI feedback, text-to-speech guidance, and a structured curriculum covering everything from neural networks to transformer architecture.

## ✨ Features

### 🎓 **Comprehensive Learning Path**
- **13 Interactive Sessions**: From Python basics to LLM inference
- **Hands-on Coding**: Type code manually to build muscle memory
- **Progressive Difficulty**: Each session builds on previous knowledge
- **Real-world Applications**: Learn concepts used in ChatGPT, GPT-4, and other LLMs

### 🤖 **AI-Powered Learning**
- **Sandra, Your AI Mentor**: Get personalized feedback on your code
- **Smart Hints**: Context-aware suggestions when you're stuck
- **Error Analysis**: Intelligent debugging assistance
- **Text-to-Speech**: Optional audio guidance for immersive learning

### 💻 **Advanced IDE Features**
- **Syntax Highlighting**: Professional code editor with line numbers
- **Auto-indentation**: Smart code formatting
- **Real-time Execution**: Run Python code instantly with output
- **Save/Load Projects**: Manage your learning progress

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

# Use custom model
convai --model-path /path/to/your/model.gguf

# Check dependencies
convai --check-deps
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
- **Python 3.8+**
- **tkinter** (usually included with Python)
- **llama-cpp-python** (for AI mentor)
- **PyTorch** (for neural network operations)
- **NumPy** (for numerical computations)

### Optional Dependencies
- **kokoro-tts** (for text-to-speech features)
- **sounddevice** (for audio output)

### Hardware Requirements
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space (for AI model download)
- **GPU**: Optional, but recommended for faster AI responses

## 🛠️ Advanced Usage

### Custom Model Configuration

```bash
# Use your own model
convai --model-path /path/to/custom/model.gguf

# Custom data directory
convai --data-dir /path/to/custom/data

# Debug mode
convai --debug
```

### Environment Variables

```bash
# Set custom model path
export CONVAI_MODEL_PATH="/path/to/model.gguf"

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
- **Discussions**: [Community discussions](https://github.com/ConvAI-Innovations/ailearning/discussions)
- **Email**: contact@convai-innovations.com

### Common Issues

**Q: The AI mentor isn't working**
A: Make sure you have `llama-cpp-python` installed and a stable internet connection for model download.

**Q: No audio from Sandra**
A: Install audio dependencies: `pip install convai-innovations[audio]`

**Q: Application crashes on startup**
A: Check dependencies with `convai --check-deps` and ensure Python 3.8+.

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

This is a copyleft license that requires derivative works to also be licensed under GPL-3.0.

## 🙏 Acknowledgments

- **Hugging Face** for model hosting and transformers library
- **llama.cpp** team for efficient LLM inference
- **PyTorch** team for the deep learning framework
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