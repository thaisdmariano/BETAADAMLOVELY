# Adam Lovely AI Chatbot 🤖💜

An advanced AI chatbot powered by the custom INSEPA framework with deep learning capabilities.

## 🎯 Overview

Adam Lovely is an adaptive AI chatbot that learns from user interactions without relying on external models like GPT or BERT. It uses a custom transformer-based architecture with the INSEPA (Intelligent Neural System for Enhanced Pattern Analysis) framework.

## ✨ Recent Enhancements

### Learning & Productivity Improvements

1. **Enhanced Weighted Choice Mechanism** 🎯
   - Exponential weighting for liked responses (2^n weight for n likes)
   - Significantly stronger preference for user-approved responses
   - Better learning from positive feedback

2. **Optimized Training Process** 🚀
   - Learning rate scheduling with automatic adjustment
   - Early stopping to prevent overfitting
   - Real-time monitoring of training and validation loss
   - Faster convergence with adaptive learning rates

3. **Improved Data Augmentation** 📊
   - Enhanced variation generation (70% probability)
   - Richer training data through token variations
   - Better generalization capabilities

4. **Better Autoencoder Training** 🧠
   - Unsupervised learning for new response generation
   - Learning rate scheduling for optimal convergence
   - Early stopping mechanism

5. **Robust Input Parsing** 🔍
   - Improved emoji and reaction detection
   - Better handling of reactions at various positions
   - More accurate text/emotion separation

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run BETADAMLOVELY.py
```

## 📚 Key Features

### Custom INSEPA Framework
- **No External AI Dependencies**: Pure custom implementation
- **Field-Based Tokenization**: E (Expression), RE (Reaction), CE (Context), PIDE (Internal Thought)
- **N-gram Embeddings**: Character-level pattern recognition
- **Transformer Architecture**: Self-attention mechanisms for context understanding
- **Multi-Head Outputs**: Text, emoji, context, and position prediction

### Learning Mechanisms
- **User Feedback Integration**: Like system with exponential weighting
- **Variation System**: Token-level variations from the unconscious memory (INCO)
- **Autoencoder Generation**: Unsupervised creation of new response patterns
- **Adaptive Training**: Automatic learning rate adjustment

### Memory System
- **Conscious Memory**: Structured blocks with input-output patterns
- **Unconscious Memory**: Token variations and alternative expressions
- **Persistent Storage**: JSON-based memory with automatic backups

## 🏗️ Architecture

```
Input Text + Reaction + Context
         ↓
    Tokenization (INSEPA)
         ↓
    Embeddings (Value, Mother, Position)
         ↓
    Transformer Encoder
         ↓
    Generative Decoder
         ↓
    Multi-Head Output
         ↓
Response Text + Emoji + Context
```

## 📖 Usage

### Creating a New IM (Intelligence Module)

1. Access "Gerenciar IMs" (requires admin password: `adam123`)
2. Select "Criar novo IM"
3. Fill in the IM details (name, gender, voice)
4. Add blocks using the INSEPA template format

### Training

1. Select "Treinar" from the menu
2. Choose the IM domain
3. Training will automatically use learning rate scheduling and early stopping
4. Monitor real-time metrics: Train Loss, Val Loss, Learning Rate

### Conversing

1. Select "Conversar" from the menu
2. Choose the IM to chat with
3. Type your message (optionally include emoji reactions)
4. Use 👍 Like button to reinforce good responses
5. Special commands:
   - `sair`: Exit conversation
   - `reiniciar`: Restart conversation
   - `insight`: Get AI reasoning about last response

## 🔧 Configuration

Edit constants at the top of `BETADAMLOVELY.py`:

```python
EMBED_DIM = 16        # Embedding dimension
HIDDEN_DIM = 64       # Hidden layer size
LATENT_DIM = 128      # Autoencoder latent space
PATIENCE = 5          # Early stopping patience
BATCH_SIZE = 8        # Training batch size
LR = 1e-3             # Initial learning rate
EPOCHS = 50           # Maximum training epochs
N_GRAM = 2            # N-gram size for tokenization
```

## 📊 Performance Optimizations

- **Learning Rate Scheduling**: Automatic reduction when loss plateaus
- **Early Stopping**: Prevents overfitting and saves time
- **Batch Processing**: Efficient data loading and processing
- **Gradient Clipping**: Stable training (inherent in Adam optimizer)
- **Checkpoint Saving**: Only saves best models to reduce disk usage

## 🔐 Security

- Admin password protection for IM management
- Automated backup system for memory files
- Separated conscious and unconscious memory storage

## 📝 File Structure

```
BETAADAMLOVELY/
├── BETADAMLOVELY.py              # Main application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
├── Adam_Lovely_memory_v2.json    # Conscious memory (auto-created)
├── Adam_Lovely_inconscious_v2.json # Unconscious memory (auto-created)
└── insepa_*.pt                   # Trained models (auto-created)
```

## 🤝 Contributing

This is a custom framework without external AI dependencies. When contributing:
- Maintain backward compatibility with JSON structures
- Follow the existing INSEPA conventions
- Add tests for new features
- Document significant changes

## 📄 License

This project implements a custom AI framework (INSEPA) for educational and research purposes.

## 🙏 Acknowledgments

- Built with PyTorch for deep learning
- Streamlit for the interactive UI
- Custom INSEPA framework for intelligent pattern analysis

## 📞 Support

For issues or questions:
1. Check the code comments and docstrings
2. Review the Statistics menu for system insights
3. Use the Backup menu for data management
4. Verify requirements.txt dependencies are installed

---

**Note**: This chatbot learns and adapts based on user interactions. The more you interact and provide feedback (likes), the better it becomes at generating relevant responses! 🚀
