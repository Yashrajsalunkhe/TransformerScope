# TransformerScope 🔍

Interactive tool for analyzing attention patterns in transformer models, helping understand how LLMs process different types of text inputs.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🎯 **Layer-wise attention visualization** - Explore attention patterns across transformer layers
- 🔥 **Interactive attention heatmaps** - See which tokens attend to which other tokens  
- 📊 **Token importance scoring** - Identify the most important tokens in your text
- 📈 **Layer comparison analysis** - Compare attention entropy and patterns across layers
- 🧠 **Attention head analysis** - Understand how different heads specialize
- 🌊 **Attention flow diagrams** - Visualize attention connections as network graphs
- 📱 **User-friendly Streamlit interface** - No coding required to use

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Yashrajsalunkhe/TransformerScope.git
cd TransformerScope
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 How to Use

1. **Select a Model**: Choose from DistilBERT, BERT, or GPT-2 models
2. **Enter Text**: Use sample texts or enter your own
3. **Choose Analysis**: Pick from 5 different visualization types:
   - Attention Heatmap
   - Token Importance  
   - Layer Comparison
   - Head Patterns
   - Attention Flow
4. **Explore**: Interact with the visualizations and adjust layers/heads

## 🎨 Visualization Types

### Attention Heatmap
Interactive heatmap showing which tokens attend to which others
- Hover for detailed attention weights
- Select specific layers and heads
- Normalize attention for better comparison

### Token Importance
Bar chart showing total attention received by each token
- Identifies the most "important" tokens in context
- Color-coded by attention strength

### Layer Comparison  
Line plots comparing attention patterns across layers
- Attention entropy (how spread out attention is)
- Maximum attention values
- Helps understand model depth utilization

### Head Pattern Analysis
Compare different attention heads within a layer
- Self-attention vs cross-attention patterns
- Head specialization metrics
- Entropy and focus measurements

### Attention Flow Diagram
Network visualization of strong attention connections
- Nodes represent tokens
- Edges show attention flow above threshold
- Circular layout for clear visualization

## 🏗️ Architecture

```
src/
├── attention_extractor.py  # Model loading and attention extraction
├── visualizer.py          # Plotly visualization functions  
└── __init__.py           # Package initialization

app.py                    # Main Streamlit application
requirements.txt          # Python dependencies
examples/                 # Sample texts for testing
```

## 🔧 Supported Models

- **DistilBERT** (`distilbert-base-uncased`) - Lightweight, fast
- **BERT** (`bert-base-uncased`) - Classic transformer model  
- **DistilGPT-2** (`distilgpt2`) - Decoder-only architecture
- **GPT-2** (`gpt2`) - Generative pre-trained transformer

## 📊 Understanding the Visualizations

### What is Attention?
Attention mechanisms allow models to focus on relevant parts of the input when processing each token. High attention weights indicate strong relationships between tokens.

### Interpreting Patterns
- **High self-attention**: Token attending to itself
- **Sequential attention**: Following word order patterns  
- **Semantic attention**: Focus on semantically related tokens
- **Positional patterns**: Position-based attention in early layers

## 🛠️ Technical Details

- Built with **Streamlit** for the web interface
- **Hugging Face Transformers** for model loading
- **PyTorch** for tensor operations
- **Plotly** for interactive visualizations
- **Caching** for improved performance

## 🎯 Use Cases

- **Education**: Understand how transformers work
- **Research**: Analyze model behavior on specific inputs
- **Debugging**: Identify attention issues in fine-tuned models
- **Interpretability**: Explain model decisions through attention

## 📝 Example Texts

The app includes several pre-loaded examples:
- Simple sentences for basic analysis
- Pronoun resolution examples
- Complex sentences with multiple clauses
- Questions and technical text

## 🤝 Contributing

Contributions welcome! Ideas for improvements:
- Support for more model architectures
- Additional visualization types
- Batch processing capabilities
- Export functionality for visualizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Streamlit for the amazing web framework
- The attention mechanism research community

