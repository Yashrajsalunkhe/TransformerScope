"""
Attention extraction utilities for transformer models
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import streamlit as st

@st.cache_resource
def load_model_and_tokenizer(model_name):
    """
    Load and cache the model and tokenizer
    Uses Streamlit caching to avoid reloading
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.eval()  # Set to evaluation mode
        return tokenizer, model
    except Exception as e:
        if "404" in str(e) or "Repository not found" in str(e):
            st.error(f"Model '{model_name}' not found. Please check the model name or try a different model.")
        elif "connection" in str(e).lower() or "timeout" in str(e).lower():
            st.error(f"Network error loading {model_name}. Check your internet connection and try again.")
        elif "memory" in str(e).lower() or "cuda" in str(e).lower():
            st.error(f"Memory error loading {model_name}. Try a smaller model like 'distilbert-base-uncased'.")
        else:
            st.error(f"Error loading model {model_name}: {str(e)}")
            st.info("💡 Try selecting a different model from the dropdown.")
        return None, None

def extract_attention_weights(model, tokenizer, text, max_length=512):
    """
    Extract attention weights from the model for given text
    
    Returns:
    - attentions: tuple of attention tensors (layers, heads, seq_len, seq_len)
    - tokens: list of token strings
    - token_ids: tensor of token IDs
    """
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=max_length, 
        truncation=True,
        padding=True
    )
    
    # Convert token IDs to tokens for display
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Get model outputs with attention
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract attention weights
    attentions = outputs.attentions  # Tuple of tensors (num_layers,)
    # Each tensor shape: (batch_size, num_heads, seq_len, seq_len)
    
    return attentions, tokens, inputs["input_ids"]

def process_attention_data(attentions, tokens, layer_idx=0, head_idx=0):
    """
    Process attention data for visualization
    
    Args:
    - attentions: attention weights from model
    - tokens: list of token strings
    - layer_idx: which layer to extract
    - head_idx: which attention head to extract
    
    Returns:
    - attention_matrix: 2D numpy array for heatmap
    - token_labels: cleaned token labels
    """
    if layer_idx >= len(attentions):
        raise ValueError(f"Layer {layer_idx} not available. Model has {len(attentions)} layers.")
    
    # Get attention for specific layer
    layer_attention = attentions[layer_idx]  # Shape: (1, num_heads, seq_len, seq_len)
    
    if head_idx >= layer_attention.shape[1]:
        raise ValueError(f"Head {head_idx} not available. Layer has {layer_attention.shape[1]} heads.")
    
    # Extract specific head
    attention_matrix = layer_attention[0, head_idx, :, :].numpy()  # Shape: (seq_len, seq_len)
    
    # Clean token labels (remove special tokens formatting)
    token_labels = []
    for token in tokens:
        if token.startswith('##'):
            token_labels.append(token[2:])  # Remove ## prefix
        elif token in ['[CLS]', '[SEP]', '<s>', '</s>']:
            token_labels.append(f"[{token}]")
        else:
            token_labels.append(token)
    
    return attention_matrix, token_labels

def get_attention_stats(attentions):
    """
    Get statistics about attention patterns
    """
    stats = {}
    
    stats['num_layers'] = len(attentions)
    stats['num_heads'] = attentions[0].shape[1]
    stats['sequence_length'] = attentions[0].shape[2]
    
    # Calculate attention entropy for each layer (measure of attention spread)
    layer_entropies = []
    for layer_attention in attentions:
        # Average across batch and heads, then calculate entropy
        avg_attention = layer_attention.mean(dim=(0, 1))  # Shape: (seq_len, seq_len)
        
        # Calculate entropy for each token's attention distribution
        entropies = []
        for i in range(avg_attention.shape[0]):
            attention_dist = avg_attention[i]
            # Add small epsilon to avoid log(0)
            entropy = -torch.sum(attention_dist * torch.log(attention_dist + 1e-8))
            entropies.append(entropy.item())
        
        layer_entropies.append(np.mean(entropies))
    
    stats['layer_entropies'] = layer_entropies
    
    # Find most attended tokens overall
    total_attention = torch.zeros(stats['sequence_length'])
    for layer_attention in attentions:
        # Sum attention received by each token across all layers and heads
        total_attention += layer_attention.sum(dim=(0, 1, 2))  # Sum across batch, heads, and source positions
    
    stats['total_attention'] = total_attention.numpy()
    
    return stats

def analyze_head_patterns(attentions, tokens):
    """
    Analyze patterns across different attention heads
    """
    patterns = {}
    
    for layer_idx, layer_attention in enumerate(attentions):
        layer_patterns = {}
        
        for head_idx in range(layer_attention.shape[1]):
            head_attention = layer_attention[0, head_idx, :, :].numpy()
            
            # Calculate head specialization metrics
            # 1. Attention to self (diagonal)
            self_attention = np.diag(head_attention).mean()
            
            # 2. Attention spread (entropy)
            attention_entropy = 0
            for i in range(head_attention.shape[0]):
                dist = head_attention[i]
                dist = dist / (dist.sum() + 1e-8)  # Normalize
                entropy = -np.sum(dist * np.log(dist + 1e-8))
                attention_entropy += entropy
            attention_entropy /= head_attention.shape[0]
            
            # 3. Maximum attention value
            max_attention = head_attention.max()
            
            layer_patterns[f'head_{head_idx}'] = {
                'self_attention': self_attention,
                'entropy': attention_entropy,
                'max_attention': max_attention,
                'attention_matrix': head_attention
            }
        
        patterns[f'layer_{layer_idx}'] = layer_patterns
    
    return patterns
