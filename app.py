import streamlit as st
import torch
import numpy as np
from src.attention_extractor import (
    load_model_and_tokenizer, 
    extract_attention_weights, 
    process_attention_data,
    get_attention_stats,
    analyze_head_patterns
)
from src.visualizer import (
    create_attention_heatmap,
    create_token_attention_bar,
    create_layer_comparison,
    create_head_pattern_comparison,
    create_attention_flow_diagram
)

# Page configuration
st.set_page_config(
    page_title="LLM Attention Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 LLM Attention Visualizer")
st.markdown("**Interactive tool for analyzing attention patterns in transformer models**")
st.markdown("---")

# Sidebar for model configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
model_options = [
    "distilbert-base-uncased",
    "bert-base-uncased", 
    "distilgpt2",
    "gpt2"
]

selected_model = st.sidebar.selectbox(
    "Choose Model:",
    model_options,
    index=0,
    help="Select a pre-trained transformer model to analyze"
)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Text")
    
    # Sample texts for quick testing
    sample_texts = {
        "Simple": "The cat sat on the mat.",
        "Complex": "The quick brown fox jumps over the lazy dog while the sun sets beautifully.",
        "Question": "What is the capital of France and why is it important?",
        "Attention Example": "John gave Mary the book. She thanked him for it.",
        "Custom": ""
    }
    
    text_choice = st.radio(
        "Choose sample text or enter custom:",
        list(sample_texts.keys()),
        horizontal=True
    )
    
    if text_choice == "Custom":
        input_text = st.text_area(
            "Enter your text:",
            height=100,
            placeholder="Type or paste your text here..."
        )
    else:
        input_text = st.text_area(
            "Text to analyze:",
            value=sample_texts[text_choice],
            height=100
        )

with col2:
    st.subheader("🎯 Analysis Options")
    
    analysis_type = st.selectbox(
        "Analysis Type:",
        ["Attention Heatmap", "Token Importance", "Layer Comparison", "Head Patterns", "Attention Flow"]
    )
    
    normalize_attention = st.checkbox("Normalize Attention", value=True)
    show_stats = st.checkbox("Show Statistics", value=True)

# Analysis button
if st.button("🚀 Analyze Attention Patterns", type="primary"):
    if input_text.strip():
        with st.spinner(f"Loading {selected_model} and analyzing attention..."):
            try:
                # Load model and tokenizer
                tokenizer, model = load_model_and_tokenizer(selected_model)
                
                if tokenizer is None or model is None:
                    st.error("Failed to load model. Please try a different model.")
                    st.stop()
                
                # Extract attention weights
                attentions, tokens, token_ids = extract_attention_weights(
                    model, tokenizer, input_text
                )
                
                # Get basic statistics
                stats = get_attention_stats(attentions)
                
                st.success("✅ Analysis complete!")
                
                # Show basic info
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Model", selected_model.split('-')[0].upper())
                
                with col2:
                    st.metric("Tokens", len(tokens))
                
                with col3:
                    st.metric("Layers", stats['num_layers'])
                
                with col4:
                    st.metric("Heads/Layer", stats['num_heads'])
                
                # Display tokenized input
                st.subheader("🔤 Tokenized Input")
                token_html = ""
                for i, token in enumerate(tokens):
                    clean_token = token.replace('Ġ', '').replace('##', '')  # Clean GPT-2 and BERT tokens
                    if not clean_token.strip():
                        clean_token = '▁'  # Show space tokens
                    token_html += f'<span style="background-color:#f0f2f6; padding:2px 6px; margin:2px; border-radius:3px; font-family:monospace; font-size:12px;" title="Position {i}">{clean_token}</span>'
                
                st.markdown(token_html, unsafe_allow_html=True)
                
                # Layer and head selection for detailed analysis
                if analysis_type in ["Attention Heatmap", "Head Patterns", "Attention Flow"]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        selected_layer = st.selectbox(
                            "Select Layer:",
                            range(stats['num_layers']),
                            index=min(2, stats['num_layers']-1)  # Default to layer 2 or last layer
                        )
                    
                    with col2:
                        if analysis_type == "Attention Heatmap" or analysis_type == "Attention Flow":
                            selected_head = st.selectbox(
                                "Select Head:",
                                range(stats['num_heads']),
                                index=0
                            )
                
                # Generate visualizations based on analysis type
                st.subheader("📊 Visualization Results")
                
                if analysis_type == "Attention Heatmap":
                    attention_matrix, token_labels = process_attention_data(
                        attentions, tokens, selected_layer, selected_head
                    )
                    
                    fig = create_attention_heatmap(
                        attention_matrix, token_labels, selected_layer, selected_head, normalize_attention
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional bar chart
                    bar_fig = create_token_attention_bar(attention_matrix, token_labels, normalize_attention)
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                elif analysis_type == "Token Importance":
                    # Show token importance across all layers
                    st.write("**Token importance based on total attention received across all layers:**")
                    
                    importance_data = []
                    for i, token in enumerate(tokens):
                        total_attention = stats['total_attention'][i]
                        importance_data.append({
                            'Token': token.replace('Ġ', '').replace('##', ''),
                            'Position': i,
                            'Total Attention': total_attention,
                            'Normalized Importance': total_attention / stats['total_attention'].max()
                        })
                    
                    # Sort by importance
                    importance_data.sort(key=lambda x: x['Total Attention'], reverse=True)
                    
                    # Display as metrics
                    cols = st.columns(min(5, len(importance_data)))
                    for i, data in enumerate(importance_data[:5]):
                        with cols[i]:
                            st.metric(
                                label=f"#{i+1}: {data['Token'][:10]}",
                                value=f"{data['Normalized Importance']:.2f}",
                                help=f"Position: {data['Position']}, Raw attention: {data['Total Attention']:.3f}"
                            )
                
                elif analysis_type == "Layer Comparison":
                    entropy_fig = create_layer_comparison(attentions, tokens, 'entropy')
                    st.plotly_chart(entropy_fig, use_container_width=True)
                    
                    max_att_fig = create_layer_comparison(attentions, tokens, 'max_attention')
                    st.plotly_chart(max_att_fig, use_container_width=True)
                
                elif analysis_type == "Head Patterns":
                    head_patterns = analyze_head_patterns(attentions, tokens)
                    
                    head_fig = create_head_pattern_comparison(
                        head_patterns[f'layer_{selected_layer}'], selected_layer
                    )
                    st.plotly_chart(head_fig, use_container_width=True)
                    
                    # Show head specialization details
                    st.write(f"**Head Analysis for Layer {selected_layer}:**")
                    layer_data = head_patterns[f'layer_{selected_layer}']
                    
                    cols = st.columns(min(4, len(layer_data)))
                    for i, (head_name, head_data) in enumerate(list(layer_data.items())[:4]):
                        with cols[i]:
                            st.metric(
                                label=f"Head {i}",
                                value=f"{head_data['entropy']:.2f}",
                                delta=f"Self: {head_data['self_attention']:.2f}",
                                help=f"Entropy: {head_data['entropy']:.3f}, Max attention: {head_data['max_attention']:.3f}"
                            )
                
                elif analysis_type == "Attention Flow":
                    attention_matrix, token_labels = process_attention_data(
                        attentions, tokens, selected_layer, selected_head
                    )
                    
                    flow_fig = create_attention_flow_diagram(attention_matrix, token_labels)
                    st.plotly_chart(flow_fig, use_container_width=True)
                
                # Show statistics if requested
                if show_stats:
                    st.subheader("📈 Attention Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Layer Entropy Values:**")
                        for i, entropy in enumerate(stats['layer_entropies']):
                            st.write(f"Layer {i}: {entropy:.3f}")
                    
                    with col2:
                        st.write("**Model Architecture:**")
                        st.write(f"- Layers: {stats['num_layers']}")
                        st.write(f"- Attention Heads: {stats['num_heads']}")
                        st.write(f"- Sequence Length: {stats['sequence_length']}")
                        st.write(f"- Total Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Try with a shorter text or different model if the error persists.")
                import traceback
                with st.expander("Debug Information"):
                    st.code(traceback.format_exc())
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔍 <strong>LLM Attention Visualizer</strong> | Understanding transformer models through attention analysis</p>
    <p>Select different analysis types and layers to explore how your chosen model processes text</p>
</div>
""", unsafe_allow_html=True)
