"""
Visualization functions for attention analysis
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st

def create_attention_heatmap(attention_matrix, token_labels, layer_idx, head_idx, normalize=True):
    """
    Create an interactive attention heatmap using Plotly
    """
    if normalize:
        # Normalize each row to sum to 1
        attention_matrix = attention_matrix / (attention_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=token_labels,
        y=token_labels,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='<b>From:</b> %{y}<br><b>To:</b> %{x}<br><b>Attention:</b> %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}',
        xaxis_title='Attended To (Keys)',
        yaxis_title='Attending From (Queries)',
        width=700,
        height=600,
        font=dict(size=12)
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_token_attention_bar(attention_matrix, token_labels, normalize=True):
    """
    Create bar chart showing total attention received by each token
    """
    if normalize:
        attention_matrix = attention_matrix / (attention_matrix.sum(axis=1, keepdims=True) + 1e-8)
    
    # Calculate total attention received by each token (sum of columns)
    total_attention = attention_matrix.sum(axis=0)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Token': token_labels,
        'Total Attention': total_attention,
        'Position': range(len(token_labels))
    })
    
    fig = px.bar(
        df, 
        x='Token', 
        y='Total Attention',
        title='Total Attention Received by Each Token',
        color='Total Attention',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title='Tokens',
        yaxis_title='Total Attention Weight',
        showlegend=False,
        width=800,
        height=400
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_layer_comparison(attentions, tokens, metric='entropy'):
    """
    Compare attention patterns across layers
    """
    layer_data = []
    
    for layer_idx, layer_attention in enumerate(attentions):
        if metric == 'entropy':
            # Calculate average entropy across heads
            layer_entropies = []
            for head_idx in range(layer_attention.shape[1]):
                head_attention = layer_attention[0, head_idx, :, :].numpy()
                
                # Calculate entropy for each position
                entropies = []
                for i in range(head_attention.shape[0]):
                    dist = head_attention[i]
                    dist = dist / (dist.sum() + 1e-8)
                    entropy = -np.sum(dist * np.log(dist + 1e-8))
                    entropies.append(entropy)
                
                layer_entropies.append(np.mean(entropies))
            
            layer_data.append({
                'Layer': layer_idx,
                'Average Entropy': np.mean(layer_entropies),
                'Std Entropy': np.std(layer_entropies)
            })
        
        elif metric == 'max_attention':
            # Calculate average max attention across heads
            max_attentions = []
            for head_idx in range(layer_attention.shape[1]):
                head_attention = layer_attention[0, head_idx, :, :].numpy()
                max_attentions.append(head_attention.max())
            
            layer_data.append({
                'Layer': layer_idx,
                'Average Max Attention': np.mean(max_attentions),
                'Std Max Attention': np.std(max_attentions)
            })
    
    df = pd.DataFrame(layer_data)
    
    if metric == 'entropy':
        fig = px.line(
            df, 
            x='Layer', 
            y='Average Entropy',
            title='Attention Entropy Across Layers',
            error_y='Std Entropy'
        )
        fig.update_layout(
            xaxis_title='Layer Index',
            yaxis_title='Average Attention Entropy',
            width=700,
            height=400
        )
    else:
        fig = px.line(
            df, 
            x='Layer', 
            y='Average Max Attention',
            title='Max Attention Values Across Layers',
            error_y='Std Max Attention'
        )
        fig.update_layout(
            xaxis_title='Layer Index',
            yaxis_title='Average Max Attention',
            width=700,
            height=400
        )
    
    return fig

def create_head_pattern_comparison(layer_patterns, layer_idx):
    """
    Compare patterns across attention heads in a single layer
    """
    heads_data = []
    
    for head_name, head_data in layer_patterns.items():
        head_idx = int(head_name.split('_')[1])
        heads_data.append({
            'Head': head_idx,
            'Self Attention': head_data['self_attention'],
            'Entropy': head_data['entropy'],
            'Max Attention': head_data['max_attention']
        })
    
    df = pd.DataFrame(heads_data)
    
    # Create subplot with multiple metrics
    fig = go.Figure()
    
    # Self attention
    fig.add_trace(go.Scatter(
        x=df['Head'],
        y=df['Self Attention'],
        mode='lines+markers',
        name='Self Attention',
        line=dict(color='blue')
    ))
    
    # Entropy (normalized to 0-1 for comparison)
    entropy_normalized = df['Entropy'] / df['Entropy'].max()
    fig.add_trace(go.Scatter(
        x=df['Head'],
        y=entropy_normalized,
        mode='lines+markers',
        name='Entropy (normalized)',
        line=dict(color='red')
    ))
    
    # Max attention
    fig.add_trace(go.Scatter(
        x=df['Head'],
        y=df['Max Attention'],
        mode='lines+markers',
        name='Max Attention',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title=f'Attention Head Patterns - Layer {layer_idx}',
        xaxis_title='Head Index',
        yaxis_title='Normalized Values',
        width=700,
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

def create_attention_flow_diagram(attention_matrix, token_labels, threshold=0.1):
    """
    Create a flow diagram showing major attention connections
    """
    # Find strong attention connections above threshold
    connections = []
    
    for i, from_token in enumerate(token_labels):
        for j, to_token in enumerate(token_labels):
            attention_weight = attention_matrix[i, j]
            if attention_weight > threshold and i != j:  # Ignore self-attention
                connections.append({
                    'from': from_token,
                    'to': to_token,
                    'weight': attention_weight,
                    'from_pos': i,
                    'to_pos': j
                })
    
    if not connections:
        # If no connections above threshold, lower it
        threshold = attention_matrix.max() * 0.5
        for i, from_token in enumerate(token_labels):
            for j, to_token in enumerate(token_labels):
                attention_weight = attention_matrix[i, j]
                if attention_weight > threshold and i != j:
                    connections.append({
                        'from': from_token,
                        'to': to_token,
                        'weight': attention_weight,
                        'from_pos': i,
                        'to_pos': j
                    })
    
    # Create network-style visualization
    node_x = []
    node_y = []
    node_text = []
    
    # Position nodes in a circle
    n_tokens = len(token_labels)
    for i, token in enumerate(token_labels):
        angle = 2 * np.pi * i / n_tokens
        x = np.cos(angle)
        y = np.sin(angle)
        node_x.append(x)
        node_y.append(y)
        node_text.append(token)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for conn in connections:
        from_idx = conn['from_pos']
        to_idx = conn['to_pos']
        
        edge_x.extend([node_x[from_idx], node_x[to_idx], None])
        edge_y.extend([node_y[from_idx], node_y[to_idx], None])
        edge_weights.append(conn['weight'])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='lightblue'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        hoverinfo='text',
        marker=dict(
            size=30,
            color='lightcoral',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'Attention Flow (threshold > {threshold:.2f})',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600,
        height=600,
        plot_bgcolor='white'
    )
    
    return fig
