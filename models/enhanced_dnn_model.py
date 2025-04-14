import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class EnhancedDNN(nn.Module):
    """
    Enhanced DNN that builds upon the original DNN structure.
    Adds support for item counts and co-counts with multiple enhancement options.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5,
                 use_item_counts=False, use_co_counts=True, 
                 co_count_enhancement="feature_attn",  # Options: "simple", "attention", "feature_attn", "cross_attn", "gating"
                 num_heads=1):
        super(EnhancedDNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        
        # Ablation control flags
        self.use_item_counts = use_item_counts
        self.use_co_counts = use_co_counts
        
        # Enhancement type for co-counts
        self.co_count_enhancement = co_count_enhancement
        
        # Original timestep embedding layer (preserved from original DNN)
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)
        
        # If we're using counts, we need embedding networks for them
        if self.use_item_counts:
            self.item_count_proj = nn.Linear(self.in_dims[0], self.time_emb_dim)
            
        if self.use_co_counts:
            # Base projection for co-counts
            self.co_count_proj = nn.Linear(self.in_dims[0], self.time_emb_dim)
            
            # Additional components for different enhancement types
            if co_count_enhancement == "attention":
                # Self-attention mechanism for co-counts
                self.co_count_attn = nn.MultiheadAttention(self.time_emb_dim, num_heads=num_heads, batch_first=True)
                self.co_count_norm = nn.LayerNorm(self.time_emb_dim)
                
            elif co_count_enhancement == "feature_attn":
                # Feature-wise attention with softmax normalization
                self.co_count_attention = nn.Sequential(
                    nn.Linear(self.in_dims[0], self.in_dims[0] // 2),
                    nn.Tanh(),
                    nn.Linear(self.in_dims[0] // 2, 1)
                )
                
            elif co_count_enhancement == "cross_attn":
                # Cross-attention between interactions and co-counts
                self.interaction_query = nn.Linear(self.in_dims[0], self.time_emb_dim)
                self.co_count_key = nn.Linear(self.in_dims[0], self.time_emb_dim)
                self.co_count_value = nn.Linear(self.in_dims[0], self.time_emb_dim)
                
            elif co_count_enhancement == "gating":
                # Gating mechanism to control co-count influence
                self.co_count_gate = nn.Linear(self.time_emb_dim * 2, self.time_emb_dim)
            
        # Calculate input dimension based on what features are used
        if self.time_type == "cat":
            # Base dimension: original input + time embedding
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim]
            
            # Add dimensions for enabled count features
            if self.use_item_counts:
                in_dims_temp[0] += self.time_emb_dim
                
            if self.use_co_counts:
                in_dims_temp[0] += self.time_emb_dim
                
            # Add remaining dimensions
            in_dims_temp = in_dims_temp + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
            
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        """Initialize the network weights using the exact method from original DNN."""
        # Initialize in_layers using the original initialization
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            
            # Normal Initialization for biases
            layer.bias.data.normal_(0.0, 0.001)
            
        # Initialize out_layers using the original initialization
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            
            # Normal Initialization for biases
            layer.bias.data.normal_(0.0, 0.001)
            
        # Initialize emb_layer using the original initialization
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        
        # Initialize count projection layers if they exist
        if self.use_item_counts:
            self._init_layer(self.item_count_proj)
            
        if self.use_co_counts:
            self._init_layer(self.co_count_proj)
            
            # Initialize additional components based on enhancement type
            if self.co_count_enhancement == "attention":
                # Complex modules like MultiheadAttention have their own initialization
                self._init_layer(self.co_count_norm)
                
            elif self.co_count_enhancement == "feature_attn":
                for layer in self.co_count_attention:
                    if isinstance(layer, nn.Linear):
                        self._init_layer(layer)
                        
            elif self.co_count_enhancement == "cross_attn":
                self._init_layer(self.interaction_query)
                self._init_layer(self.co_count_key)
                self._init_layer(self.co_count_value)
                
            elif self.co_count_enhancement == "gating":
                self._init_layer(self.co_count_gate)
    
    def _init_layer(self, layer):
        """Helper method to initialize a layer with the original initialization scheme."""
        if not isinstance(layer, nn.Linear):
            return
            
        size = layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        layer.weight.data.normal_(0.0, std)
        layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, item_counts=None, co_counts=None):
        """
        Forward pass with optional count features
        
        Args:
            x: Binary user-item interaction matrix (batch_size, n_items)
            timesteps: Diffusion timesteps (batch_size,)
            item_counts: Item interaction counts (batch_size, n_items)
            co_counts: Co-interaction counts (batch_size, n_items)
        """
        # Create timestep embeddings (exactly as in original DNN)
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        
        # Normalize input if needed (exactly as in original DNN)
        if self.norm:
            x = F.normalize(x)
            
        # Apply dropout to input (exactly as in original DNN)
        x = self.drop(x)
        
        # Start with original concatenation (interactions + time embedding)
        embeddings = [x, emb]
        
        # Process and add item counts if enabled
        if self.use_item_counts and item_counts is not None:
            item_emb = self.item_count_proj(item_counts)
            embeddings.append(item_emb)
            
        # Process and add co-counts if enabled
        if self.use_co_counts and co_counts is not None:
            # Process co-counts based on the selected enhancement type
            if self.co_count_enhancement == "simple":
                # Simple linear projection (original method)
                co_emb = self.co_count_proj(co_counts)
                
            elif self.co_count_enhancement == "attention":
                # Self-attention mechanism
                co_emb = self.co_count_proj(co_counts)
                
                # Reshape for attention - add a sequence dimension
                batch_size = co_emb.shape[0]
                co_emb_reshaped = co_emb.view(batch_size, 1, -1)
                
                # Apply self-attention
                co_emb_attn, _ = self.co_count_attn(co_emb_reshaped, co_emb_reshaped, co_emb_reshaped)
                co_emb_attn = co_emb_attn.view(batch_size, -1)  # Remove sequence dimension
                
                # Add residual connection and normalization
                co_emb = self.co_count_norm(co_emb + co_emb_attn)
                
            elif self.co_count_enhancement == "feature_attn":
                # Feature-wise attention with softmax normalization
                #attn_weights = self.co_count_attention(co_counts)
                temperature = 1.0
                attn_weights = co_counts/temperature
                attn_weights = F.softmax(attn_weights, dim=1)  # Softmax normalization
                
                # Apply attention to co_counts
                #weighted_co_counts = co_counts * attn_weights
                weighted_co_counts = attn_weights
                
                # Project to embedding space
                co_emb = self.co_count_proj(weighted_co_counts)
                
            elif self.co_count_enhancement == "cross_attn":
                # Cross-attention between interactions and co-counts
                batch_size = x.shape[0]
                
                # Create query from user interactions
                query = self.interaction_query(x).view(batch_size, 1, -1)  # [batch_size, 1, emb_dim]
                
                # Create keys and values from co-counts
                keys = self.co_count_key(co_counts).view(batch_size, 1, -1)  # [batch_size, 1, emb_dim]
                values = self.co_count_value(co_counts).view(batch_size, 1, -1)  # [batch_size, 1, emb_dim]
                
                # Compute scaled dot-product attention
                attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(self.time_emb_dim)
                attn_weights = F.softmax(attn_scores, dim=-1)
                co_emb = torch.matmul(attn_weights, values).view(batch_size, -1)
                
            elif self.co_count_enhancement == "gating":
                # Gating mechanism
                co_emb_base = self.co_count_proj(co_counts)
                
                # Create gate based on both interaction embedding and co-count embedding
                gate_input = torch.cat([emb, co_emb_base], dim=-1)
                gate = torch.sigmoid(self.co_count_gate(gate_input))
                
                # Apply gate to co-count embedding
                co_emb = gate * co_emb_base
            
            else:
                # Default to simple projection if enhancement type is unknown
                co_emb = self.co_count_proj(co_counts)
            
            embeddings.append(co_emb)
            
        # Concatenate all embeddings
        h = torch.cat(embeddings, dim=-1)
        
        # Forward pass through the network (using tanh as in original DNN)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
            
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
                
        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                     These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


