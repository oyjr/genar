"""
VQVAE for Spatial Transcriptomics - Single Spot Version

This module implements Vector Quantized Variational AutoEncoder adapted for 
single spot gene expression vectors. It encodes gene expression vectors into 
discrete tokens for VAR autoregressive modeling.

Key Design:
1. Input: Gene expression vector [num_genes]
2. Output: Reconstructed gene expression vector [num_genes]
3. Codebook: Learned representations for gene expression patterns
4. Quantization: Gene expression vectors -> discrete tokens

Author: VAR-ST Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for gene expression vectors
    
    Quantizes continuous gene expression patterns into discrete tokens
    that can be processed by the autoregressive VAR transformer.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        """
        Initialize Vector Quantizer
        
        Args:
            num_embeddings: Size of the codebook (number of discrete tokens)
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for the commitment loss
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Learnable codebook - each entry represents a gene expression pattern
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        print(f"🧬 初始化向量量化器:")
        print(f"  - 码本大小: {num_embeddings}")
        print(f"  - 嵌入维度: {embedding_dim}")
        print(f"  - 承诺损失权重: {commitment_cost}")
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of vector quantization
        
        Args:
            inputs: [B, embedding_dim] - continuous gene expression features
        
        Returns:
            Dict containing quantized features, indices, and loss
        """
        # Calculate distances to all codebook entries
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # Find closest codebook entry
        encoding_indices = torch.argmin(distances, dim=1)  # [B]
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()  # [B, num_embeddings]
        
        # Quantize: replace continuous features with discrete codebook vectors
        quantized = torch.matmul(encodings, self.embedding.weight)  # [B, embedding_dim]
        
        # Compute VQ losses
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)  # Commitment loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach())  # Codebook loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return {
            'quantized': quantized,      # [B, embedding_dim]
            'indices': encoding_indices, # [B]
            'loss': loss,
            'perplexity': perplexity,
            'encodings': encodings       # [B, num_embeddings]
        }
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get codebook entries for given indices
        
        Args:
            indices: [B] - token indices
        
        Returns:
            quantized: [B, embedding_dim] - corresponding features
        """
        return self.embedding(indices)


class VQVAE(nn.Module):
    """
    VQVAE for single spot gene expression vectors
    
    Pipeline:
    1. Encoder: gene expression vector -> continuous latent features
    2. VQ: continuous features -> discrete tokens + quantized features  
    3. Decoder: quantized features -> reconstructed gene expression vector
    """
    
    def __init__(
        self,
        input_dim: int = 200,              # Number of genes
        hidden_dim: int = 256,             # Hidden layer dimension
        latent_dim: int = 32,              # Latent space dimension (VQ embedding dim)
        num_embeddings: int = 8192,        # VQ codebook size
        commitment_cost: float = 0.25,     # VQ commitment loss weight
        num_layers: int = 3,               # Number of encoder/decoder layers
    ):
        """
        Initialize VQVAE for gene expression vectors
        
        Args:
            input_dim: Dimension of input gene expression vector
            hidden_dim: Hidden layer dimension
            latent_dim: Dimension of VQ embedding space
            num_embeddings: Size of VQ codebook
            commitment_cost: Weight for VQ commitment loss
            num_layers: Number of layers in encoder/decoder
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        print(f"🧬 初始化 VQVAE (单Spot模式):")
        print(f"  - 输入维度: {input_dim}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - 潜在维度: {latent_dim}")
        print(f"  - 码本大小: {num_embeddings}")
        
        # Encoder: gene expression vector -> latent features
        encoder_layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer outputs latent_dim
                encoder_layers.extend([
                    nn.Linear(current_dim, latent_dim),
                    nn.Tanh()  # Bounded activation for stable quantization
                ])
            else:
                encoder_layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ])
                current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        
        # Decoder: quantized features -> gene expression vector
        decoder_layers = []
        current_dim = latent_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer outputs input_dim
                decoder_layers.append(nn.Linear(current_dim, input_dim))
            else:
                decoder_layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                ])
                current_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        print(f"✅ VQVAE初始化完成")
        print(f"  - 编码器: {input_dim} → {latent_dim}")
        print(f"  - 解码器: {latent_dim} → {input_dim}")
    
    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode gene expression to latent features
        
        Args:
            x: [B, input_dim] - gene expression vectors
        
        Returns:
            latent: [B, latent_dim] - latent features
        """
        return self.encoder(x)
    
    def encode_to_tokens(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode gene expression to discrete tokens
        
        Args:
            x: [B, input_dim] - gene expression vectors
        
        Returns:
            Dict containing tokens and VQ loss
        """
        # Encode to latent space
        latent = self.encode_to_latent(x)
        
        # Quantize
        vq_result = self.vq(latent)
        
        return {
            'tokens': vq_result['indices'],        # [B] - discrete tokens
            'loss': vq_result['loss'],             # VQ loss
            'perplexity': vq_result['perplexity'], # Codebook utilization
            'latent': latent,                      # [B, latent_dim] - continuous latent
            'quantized': vq_result['quantized']    # [B, latent_dim] - quantized latent
        }
    
    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from latent features to gene expression
        
        Args:
            z: [B, latent_dim] - latent features
        
        Returns:
            reconstructed: [B, input_dim] - reconstructed gene expression
        """
        return self.decoder(z)
    
    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete tokens to gene expression
        
        Args:
            tokens: [B] - discrete token indices
        
        Returns:
            reconstructed: [B, input_dim] - reconstructed gene expression
        """
        # Get quantized features from tokens
        quantized = self.vq.get_codebook_entry(tokens)
        
        # Decode to gene expression
        return self.decode_from_latent(quantized)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> quantize -> decode
        
        Args:
            x: [B, input_dim] - input gene expression vectors
        
        Returns:
            Dict containing reconstruction and losses
        """
        # Encode and quantize
        encode_result = self.encode_to_tokens(x)
        
        # Decode
        reconstructed = self.decode_from_latent(encode_result['quantized'])
        
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, x)
        
        return {
            'reconstruction': reconstructed,       # [B, input_dim]
            'tokens': encode_result['tokens'],     # [B]
            'vq_loss': encode_result['loss'],      # VQ loss
            'recon_loss': recon_loss,              # Reconstruction loss
            'loss': encode_result['loss'] + recon_loss,  # Total loss
            'perplexity': encode_result['perplexity']
        }