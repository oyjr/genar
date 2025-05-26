#!/usr/bin/env python3
"""
Gene Expression Visualization Module for MFBP.
Based on STEm's visualization approach but adapted for MFBP architecture.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import torch
import anndata
from pathlib import Path

class GeneVisualizer:
    """
    Gene expression visualization class for spatial transcriptomics.
    
    This class provides comprehensive visualization capabilities including:
    - Gene variation curves (mean and variance analysis)
    - Spatial gene expression maps
    - Correlation analysis plots
    - Performance comparison visualizations
    """
    
    def __init__(self, save_dir: str = "./vis", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the gene visualizer.
        
        Args:
            save_dir: Directory to save visualization outputs
            figsize: Default figure size for plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize
        
        # Set up matplotlib style for better-looking plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_gene_variation_curves(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 save_name: str = "gene_variation_curves",
                                 show_plots: bool = False) -> None:
        """
        Plot gene variation curves following STEm's exact implementation.
        
        Args:
            y_true: Ground truth gene expression [num_spots, num_genes]
            y_pred: Predicted gene expression [num_spots, num_genes]
            save_name: Name for saved plot file
            show_plots: Whether to display plots interactively
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Normalized Mean (STEm approach)
        pred_mean = np.mean(y_pred, axis=0)
        pred_mean_norm = pred_mean / np.sum(pred_mean)
        gt_mean = np.mean(y_true, axis=0)
        gt_mean_norm = gt_mean / np.sum(gt_mean)
        gt_mean_sorted = np.sort(gt_mean_norm)
        pred_mean_sorted = pred_mean_norm[np.argsort(gt_mean_norm)]
        axs[0, 0].plot(np.arange(len(gt_mean_sorted)), gt_mean_sorted, label="Ground Truth", c="b")
        axs[0, 0].scatter(np.arange(len(pred_mean_sorted)), pred_mean_sorted, s=5, label="Predicted", c="orange")
        axs[0, 0].set_title("Normalized Mean")
        axs[0, 0].set_xlabel("gene index ordered by mean")
        axs[0, 0].set_ylabel("normalized mean")
        axs[0, 0].legend()
        
        # 2. Absolute Mean (STEm approach)
        gt_mean_sorted = np.sort(gt_mean)
        pred_mean_sorted = pred_mean[np.argsort(gt_mean)]
        axs[1, 0].plot(np.arange(len(gt_mean_sorted)), gt_mean_sorted, label="Ground Truth", c="b")
        axs[1, 0].scatter(np.arange(len(pred_mean_sorted)), pred_mean_sorted, s=5, label="Predicted", c="orange")
        axs[1, 0].set_title("Absolute Mean")
        axs[1, 0].set_xlabel("gene index ordered by mean")
        axs[1, 0].set_ylabel("absolute mean")
        axs[1, 0].legend()
        
        # 3. Normalized Variance (STEm approach)
        pred_var = np.var(y_pred, axis=0)
        pred_var_norm = pred_var / np.sum(pred_var)
        gt_var = np.var(y_true, axis=0)
        gt_var_norm = gt_var / np.sum(gt_var)
        gt_var_sorted = np.sort(gt_var_norm)
        pred_var_sorted = pred_var_norm[np.argsort(gt_var_norm)]
        axs[0, 1].plot(np.arange(len(gt_var_sorted)), gt_var_sorted, label="Ground Truth", c="b")
        axs[0, 1].scatter(np.arange(len(pred_var_sorted)), pred_var_sorted, s=5, label="Predicted", c="orange")
        axs[0, 1].set_title("Normalized Variance")
        axs[0, 1].set_xlabel("gene index ordered by var")
        axs[0, 1].set_ylabel("normalized variance")
        axs[0, 1].legend()
        
        # 4. Absolute Variance (STEm approach)
        gt_var_sorted = np.sort(gt_var)
        pred_var_sorted = pred_var[np.argsort(gt_var)]
        axs[1, 1].plot(np.arange(len(gt_var_sorted)), gt_var_sorted, label="Ground Truth", c="b")
        axs[1, 1].scatter(np.arange(len(pred_var_sorted)), pred_var_sorted, s=5, label="Predicted", c="orange")
        axs[1, 1].set_title("Absolute Variance")
        axs[1, 1].set_xlabel("gene index ordered by var")
        axs[1, 1].set_ylabel("absolute variance")
        axs[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot (STEm approach)
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300)
        print(f"Gene variation curves saved to: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_correlation_analysis(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                gene_names: Optional[List[str]] = None,
                                save_name: str = "correlation_analysis",
                                show_plots: bool = False) -> np.ndarray:
        """
        Plot comprehensive correlation analysis.
        
        Args:
            y_true: Ground truth gene expression [num_spots, num_genes]
            y_pred: Predicted gene expression [num_spots, num_genes]
            gene_names: Optional list of gene names
            save_name: Name for saved plot file
            show_plots: Whether to display plots interactively
            
        Returns:
            Array of gene-wise correlation coefficients
        """
        # Calculate gene-wise correlations
        num_genes = y_true.shape[1]
        correlations = []
        
        for i in range(num_genes):
            true_gene = y_true[:, i]
            pred_gene = y_pred[:, i]
            
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            correlations.append(corr)
        
        correlations = np.array(correlations)
        
        # Create comprehensive correlation plots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gene Expression Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Correlation histogram
        axs[0, 0].hist(correlations, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axs[0, 0].axvline(np.mean(correlations), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(correlations):.3f}')
        axs[0, 0].axvline(np.median(correlations), color='orange', linestyle='--', 
                         label=f'Median: {np.median(correlations):.3f}')
        axs[0, 0].set_title('Gene-wise Correlation Distribution')
        axs[0, 0].set_xlabel('Pearson Correlation Coefficient')
        axs[0, 0].set_ylabel('Number of Genes')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # 2. Correlation vs Gene Index
        sorted_indices = np.argsort(correlations)[::-1]
        axs[0, 1].plot(correlations[sorted_indices], marker='o', markersize=3, alpha=0.7)
        axs[0, 1].set_title('Correlations Ranked by Performance')
        axs[0, 1].set_xlabel('Gene Rank')
        axs[0, 1].set_ylabel('Pearson Correlation')
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Top and bottom performing genes
        top_10_indices = sorted_indices[:10]
        bottom_10_indices = sorted_indices[-10:]
        
        x_pos = np.arange(10)
        width = 0.35
        
        axs[1, 0].bar(x_pos - width/2, correlations[top_10_indices], width, 
                     label='Top 10 Genes', color='green', alpha=0.7)
        axs[1, 0].bar(x_pos + width/2, correlations[bottom_10_indices], width, 
                     label='Bottom 10 Genes', color='red', alpha=0.7)
        axs[1, 0].set_title('Top vs Bottom Performing Genes')
        axs[1, 0].set_xlabel('Gene Rank')
        axs[1, 0].set_ylabel('Pearson Correlation')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # 4. Correlation statistics summary
        stats_text = f"""
        Correlation Statistics:
        Mean: {np.mean(correlations):.4f}
        Median: {np.median(correlations):.4f}
        Std: {np.std(correlations):.4f}
        Min: {np.min(correlations):.4f}
        Max: {np.max(correlations):.4f}
        
        PCC-10: {np.mean(correlations[sorted_indices[:10]]):.4f}
        PCC-50: {np.mean(correlations[sorted_indices[:50]]):.4f}
        PCC-200: {np.mean(correlations[sorted_indices[:200]] if len(correlations) >= 200 else correlations):.4f}
        """
        
        axs[1, 1].text(0.1, 0.5, stats_text, transform=axs[1, 1].transAxes, 
                      fontsize=12, verticalalignment='center',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        axs[1, 1].set_title('Summary Statistics')
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation analysis saved to: {save_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        return correlations
    
    def plot_spatial_gene_expression(self, 
                                   adata: anndata.AnnData,
                                   y_pred: np.ndarray,
                                   gene_names: List[str],
                                   marker_genes: List[str],
                                   data_path: str,
                                   slide_id: str,
                                   save_name: str = "spatial_expression",
                                   show_plots: bool = False) -> None:
        """
        Plot spatial gene expression maps for marker genes on WSI tissue images.
        Completely follows STEm's implementation approach.
        
        Args:
            adata: AnnData object containing spatial coordinates and ground truth
            y_pred: Predicted gene expression [num_spots, num_genes]
            gene_names: List of all gene names
            marker_genes: List of marker genes to visualize
            data_path: Path to dataset directory
            slide_id: Slide identifier for finding WSI image
            save_name: Base name for saved plot files
            show_plots: Whether to display plots interactively
        """
        # Handle PIL image size limit for large WSI images
        Image.MAX_IMAGE_PIXELS = None
        
        # Load WSI image following STEm's approach
        img_path = data_path + "wsis/"
        img_raw = None
        
        try:
            # Try different file extensions as in STEm
            for suffix in ['.tif', '.jpg', '.png']:
                try:
                    full_img_path = img_path + slide_id + suffix
                    img_raw = Image.open(full_img_path)
                    print(f"Successfully loaded WSI image: {full_img_path}")
                    break
                except FileNotFoundError:
                    continue
            else:
                print(f"Warning: Could not find WSI image for slide {slide_id}, skipping marker gene visualization")
                return
        except Exception as e:
            print(f"Error loading WSI image: {e}")
            return
        
        # Get spatial coordinates (STEm approach)
        x = adata.obsm["spatial"][:, 0]
        y = adata.obsm["spatial"][:, 1]
        
        # Plot each marker gene following STEm's exact approach
        for gene in marker_genes:
            # Check if gene is in selected gene list (STEm approach)
            if gene not in gene_names:
                print(f"Warning: Gene {gene} not found in gene list, skipping")
                continue
            
            try:
                fig, axs = plt.subplots(1, 2, figsize=(16, 8))
                fig.suptitle(f"Gene: {gene}", fontsize=16)
                
                gene_idx = np.where(np.array(gene_names) == gene)[0][0]
                
                # Ground Truth panel (STEm approach)
                axs[0].imshow(img_raw)
                # Handle different matrix types as in STEm
                if isinstance(adata.X, np.ndarray):
                    color_gt = adata[:, adata.var_names == gene].X.flatten()
                else:
                    color_gt = adata[:, adata.var_names == gene].X.toarray().flatten()
                im0 = axs[0].scatter(x, y, c=color_gt, s=15, alpha=0.9, cmap='viridis')
                axs[0].set_title("Ground Truth", fontsize=14)
                fig.colorbar(im0, ax=axs[0])
                
                # Prediction panel (STEm approach)
                axs[1].imshow(img_raw)
                color_pred = y_pred[:, gene_idx]
                im1 = axs[1].scatter(x, y, c=color_pred, s=15, alpha=0.9, cmap='viridis') 
                axs[1].set_title("Prediction", fontsize=14)
                fig.colorbar(im1, ax=axs[1])
                
                # Save plot (STEm approach)
                plt.tight_layout()
                save_path = self.save_dir / f'marker_gene_{gene}.png'
                plt.savefig(save_path, dpi=300)
                print(f"Spatial expression plot for {gene} saved to: {save_path}")
                
                if show_plots:
                    plt.show()
                else:
                    plt.close()
                    
            except Exception as e:
                print(f"Error plotting gene {gene}: {e}")
                continue
        
        print(f"Completed spatial gene expression visualization for {len(marker_genes)} genes")
    
    def create_summary_report(self, 
                            metrics: Dict[str, float],
                            correlations: np.ndarray,
                            save_name: str = "summary_report") -> None:
        """
        Create a comprehensive summary report with all key metrics and visualizations.
        
        Args:
            metrics: Dictionary containing evaluation metrics (PCC-10, PCC-50, etc.)
            correlations: Array of gene-wise correlations
            save_name: Name for saved report file
        """
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('MFBP Model Performance Summary Report', fontsize=20, fontweight='bold')
        
        # 1. Key Metrics Bar Chart
        metric_names = ['PCC-10', 'PCC-50', 'PCC-200', 'MSE', 'MAE', 'RVD']
        metric_values = [metrics.get(name, 0) for name in metric_names]
        
        colors = ['green', 'blue', 'orange', 'red', 'purple', 'brown']
        bars = axs[0, 0].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axs[0, 0].set_title('Key Performance Metrics', fontsize=14, fontweight='bold')
        axs[0, 0].set_ylabel('Metric Value')
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axs[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Correlation Distribution
        axs[0, 1].hist(correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axs[0, 1].axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {np.mean(correlations):.3f}')
        axs[0, 1].set_title('Gene Correlation Distribution', fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel('Pearson Correlation')
        axs[0, 1].set_ylabel('Number of Genes')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # 3. Top Genes Performance
        sorted_indices = np.argsort(correlations)[::-1]
        top_20_corr = correlations[sorted_indices[:20]]
        axs[0, 2].plot(range(1, 21), top_20_corr, 'o-', color='green', linewidth=2, markersize=6)
        axs[0, 2].set_title('Top 20 Genes Performance', fontsize=14, fontweight='bold')
        axs[0, 2].set_xlabel('Gene Rank')
        axs[0, 2].set_ylabel('Pearson Correlation')
        axs[0, 2].grid(True, alpha=0.3)
        
        # 4. Performance Categories
        excellent = np.sum(correlations > 0.8)
        good = np.sum((correlations > 0.6) & (correlations <= 0.8))
        fair = np.sum((correlations > 0.4) & (correlations <= 0.6))
        poor = np.sum(correlations <= 0.4)
        
        categories = ['Excellent\n(>0.8)', 'Good\n(0.6-0.8)', 'Fair\n(0.4-0.6)', 'Poor\n(≤0.4)']
        counts = [excellent, good, fair, poor]
        colors_cat = ['darkgreen', 'green', 'orange', 'red']
        
        axs[1, 0].pie(counts, labels=categories, colors=colors_cat, autopct='%1.1f%%', startangle=90)
        axs[1, 0].set_title('Gene Performance Categories', fontsize=14, fontweight='bold')
        
        # 5. Model Comparison (placeholder for future use)
        models = ['MFBP']
        pcc_10_values = [metrics.get('PCC-10', 0)]
        
        axs[1, 1].bar(models, pcc_10_values, color='blue', alpha=0.7)
        axs[1, 1].set_title('Model Comparison (PCC-10)', fontsize=14, fontweight='bold')
        axs[1, 1].set_ylabel('PCC-10 Score')
        axs[1, 1].set_ylim(0, 1)
        
        # Add value label
        for i, v in enumerate(pcc_10_values):
            axs[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Summary Statistics Table
        stats_text = f"""
        Model Performance Summary
        ========================
        
        Overall Metrics:
        • PCC-10:  {metrics.get('PCC-10', 0):.4f}
        • PCC-50:  {metrics.get('PCC-50', 0):.4f}
        • PCC-200: {metrics.get('PCC-200', 0):.4f}
        • MSE:     {metrics.get('MSE', 0):.4f}
        • MAE:     {metrics.get('MAE', 0):.4f}
        • RVD:     {metrics.get('RVD', 0):.4f}
        
        Gene-wise Statistics:
        • Total Genes: {len(correlations)}
        • Mean Correlation: {np.mean(correlations):.4f}
        • Std Correlation:  {np.std(correlations):.4f}
        • Best Gene Corr:  {np.max(correlations):.4f}
        • Worst Gene Corr: {np.min(correlations):.4f}
        
        Performance Distribution:
        • Excellent (>0.8): {excellent} genes ({excellent/len(correlations)*100:.1f}%)
        • Good (0.6-0.8):   {good} genes ({good/len(correlations)*100:.1f}%)
        • Fair (0.4-0.6):   {fair} genes ({fair/len(correlations)*100:.1f}%)
        • Poor (≤0.4):      {poor} genes ({poor/len(correlations)*100:.1f}%)
        """
        
        axs[1, 2].text(0.05, 0.95, stats_text, transform=axs[1, 2].transAxes, 
                      fontsize=10, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        axs[1, 2].set_title('Detailed Statistics', fontsize=14, fontweight='bold')
        axs[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save report
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary report saved to: {save_path}")
        
        plt.close() 