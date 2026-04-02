"""
Imaging Transcriptomics analysis wrapper
"""
import logging
import numpy as np
import pandas as pd
import imaging_transcriptomics as imt
from ..processing.data_processor import N_LH_REGIONS

logger = logging.getLogger(__name__)

class TranscriptomicsAnalyzer:
    """Wrapper for Imaging Transcriptomics Toolbox PLS analysis"""
    
    def __init__(self, vectors, zscored_data):
        """
        Initialize the analyzer
        
        Args:
            vectors: Input map for PLS analysis
            results: results
        """
        self.vectors = vectors
        self.results = None
        self.zscored_data = zscored_data
        self.univariate_maps = None
    
    def run_pls(self):
        """
        Run PLS analysis using Imaging Transcriptomics Toolbox
        
        Args:
            **kwargs: Additional parameters for PLS analysis
            
        Returns:
            results_dict: Dictionary containing all results
            {comparison_name: {
                'gene_results': gene_results object from PLS,
                'gene_df': DataFrame with genes/zscores/pvals,
            }}
        """
        beta_dict = {key: df.set_index('region')['beta'] for key, df in self.vectors.items()}
        strength_maps = pd.DataFrame(beta_dict)

        # Run PLS analysis for each comparison on the singular map
        logger.info(f"Performing PLS analysis on strength maps.")
        results_dict = {}

        for comparison_name in strength_maps.columns:
            logger.info(f"Performing PLS analysis on {comparison_name} maps.")
             # PLS on numpy array from the strength map
            strength_map = strength_maps[comparison_name].values
            pls_result = self.run_pls_imt(strength_map[:N_LH_REGIONS])  # Left hemisphere cortical regions only

            # Extract data for the specified component
            genes = pls_result.results.orig.genes[0, :]
            zscores = pls_result.results.orig.zscored[0, :]
            pvals = pls_result.results.boot.pval[0, :]
            pvals_corr = pls_result.results.boot.pval_corr[0, :]
            
            # Create DataFrame
            data = zip(genes, zscores, pvals, pvals_corr)
            df = pd.DataFrame(data, columns = ["Gene", "Z-score", "p-value", "fdr"])
            results_dict[comparison_name] = {
                'gene_results': pls_result,       # Raw PLS output
                'gene_df': df,                    # Gene DataFrame
                'n_significant': len(df[df['fdr'] < 0.05]['Gene'].tolist())}
            
        # Save results
        self.results = results_dict
        
        return results_dict
    
    def get_results(self):
        """
        Get PLS results
        
        Returns:
            Dictionary containing all PLS results
        """
        if self.results is None:
            raise ValueError("Must run PLS analysis first")
        
        return self.results, self.univariate_maps

    def create_strength_maps_from_results(self, all_results):
        """
        Create average strength brain maps from mass univariate GLM results.
        For each comparison, averages beta values across all metrics for each region.
        
        Args:
            all_results: Dictionary with results from mass univariate GLM analyses
                        Keys: comparison names (e.g., 'HC_vs_Group1')
                        Values: DataFrames with GLM results
        
        Returns:
            composite_maps_df: DataFrame with regions as rows, comparisons as columns
                            Shape: (68 regions, n_comparisons)
        """
        strength_data = {}
        
        for comparison_name, results_df in all_results.items():
            logger.info(f"\nCreating strength map for {comparison_name}...")
            
            # For each region, average beta across all metrics
            mean_betas = results_df.groupby('region')['beta'].mean()
            
            # Store as column
            strength_data[comparison_name] = mean_betas
            
            # Print summary
            logger.info(f"  Regions: {len(mean_betas)}")
            logger.info(f"  Beta range: [{mean_betas.min():.3f}, {mean_betas.max():.3f}]")
        
        # Combine into single DataFrame
        composite_maps_df = pd.DataFrame(strength_data)

        logger.info(f"\nCreated univariate maps DataFrame: {composite_maps_df.shape}")
        logger.info(f"Regions (rows): {len(composite_maps_df)}")
        logger.info(f"Comparisons (columns): {len(composite_maps_df.columns)}")
        
        return composite_maps_df
    
    def run_pls_imt(self, strength_map):
        # Add description
        pls_analysis = imt.ImagingTranscriptomics(strength_map, method = 'pls', n_components = 1, regions = 'cort')
        pls_analysis.run(gsea = False, save_res = False)
        return pls_analysis.gene_results
    
