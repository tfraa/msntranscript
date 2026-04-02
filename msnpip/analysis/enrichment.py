"""
Enrichment and over-representation analysis using gseapy
"""
import logging
import gseapy as gp
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class EnrichmentAnalyzer:
    """Performs enrichment with gseapy"""
    
    def __init__(self, pls_results):
        """
        Initialize the analyzer
        
        Args:
            pls_results: Results from PLS analysis
        """
        self.gene_sets_dir = Path(__file__).parent.parent / 'genes'
        self.pls_results = pls_results
        self.enrichment_results = {}
    
    def run_enrichment(self, **kwargs):
        """
        Run GSEA enrichment analysis
        
        Args:
            **kwargs: Additional parameters for gseapy.prerank
            
        Returns:
            Dictionary containing enrichment results
        """

        enrichment_results_all = {}

        for comparison_name, results in self.pls_results.items():
            # Get the gene DataFrame
            gene_df = results['gene_df']
            
            # Extract just Gene and Z-score columns, sort by Z-score (high to low)
            genes_rnk = gene_df[['Gene', 'Z-score']].sort_values(by = 'Z-score', ascending = False).reset_index(drop = True)
            enrichment_results_all[comparison_name] = {}

            for genepath in self.gene_sets_dir.iterdir():
                try:
                    if genepath.suffix != ".gmt":
                        continue
                    logger.info(f"Running {genepath.stem} enrichment analysis...")

                    enr = gp.prerank(
                        rnk = genes_rnk,
                        gene_sets = str(genepath),
                        outdir = None,
                        no_plot = True
                    )

                    res = enr.res2d
                    enrichment_results_all[comparison_name][genepath.stem] = res
                    
                    sig = res[res['FDR q-val'] < 0.05]
                    if sig.empty:
                        logger.info(f"No significant {genepath.stem} enrichment found for {comparison_name} data.")
                    else:
                        logger.info(f"Significant genes found in {genepath.stem} enrichment for {comparison_name} data.")
                    
                except Exception as e:
                    logger.error(f"Error in {genepath.stem} enrichment for {comparison_name}: {e}")
        
        self.enrichment_results = enrichment_results_all
        return self.enrichment_results
    
    def get_all_results(self):
        """
        Get all enrichment results.

        Returns:
            Dictionary with structure:
            {comparison_name: {gene_set_name: DataFrame}}
        """
        return self.enrichment_results