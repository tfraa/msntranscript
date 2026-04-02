"""
Utilities for saving intermediate results
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def save_dataframe(df, filepath):
    """
    Save a DataFrame to CSV
    
    Args:
        df: DataFrame to save
        filepath: Path to save to
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents = True, exist_ok = True)
    
    df.to_csv(filepath, index = False)
    logger.info(f"Saved dataframe to {filepath}")

def save_array(dict, filepath):
    """
    Save a dictionary to disk using pickle.

    Args:
        dict: Dictionary to save
        filepath: Path to save to (.pkl format)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents = True, exist_ok = True)

    with open(filepath, 'wb') as f:
        pickle.dump(dict, f)

    logger.info(f"Saved all maps to {filepath}")

def save_results(results, filepath):
    """
    Save results dictionary using pickle
    
    Args:
        results: Dictionary to save
        filepath: Path to save to (.pkl format)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents = True, exist_ok = True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved results to {filepath}")

def save_figure(fig, filepath, dpi = 300):
    """
    Save a matplotlib figure
    
    Args:
        fig: Matplotlib figure object
        filepath: Path to save to
        dpi: Resolution for raster formats
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents = True, exist_ok = True)

    fig.savefig(filepath, format = 'png', dpi = dpi, bbox_inches = 'tight')
    logger.debug(f"Saved figure to {filepath}")