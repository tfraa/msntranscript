"""
Data validation utilities
"""
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def validate_dataframe(df, required_columns):
    """
    Validate that a dataframe has required columns
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True

def validate_patient_data(df):
    """
    Validate patient data quality
    
    Args:
        df: Patient dataframe
        
    Returns:
        True if valid, logs warnings for issues
    """
    # Check for missing patient IDs
    if df['patient_id'].isna().any():
        raise ValueError("Found missing patient IDs")
    
    # Check for duplicates
    duplicates = df['patient_id'].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate patient IDs")
    
    # Check for missing values
    missing_pct = df.isna().sum() / len(df) * 100
    high_missing = missing_pct[missing_pct > 50]
    
    if not high_missing.empty:
        logger.warning(f"Columns with > 50% missing data: {high_missing.to_dict()}")
    
    return True