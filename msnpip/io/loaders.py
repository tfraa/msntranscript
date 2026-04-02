"""
Data loaders for FreeSurfer files
"""
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FreeSurferLoader:
    """Loads and parses FreeSurfer .annot files"""
    
    def __init__(self):
        """Initialize the loader"""
        self.failed_files = []
    
    def load_patient(self, filepath: str):
        """
        Load data from a single patient's .annot file
        
        Args:
            filepath: Path to the .annot file
            
        Returns:
            Dictionary containing patient data, or None if loading failed
        """
        try:
            # Find .stats files in the subject directory
            subject_dir = Path(filepath) 
            stats_files = list(subject_dir.glob(f"**/*aparc.stats"))
            if len(stats_files) != 2:
                raise FileNotFoundError(f"Expected 2 .stats files, found {len(stats_files)} in {subject_dir}")
            
            # Merge stats files into a single DataFrame
            data = self.get_stats_data(stats_files)
            flattened = {}
            cols_to_keep = ['SurfArea', 'GrayVol', 'ThickAvg', 'MeanCurv', 'GausCurv']
            df = data[[col for col in cols_to_keep if col in data.columns]]
            flattened['patient_id'] = subject_dir.name 

            for region_name, row in df.iterrows():
                for metric in df.columns:
                    col_name = f"{region_name}_{metric}"
                    flattened[col_name] = row[metric]

            return flattened
            
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")
            self.failed_files.append(filepath)
            return None
    
    def load_all_patients(self, directory: str):
        """
        Load data from all .annot files in a directory
        
        Args:
            directory: Directory containing .annot files
            
        Returns:
            DataFrame with all patient data
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        subject_dirs = sorted([d for d in directory.iterdir() if d.is_dir()])

        if not subject_dirs:
            raise ValueError(f"No subject folders found in {directory}")
        
        logger.info(f"\nFound {len(subject_dirs)} subject folders \n")
        
        # Load all patients
        all_patients = []
        for subject_dir in subject_dirs:
            patient_data = self.load_patient(str(subject_dir))
            if patient_data is not None:
                all_patients.append(patient_data)
        
        # Report failures
        if self.failed_files:
            logger.warning(f"Failed to load {len(self.failed_files)} files:")
            for failed in self.failed_files:
                logger.warning(f"  - {failed}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_patients)

        logger.info(f"Successfully loaded {len(df)} patients.")
        
        return df
    
    def get_failed_files(self):
        """
        Get list of files that failed to load
        
        Returns:
            List of failed file paths
        """
        return self.failed_files
    
    def get_stats_data(self, stats_files):
        """
        Load stats data from the two FreeSurfer .stats file (left and right hemispheres)
        Args:
            stats_files: Paths to the .stats file
        Returns:
            pd.DataFrame with patient data
        """
        dfs = []
        # Sort so left hemisphere is always first (dfs[0]) regardless of glob order
        stats_files = sorted(stats_files, key=lambda p: (0 if 'lh' in p.name.lower() else 1))
        for hemi_file in stats_files:
            try:
                with open(hemi_file, 'r') as f:
                    lines = f.readlines()

                # Find header and data
                header = None
                data = []

                for line in lines:
                    line = line.strip()
                    if line.startswith('# ColHeaders'):
                        header = line.replace('# ColHeaders', '').strip().split()
                    elif not line.startswith('#') and line:
                        data.append(line.split())

                # Create DataFrame
                df = pd.DataFrame(data, columns = header)

                # Convert numeric columns
                for col in df.columns:
                    if col != 'StructName':
                        df[col] = pd.to_numeric(df[col])

                # Indexes as brain areas names
                df.set_index('StructName', inplace = True)

                dfs.append(df)
            except Exception as e:
                logger.warning(f"Skipping malformed stats file {hemi_file.name}: {e}")

        if len(dfs) < 2:
            raise ValueError(
                f"Expected 2 hemisphere stats files but only parsed {len(dfs)} successfully. "
                f"Check that both lh.aparc.stats and rh.aparc.stats are present and well-formed."
            )

        # Combine the two hemispheres
        lh_renamed = dfs[0].copy()
        rh_renamed = dfs[1].copy()
        lh_renamed.index = 'lh_' + lh_renamed.index
        rh_renamed.index = 'rh_' + rh_renamed.index
        merged_data = pd.concat([lh_renamed, rh_renamed])

        return merged_data