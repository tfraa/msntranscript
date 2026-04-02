"""
Data processing: merging, z-scoring, and vector computation
"""
import logging
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

# Atlas constants — Desikan-Killiany parcellation (aparc)
N_REGIONS = 68      # total cortical parcels (34 left + 34 right)
N_METRICS = 5       # SurfArea, GrayVol, ThickAvg, MeanCurv, GausCurv
N_LH_REGIONS = 34   # left-hemisphere regions used for PLS

class DataProcessor:
    """Processes patient data: merging, z-scoring, and t-maps computation"""
    def __init__(self, data):
        """
        Initialize the processor
        
        Args:
            data: Input dataframe with patient data
        """
        self.data = data.copy()
        self.zscored_data = None
        self.vectors = None
        self.univariate_results = None
        self.region_names = None
        self.strength_results = None
        self.similarity_values = None
        self.strength_values = None
    
    def compute_zscores(self):
        """
        Compute z-scores within-patient for each feature.
        
        Returns:
            DataFrame with z-scored values
        """        
        # Create a copy for z-scoring
        zscored_df = self.data.copy()

        # First column is patient_id, features start from column index 1
        n_regions = N_REGIONS
        n_metrics = N_METRICS

        logger.info(f"{n_metrics} metrics across {n_regions} regions.")

        # Normalize within each patient, for each metric separately
        for idx in zscored_df.index:
            # Process each metric
            for metric_idx in range(n_metrics):
                # Get column indices for this metric (starting from column 1, every 5th column)
                col_indices = range(1 + metric_idx, n_regions * n_metrics + 1, n_metrics)
                metric_cols = self.data.columns[col_indices]
                
                # Get all values for this metric for this patient
                patient_metric_values = self.data.loc[idx, metric_cols]

                # Compute robust z-score for this metric
                med = np.median(patient_metric_values)
                ma = stats.median_abs_deviation(patient_metric_values)
                
                # Avoid division by zero
                if ma == 0:
                    pid = self.data.loc[idx, 'patient_id'] if 'patient_id' in self.data.columns else idx
                    logger.warning(f"MAD is 0 for patient {pid}, metric index {metric_idx}")
                    zscored_df.loc[idx, metric_cols] = 0
                else:
                    zscored_df.loc[idx, metric_cols] = (0.6745 * (patient_metric_values - med)) / ma

        logger.info(f"Computed within-patient z-scores for {n_metrics} metrics across {len(zscored_df)} patients")

        self.zscored_data = zscored_df
        
        return zscored_df
    
    def compute_vectors(self):
        """
        Compute feature maps using mass univariate analysis (GLM).
        Each patient group is compared individually against HC (group 0).

        Returns:
            Dictionary with mass univariate results for each group comparison.
        """
        if self.zscored_data is None:
            raise ValueError("Must compute z-scores first.")

        ### Perform Mass Univariate Analysis on RAW DATA
        all_groups = sorted(self.data['group'].unique())
        hc_group = 0
        patient_groups = [g for g in all_groups if g != hc_group]

        logger.info(f"Groups in data: HC (0) and patient groups {patient_groups}")

        # Store results for each comparison
        all_results_univariate = {}

        mean_centered = self.data.copy()

        # Run analysis for each individual patient group vs HC
        for group_value in patient_groups:
            logger.info(f"Running analysis: HC (0) vs Group {group_value}")
            results_df = self._run_single_glm_comparison(group_value, mean_centered)
            all_results_univariate[f'Group{group_value}_vs_HC'] = results_df

        # Save all results
        self.univariate_results = all_results_univariate

        # Log summary across all comparisons
        self._print_comparison_summary(all_results_univariate)

        # MSN Matrices and Strength vectors
        sim = self.calc_distance_matrix(self.zscored_data)
        strength_df = self.calc_network_metric(sim)
        self.similarity_values = sim
        self.strength_values = strength_df

        # Calculation of betas for effects on groups vs HC
        all_results = {}

        for group_value in patient_groups:
            logger.info(f"Running MSN strength analysis: HC (0) vs Group {group_value}")
            results_df = self._run_single_glm_comparison(group_value, strength_df)
            all_results[f'Group{group_value}_vs_HC'] = results_df

        self.strength_results = all_results
        self._print_comparison_summary(all_results)

        return self.strength_results
    
    def _run_single_glm_comparison(self, group_value, df):
        """
        Run GLM comparison for HC vs a single patient group.
        
        Args:
            group_value: The patient group to compare against HC (0)
            df: Input dataframe
        
        Returns:
            DataFrame with GLM results
        """
        # Filter for HC and this specific patient group
        df_clean = df[df['group'].isin([0, group_value])].copy()
        df_clean['group'] = (df_clean['group'] == group_value).astype(int)

        logger.info(f"Data shape after filtering: {df_clean.shape}")
        logger.info(f"Group distribution: HC = {sum(df_clean['group'] == 0)}, "
                    f"Group{group_value} = {sum(df_clean['group'] == 1)}")
        
        # Get feature columns (exclude demographics and scanner covariates)
        non_feature = {'patient_id', 'participant_id', 'group', 'age', 'sex', 'tiv'}
        feature_cols = [
            col for col in df_clean.columns
            if col not in non_feature and not col.startswith('scanner_')
        ]

        logger.info(f"Running GLM for {len(feature_cols)} features...")

        # Prepare predictors — include scanner_ columns as covariates if present
        scanner_cols = [c for c in df_clean.columns if c.startswith('scanner_')]
        if scanner_cols:
            logger.info(f"Including scanner covariates: {scanner_cols}")
        X = df_clean[['group', 'age', 'sex', 'tiv'] + scanner_cols].copy()
        X = sm.add_constant(X)

        results_list = []

        for feature in feature_cols:
            y = df_clean[feature]
            model = sm.OLS(y, X).fit()

            # Extract region and metric names
            if len(feature.split('_')) == 2:
                region = feature
                metric = ''
            else:
                parts = feature.split('_')
                metric = parts[-1]
                region = '_'.join(parts[:-1])

            results_list.append({
                'feature': feature,
                'region': region,
                'metric': metric,
                'beta': model.params['group'],
                'se': model.bse['group'],
                't_value': model.tvalues['group'],
                'p_value': model.pvalues['group']
            })

        results_df = pd.DataFrame(results_list)

        # FDR correction
        reject, pvals_corrected, _, _ = multipletests(
            results_df['p_value'],
            alpha = 0.05,
            method = 'fdr_bh'
        )

        results_df['p_fdr'] = pvals_corrected
        results_df['significant'] = reject

        n_significant = reject.sum()
        logger.info(f"Significant features (FDR < 0.05): {n_significant} ({n_significant/len(results_df)*100:.1f}%)")

        return results_df

    def _print_comparison_summary(self, all_results):
        """Print summary statistics across all comparisons."""
        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY ACROSS ALL COMPARISONS")
        logger.info(f"{'='*70}")
        
        for comparison_name, results_df in all_results.items():
            n_sig = results_df['significant'].sum()
            n_total = len(results_df)
            logger.info(f"\n{comparison_name}:")
            logger.info(f"  Significant features: {n_sig}/{n_total} ({n_sig/n_total*100:.1f}%)")
            
            # By metric
            for metric in results_df['metric'].unique():
                metric_sig = results_df[results_df['metric'] == metric]['significant'].sum()
                logger.info(f"    {metric}: {metric_sig}/{N_REGIONS} regions")

    def calc_distance_matrix(self, df):
        """
        Calculate similarity matrix from multivariate Euclidean distances between brain regions for each subject.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame where rows are subjects and columns are 'regionX_metricY'
            Shape: (n_subjects, n_regions * n_metrics)
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with pairwise distances between all region pairs.
            Shape: (n_subjects, n_regions * (n_regions - 1) / 2)
        """
        
        # Reshape data: (n_subjects, n_regions, n_metrics)
        n_subjects = len(df)
        n_regions = N_REGIONS
        n_metrics = N_METRICS
        
        val = df.iloc[:, 1:n_regions * n_metrics + 1]
        region_data = val.values.reshape(n_subjects, n_regions, n_metrics)
        
        if self.region_names is None:
            unique_regions = pd.Series(val.columns).str.rsplit('_', n = 1).str[0].unique().tolist()
            self.region_names = unique_regions

        # Calculate pairwise distances for each subject using multivariate euclidean distance 
        all_distances = np.array([pdist(subject_data, metric = 'euclidean') for subject_data in region_data])

        # Convert all subjects to square matrices
        dist_matrices = np.array([squareform(distances) for distances in all_distances])
        
        # Transform distances to similarities to obtain similarity matrices
        similarity_matrices = 1 / (1 + dist_matrices/n_metrics)
        
        similarity_condensed = np.array([squareform(sim_matrix, checks = False) for sim_matrix in similarity_matrices])
        similarity_df = pd.DataFrame(similarity_condensed, index = df.index)
        sim_df = pd.concat([df.iloc[:, :1], similarity_df, df.iloc[:, (n_regions * n_metrics + 1):]], axis=1)

        return sim_df
    
    def calc_network_metric(self, similarity_df):
        
        n_regions = len(self.region_names)
        n_pairs = (n_regions * (n_regions - 1)) // 2 
        similarity_cols = similarity_df.iloc[:, 1:n_pairs+1]

        node_strengths = np.array([squareform(similarity_cols.iloc[i].values).sum(axis = 1) for i in range(len(similarity_cols))])

        strength_df = pd.DataFrame(
            node_strengths,
            index = similarity_df.index,
            columns = self.region_names
        )

        str_df = pd.concat([similarity_df.iloc[:, :1], strength_df, similarity_df.iloc[:, (n_pairs + 1):]], axis = 1)

        return str_df

    def get_processed_data(self):
        """
        Get all processed data
        
        Returns:
            Dictionary containing all intermediate results
        """
        return {
            'raw_data': self.data,
            'zscored_data': self.zscored_data,
            'vectors': self.univariate_results,
            'strength_vec': self.strength_results,
            'strength_map': self.strength_values,
            'similarity_mat': self.similarity_values,
            'region_labels': self.region_names
        }