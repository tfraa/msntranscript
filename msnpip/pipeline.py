"""
Main pipeline
"""
import logging
from pathlib import Path
import pandas as pd

from .io.loaders import FreeSurferLoader
from .io.savers import save_dataframe, save_array, save_results, save_figure
from .processing.data_processor import DataProcessor, N_REGIONS
from .processing.validators import validate_dataframe, validate_patient_data
from .analysis.transcriptomics import TranscriptomicsAnalyzer
from .analysis.enrichment import EnrichmentAnalyzer
from .visualization.visualizer import Visualizer
from .reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Main orchestrator for the imaging transcriptomics pipeline.
    Supports both full pipeline execution and partial execution from personal data.
    """
    def __init__(
        self,
        save_all: bool = False,
        save_figures: bool = False,
        figures_dir: str = "./figures",
    ):
        """
        Initialize the pipeline
        
        Args:
            save_all: Whether to save intermediate results to disk
            save_figures: Whether to save figures as individual files
            figures_dir: Directory to save individual figures
        """
        self.save_all = save_all
        self.output_dir = Path("./output")
        self.save_figures = save_figures
        self.figures_dir = Path(figures_dir)
        
        if self.save_figures:
            self.figures_dir.mkdir(parents = True, exist_ok = True)
        
        # Initialize components as needed
        self.loader = None
        self.processor = None
        self.transcriptomics = None
        self.enrichment = None
        self.visualizer = Visualizer()
        self.reporter = None
        
        # Store all data for reporting
        self.all_data = {}
        self.all_figures = []
    
    def run_full_pipeline(
        self,
        input_dir: str = None,
        demographic_file: str = None,
        dataframe: pd.DataFrame = None,
        output_pdf: str = "output/",
        groups = None,
    ):
        """
        Run the complete pipeline from .annot files (or a pre-merged DataFrame) to final report.

        Args:
            input_dir: Directory containing FreeSurfer .stats files (required if dataframe is None)
            demographic_file: Path to demographics CSV file (required if dataframe is None)
            dataframe: Pre-merged DataFrame containing morphometric features and demographic
                       columns (age, sex, tiv, group). When provided, skips FreeSurfer loading
                       and demographics merging entirely. The DataFrame is validated before use.
            output_pdf: Output folder for final PDF report
            groups: Specific groups to analyze (if None, analyzes all)
        """
        logger.info("Starting full pipeline")
        
        # Create output dir
        self.output_dir = Path(output_pdf)
        self.output_dir.mkdir(parents = True, exist_ok = True)
        self.figures_dir = Path(f"{output_pdf}/figures")
        self.figures_dir.mkdir(parents = True, exist_ok = True)

        # Phase 1: Load data
        logger.info("=" * 60)
        logger.info("  PHASE 1 / 6 — DATA LOADING")
        logger.info("=" * 60)
        if dataframe is not None:
            logger.info("Using pre-merged DataFrame as input — skipping FreeSurfer loading.")
            merged_data = self._load_from_dataframe(dataframe)
        elif input_dir is not None and demographic_file is not None:
            merged_data = self.run_loading(input_dir, demographic_file)
        else:
            raise ValueError(
                "Provide either 'dataframe' or both 'input_dir' and 'demographic_file'."
            )
        self.all_data['merged_data'] = merged_data
        # Drop rows with ANY missing value
        merged_data = merged_data.dropna()
        logger.info(f"Data shape after dropping NaN rows: {merged_data.shape}")

        # Filter by groups if specified
        if groups is not None:
            if isinstance(groups, str):
                groups = [groups]

            if 0 not in groups and '0' not in groups:
                groups = [0] + groups  # Add 0 to the beginning
                
            if 'group' not in merged_data.columns:
                logger.warning("No 'group' column found; skipping group filter")
            else:
                before = len(merged_data)
                merged_data = merged_data[merged_data['group'].isin(groups)]
                logger.info(f"Filtered to groups: {groups}, {len(merged_data)} patients remaining (was {before})")
        
        # Store group info
        self.all_data['groups'] = merged_data['group'].unique().tolist()
        
        # Phase 2: Process data
        logger.info("=" * 60)
        logger.info("  PHASE 2 / 6 — DATA PROCESSING")
        logger.info("=" * 60)
        processing_results = self.run_processing(merged_data)

        # Phase 3: Run transcriptomics
        logger.info("=" * 60)
        logger.info("  PHASE 3 / 6 — IMAGING TRANSCRIPTOMICS (PLS)")
        logger.info("=" * 60)
        pls_results = self.run_transcriptomics(processing_results['strength_vec'])

        # Phase 4: Run enrichment
        logger.info("=" * 60)
        logger.info("  PHASE 4 / 6 — GENE SET ENRICHMENT ANALYSIS")
        logger.info("=" * 60)
        enrichment_results = self.run_enrichment(pls_results)

        # Phase 5: Generate visualizations
        logger.info("=" * 60)
        logger.info("  PHASE 5 / 6 — GENERATING VISUALIZATIONS")
        logger.info("=" * 60)
        figures = self.run_visualization(self.all_data)

        # Save figures if requested
        if self.save_figures:
            self._save_all_figures(figures)

        # Phase 6: Generate report
        logger.info("=" * 60)
        logger.info("  PHASE 6 / 6 — GENERATING PDF REPORT")
        logger.info("=" * 60)
        self.generate_report(figures, output_pdf)

        logger.info("=" * 60)
        logger.info(f"  PIPELINE COMPLETE — Report saved to: {output_pdf}")
        logger.info("=" * 60)
    
    def from_vectors(
        self,
        vectors,
        output_pdf: str,
    ):
        """
        Run pipeline starting from pre-computed vectors

        Args:
            vectors: Pre-computed vectors array
            output_pdf: Output path for final PDF report
        """
        logger.info("=" * 60)
        logger.info("  STARTING PIPELINE FROM VECTORS")
        logger.info("=" * 60)

        # Create output dir
        self.output_dir = Path(output_pdf)
        self.output_dir.mkdir(parents = True, exist_ok = True)

        # Store vectors
        self.all_data['vectors'] = vectors

        logger.info("=" * 60)
        logger.info("  PHASE 1 / 4 — IMAGING TRANSCRIPTOMICS (PLS)")
        logger.info("=" * 60)
        pls_results = self.run_transcriptomics(vectors)

        logger.info("=" * 60)
        logger.info("  PHASE 2 / 4 — GENE SET ENRICHMENT ANALYSIS")
        logger.info("=" * 60)
        enrichment_results = self.run_enrichment(pls_results)

        logger.info("=" * 60)
        logger.info("  PHASE 3 / 4 — GENERATING VISUALIZATIONS")
        logger.info("=" * 60)
        figures = self.run_visualization(self.all_data)

        if self.save_figures:
            self._save_all_figures(figures)

        logger.info("=" * 60)
        logger.info("  PHASE 4 / 4 — GENERATING PDF REPORT")
        logger.info("=" * 60)
        self.generate_report(figures, output_pdf)

        logger.info("=" * 60)
        logger.info(f"  PIPELINE COMPLETE — Report saved to: {output_pdf}")
        logger.info("=" * 60)
    
    def from_pls_results(
        self,
        pls_results,
        output_pdf,
    ):
        """
        Run pipeline starting from PLS results

        Args:
            pls_results: Pre-computed PLS results
            output_pdf: Output path for final PDF report
        """
        logger.info("=" * 60)
        logger.info("  STARTING PIPELINE FROM PLS RESULTS")
        logger.info("=" * 60)

        # Create output dir
        self.output_dir = Path(output_pdf)
        self.output_dir.mkdir(parents = True, exist_ok = True)

        # Store PLS results
        self.all_data['pls_results'] = pls_results

        logger.info("=" * 60)
        logger.info("  PHASE 1 / 3 — GENE SET ENRICHMENT ANALYSIS")
        logger.info("=" * 60)
        enrichment_results = self.run_enrichment(pls_results)

        logger.info("=" * 60)
        logger.info("  PHASE 2 / 3 — GENERATING VISUALIZATIONS")
        logger.info("=" * 60)
        figures = self.run_visualization(self.all_data)

        if self.save_figures:
            self._save_all_figures(figures)

        logger.info("=" * 60)
        logger.info("  PHASE 3 / 3 — GENERATING PDF REPORT")
        logger.info("=" * 60)
        self.generate_report(figures, output_pdf)

        logger.info("=" * 60)
        logger.info(f"  PIPELINE COMPLETE — Report saved to: {output_pdf}")
        logger.info("=" * 60)
    
    def from_enrichment_results(
        self,
        enrichment_results,
        output_pdf
    ):
        """
        Generate report from existing enrichment results
        
        Args:
            enrichment_results: Pre-computed enrichment results
            output_pdf: Output path for final PDF report
        """
        logger.info("=" * 60)
        logger.info("  STARTING PIPELINE FROM ENRICHMENT RESULTS")
        logger.info("=" * 60)

        # Create output dir
        self.output_dir = Path(output_pdf)
        self.output_dir.mkdir(parents = True, exist_ok = True)

        # Store enrichment results
        self.all_data['enrichment_results'] = enrichment_results

        logger.info("=" * 60)
        logger.info("  PHASE 1 / 2 — GENERATING VISUALIZATIONS")
        logger.info("=" * 60)
        figures = self.run_visualization(self.all_data)

        if self.save_figures:
            self._save_all_figures(figures)

        logger.info("=" * 60)
        logger.info("  PHASE 2 / 2 — GENERATING PDF REPORT")
        logger.info("=" * 60)
        self.generate_report(figures, output_pdf)

        logger.info("=" * 60)
        logger.info(f"  PIPELINE COMPLETE — Report saved to: {output_pdf}")
        logger.info("=" * 60)
    
    def _load_from_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and use a pre-merged DataFrame as pipeline input.

        The DataFrame must already contain the demographic columns (age, sex, tiv, group)
        alongside all morphometric feature columns. Missing required columns raise a
        ValueError with a descriptive message before any processing begins.

        Args:
            dataframe: Pre-merged DataFrame with morphometric + demographic data

        Returns:
            Validated copy of the DataFrame
        """
        required_demographic_cols = ['age', 'sex', 'tiv', 'group']
        validate_dataframe(dataframe, required_demographic_cols)

        # Check that numeric feature columns are present (beyond demographic ones)
        non_demo_cols = [c for c in dataframe.columns if c not in required_demographic_cols + ['patient_id', 'participant_id']]
        numeric_feature_cols = [c for c in non_demo_cols if pd.api.types.is_numeric_dtype(dataframe[c])]
        if len(numeric_feature_cols) < N_REGIONS:
            raise ValueError(
                f"DataFrame has only {len(numeric_feature_cols)} numeric feature columns; "
                f"expected at least {N_REGIONS} (one per brain region). "
                f"Make sure the DataFrame includes morphometric feature columns."
            )

        logger.info(f"Input DataFrame validated: {len(dataframe)} subjects, "
                    f"{len(numeric_feature_cols)} feature columns.")
        logger.info(f"Groups present: {sorted(dataframe['group'].unique().tolist())}")

        merged_data = dataframe.copy()
        return merged_data

    def run_loading(self, input_dir: str, demographic_file: str):
        """
        Load and merge FreeSurfer and demographic data
        
        Args:
            input_dir: Directory containing .annot files
            demographic_file: Path to demographics CSV
            
        Returns:
            Merged dataframe
        """
        self.loader = FreeSurferLoader()
        
        # Load FreeSurfer data
        freesurfer_data = self.loader.load_all_patients(input_dir)
        logger.info(f"Loaded data from {len(freesurfer_data)} patients")

        # Load demographics
        demographics = pd.read_csv(demographic_file)
        demographics = demographics.reset_index()
        logger.info(f"Loaded demographic data for {len(demographics)} patients")

        def _find_col(cols, keywords, label):
            """Find the first column whose name contains any of the keywords (case-insensitive).
            Raises ValueError with a clear message if none is found."""
            matches = [c for c in cols if any(kw in c.lower() for kw in keywords)]
            if not matches:
                raise ValueError(
                    f"Could not auto-detect '{label}' column. "
                    f"Expected a column containing one of {keywords}. "
                    f"Available columns: {list(cols)}"
                )
            if len(matches) > 1:
                logger.warning(
                    f"Multiple columns match '{label}': {matches}. "
                    f"Using '{matches[0]}'. Rename your column to exactly '{label}' to avoid ambiguity."
                )
            logger.info(f"Auto-detected '{label}' column: '{matches[0]}'")
            return matches[0]

        id_col    = _find_col(demographics.columns, ['patient_id', 'participant_id', '_id', 'id'], 'patient_id')
        age_col   = _find_col(demographics.columns, ['age'], 'age')
        sex_col   = _find_col(demographics.columns, ['sex', 'gender'], 'sex')
        tiv_col   = _find_col(demographics.columns, ['tiv', 'icv'], 'tiv')
        group_col = _find_col(demographics.columns, ['group', 'grp', 'diagnosis', 'dx'], 'group')

        demographics = demographics.rename(columns={
            id_col:    'patient_id',
            age_col:   'age',
            sex_col:   'sex',
            tiv_col:   'tiv',
            group_col: 'group',
        })

        scanner_cols = [c for c in demographics.columns if c.startswith('scanner_')]
        if scanner_cols:
            logger.info(f"Detected scanner covariate columns: {scanner_cols}")
        cols_to_keep = ['patient_id', 'age', 'sex', 'tiv', 'group'] + scanner_cols
        demographics = demographics[[col for col in cols_to_keep if col in demographics.columns]]
        
        # Merge data
        merged_data = pd.merge(freesurfer_data, demographics, on = 'patient_id', how = 'inner')
        logger.info(f"Merged data: {len(merged_data)} subjects")
        validate_patient_data(merged_data)

        # Log group distribution
        if 'group' in merged_data.columns:
            group_counts = merged_data['group'].value_counts()
            logger.info("Group distribution:")
            for group, count in group_counts.items():
                logger.info(f"  {group}: {count} patients")
        
        # Store and optionally save
        self.all_data['raw_data'] = freesurfer_data
        self.all_data['demographics'] = demographics
        # Note: merged_data is stored by the caller (run_full_pipeline) to avoid duplication

        if self.save_all:
            save_dataframe(merged_data, self.output_dir / "merged_data.csv")
        
        return merged_data
    
    def run_processing(self, data):
        """
        Process data: z-scoring and effect maps computation
        
        Args:
            data: Merged dataframe
            
        Returns:
            Dictionary containing processed data and t-maps
        """
        self.processor = DataProcessor(data)
        
        # Compute z-scores
        zscored_data = self.processor.compute_zscores()
        logger.info("Computed z-scores.")

        # Compute vectors (always include combined comparison when multiple patient groups)
        vectors = self.processor.compute_vectors()
        logger.info(f"Computed the mass univariate maps.")
        
        # Get all processed data
        results = self.processor.get_processed_data()
        
        # Store and optionally save
        self.all_data['zscored_data'] = zscored_data
        self.all_data['vectors'] = results['vectors']
        self.all_data['processing_results'] = results 
        self.all_data['similarity_mat'] = results['similarity_mat']
        self.all_data['strength_map'] = results['strength_map']    
        self.all_data['strength_vec'] = results['strength_vec']
        self.all_data['region_labels'] = results['region_labels'] 

        if self.save_all:
            save_dataframe(zscored_data, self.output_dir / "zscored_data.csv")
            save_array(results['strength_vec'], self.output_dir / "strength_maps.pkl")
            save_array(results['strength_map'], self.output_dir / "strength_values.pkl")
        
        return results
    
    def run_transcriptomics(self, vectors):
        """
        Run PLS analysis using Imaging Transcriptomics Toolbox
        
        Args:
            vectors: Input dictionary of maps
            
        Returns:
            PLS results dictionary
        """
        self.transcriptomics = TranscriptomicsAnalyzer(vectors, self.all_data.get('zscored_data'))

        pls_results = self.transcriptomics.run_pls()
        logger.info("Completed PLS analysis.")
        
        # Store and optionally save
        self.all_data['pls_results'] = pls_results

        if self.save_all:
            save_results(pls_results, self.output_dir / "pls_results.pkl")
        
        return pls_results
    
    def run_enrichment(self, pls_results):
        """
        Run enrichment and over-representation analysis.
        All .gmt files bundled in the package genes/ directory are used automatically.

        Args:
            pls_results: PLS results dictionary

        Returns:
            Enrichment results dictionary
        """
        self.enrichment = EnrichmentAnalyzer(pls_results)
        
        # Run enrichment
        enrichment_results = self.enrichment.run_enrichment()
        logger.info("Completed enrichment analysis.")
        
        # Get all results
        all_results = self.enrichment.get_all_results()
        
        # Store and optionally save
        self.all_data['enrichment_results'] = all_results

        if self.save_all:
            save_results(all_results, self.output_dir / "enrichment_results.pkl")

        for comparison_name, gene_set_results in enrichment_results.items():
            for gene_set_name, enrichment_df in gene_set_results.items():
                if enrichment_df is not None and len(enrichment_df) > 0:
                    enrichment_df.to_csv(f"{self.output_dir}/{comparison_name}_{gene_set_name}.csv", index = False)
        
        logger.info(f"Saved all enrichment results to {self.output_dir}/")
        
        return all_results
    
    def run_visualization(self, all_data):
        """
        Generate all visualizations
        
        Args:
            all_data: Dictionary containing all pipeline data
            
        Returns:
            List of figure objects
        """
        figures = []

        # Pass region labels to visualizer so it can sort regions consistently
        if 'region_labels' in all_data and all_data['region_labels'] is not None:
            self.visualizer.region_labels = all_data['region_labels']

        # Generate plots for each available phase
        if 'merged_data' in all_data:
            demo_figs = self.visualizer.plot_demographics(all_data['merged_data'])
            figures.extend(demo_figs)
            logger.info(f"Generated {len(demo_figs)} raw features plots.")
        
        if 'zscored_data' in all_data and 'merged_data' in all_data:
            zscore_figs = self.visualizer.plot_zscores(
                all_data['merged_data'],
                groups = all_data['merged_data']['group'] if 'group' in all_data['merged_data'].columns else None
            )
            figures.extend(zscore_figs)
            logger.info(f"Generated {len(zscore_figs)} raw features plots.")
                
        # Group comparisons if requested
        if 'merged_data' in all_data and 'vectors' in all_data:
            maps_figs = self.visualizer.plot_group_comparisons(
                all_data['vectors'],
                all_data['merged_data']['group']
            )
            figures.extend(maps_figs)
            logger.info(f"Generated {len(maps_figs)} difference maps plots.")

        if 'similarity_mat' in all_data and 'region_labels' in all_data:
            similarity_figs = self.visualizer.plot_similarity_matrix(
                all_data['similarity_mat'],
                all_data['region_labels'] 
            )
            figures.extend(similarity_figs)
            logger.info(f"Generated {len(similarity_figs)} average plots of similarity matrices.")

        if 'strength_map' in all_data:
            strength_map_figs = self.visualizer.plot_strength_maps(
                all_data['strength_map'] 
            )
            figures.extend(strength_map_figs)
            logger.info(f"Generated {len(strength_map_figs)} average plots of strength.")

        if 'strength_vec' in all_data:
            strength_vec_figs = self.visualizer.plot_strength_diff_maps(
                all_data['strength_vec'] 
            )
            figures.extend(strength_vec_figs)
            logger.info(f"Generated {len(strength_vec_figs)} plots of difference maps.")

        if 'pls_results' in all_data:
            pls_figs = self.visualizer.plot_pls_results(all_data['pls_results'])
            figures.extend(pls_figs)
            logger.info(f"Generated {len(pls_figs)} PLS plots")
        
        if 'enrichment_results' in all_data:
            enrich_figs = self.visualizer.plot_enrichment(all_data['enrichment_results'])
            figures.extend(enrich_figs)
            logger.info(f"Generated {len(enrich_figs)} enrichment plots")
        
        self.all_figures = figures
        return figures
    
    def _save_all_figures(self, figures):
        """
        Save all figures as individual files
        
        Args:
            figures: List of (figure, name) tuples or figure objects
        """
        logger.info(f"Saving {len(figures)} figures to {self.figures_dir}")
        
        for i, fig_data in enumerate(figures):
            # Handle both tuples (fig, name) and plain figures
            if isinstance(fig_data, tuple):
                fig, name = fig_data
            else:
                fig = fig_data
                name = f"figure_{i + 1:03d}"
            
            # Create filename
            filename = f"{name}.png"
            filepath = self.figures_dir / filename
            
            # Save figure
            save_figure(fig, filepath)
            logger.debug(f"Saved figure: {filepath}")
        
        logger.info(f"All figures saved to {self.figures_dir}")
    
    def generate_report(self, figures, output_path):
        """
        Generate final PDF report
        
        Args:
            figures: List of figure objects
            output_path: Path to save PDF
        """
        self.reporter = ReportGenerator(figures, self.all_data)
        self.reporter.generate_pdf(output_path)
        logger.info(f"Generated report: saved to {output_path}.")