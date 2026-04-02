"""
Command-line interface for the imaging transcriptomics pipeline
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from .pipeline import Pipeline
from .utils import setup_logging

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description = "MSN Transcriptomics Pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter
    )
    
    # Global arguments
    parser.add_argument(
        '--verbose', '-v',
        action = 'store_true',
        help = 'Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest = 'command', help = 'Available commands')
    
    # Full pipeline command
    full_parser = subparsers.add_parser(
        'full',
        help = 'Run the complete pipeline from .annot files to final report'
    )
    full_parser.add_argument(
        '--input',
        required = False,
        default = None,
        type = str,
        help = 'Directory containing patients\' .stats files (required unless --dataframe is used)'
    )
    full_parser.add_argument(
        '--demographics',
        required = False,
        default = None,
        type = str,
        help = 'Path to demographics CSV file (required unless --dataframe is used)'
    )
    full_parser.add_argument(
        '--dataframe',
        required = False,
        default = None,
        type = str,
        help = 'Path to a pre-merged CSV file (morphometric features + demographics). '
               'When provided, skips FreeSurfer loading and demographics merging.'
    )
    full_parser.add_argument(
        '--output',
        required = True,
        type = str,
        help = 'Output path for final PDF report'
    )
    full_parser.add_argument(
        '--save-all',
        action = 'store_true',
        help = 'Save everything, even intermediate results to disk'
    )
    full_parser.add_argument(
        '--save-figures',
        action = 'store_true',
        help = 'Save all figures as separate image files in addition to PDF'
    )
    full_parser.add_argument(
        '--figures-dir',
        type = str,
        default = './figures',
        help = 'Directory to save individual figures (default: ./figures)'
    )
    full_parser.add_argument(
        '--groups',
        nargs = '+',
        default = None,
        help = 'Specific groups to analyze (if not specified, analyzes all groups)'
    )
    
    # From vectors command
    vectors_parser = subparsers.add_parser(
        'from-vectors',
        help = 'Run pipeline starting from previously obtained vectors'
    )
    vectors_parser.add_argument(
        '--vectors',
        required = True,
        type = str,
        help = 'Path to strength vectors file (.pkl saved by the pipeline, or .csv)'
    )
    vectors_parser.add_argument(
        '--output',
        required = True,
        type = str,
        help = 'Output path for final PDF report'
    )
    vectors_parser.add_argument(
        '--save-all',
        action = 'store_true',
        help = 'Save intermediate results to disk'
    )
    vectors_parser.add_argument(
        '--save-figures',
        action = 'store_true',
        help = 'Save all figures as separate image files'
    )
    vectors_parser.add_argument(
        '--figures-dir',
        type = str,
        default = './figures',
        help = 'Directory to save individual figures'
    )
    
    # From PLS results command
    pls_parser = subparsers.add_parser(
        'from-pls',
        help = 'Run pipeline starting from PLS results'
    )
    pls_parser.add_argument(
        '--pls-results',
        required = True,
        type = str,
        help = 'Path to PLS results file (.pkl or .csv)'
    )
    pls_parser.add_argument(
        '--output',
        required = True,
        type = str,
        help = 'Output path for final PDF report'
    )
    pls_parser.add_argument(
        '--save-all',
        action = 'store_true',
        help = 'Save everything, even intermediate results to disk'
    )
    pls_parser.add_argument(
        '--save-figures',
        action = 'store_true',
        help = 'Save all figures as separate image files'
    )
    pls_parser.add_argument(
        '--figures-dir',
        type = str,
        default = './figures',
        help = 'Directory to save individual figures'
    )
    
    # From enrichment results command
    enrich_parser = subparsers.add_parser(
        'from-enrichment',
        help = 'Generate report from existing enrichment results'
    )
    enrich_parser.add_argument(
        '--enrichment-results',
        required = True,
        type = str,
        help = 'Path to enrichment results file (.pkl)'
    )
    enrich_parser.add_argument(
        '--output',
        required = True,
        type = str,
        help = 'Output path for final PDF report'
    )
    enrich_parser.add_argument(
        '--save-figures',
        action = 'store_true',
        help ='Save all figures as separate image files'
    )
    enrich_parser.add_argument(
        '--figures-dir',
        type = str,
        default ='./figures',
        help ='Directory to save individual figures'
    )
    
    # List libraries command
    list_parser = subparsers.add_parser(
        'list-libraries',
        help = 'List available gene libraries'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose = args.verbose)
    
    # Handle commands
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'list-libraries':
            handle_list_libraries()
        elif args.command == 'full':
            handle_full_pipeline(args)
        elif args.command == 'from-vectors':
            handle_from_vectors(args)
        elif args.command == 'from-pls':
            handle_from_pls(args)
        elif args.command == 'from-enrichment':
            handle_from_enrichment(args)
    except Exception as e:
        print(f"Error: {e}", file = sys.stderr)
        sys.exit(1)

def handle_list_libraries():
    """List available gene libraries"""
    from .genes import list_available_libraries
    
    libraries = list_available_libraries()
    print("Available gene libraries:")
    for lib in libraries:
        print(f"  - {lib}")

def handle_full_pipeline(args):
    """ Handling of the full pipeline execution """
    # Validate that either --dataframe or both --input and --demographics are provided
    if args.dataframe is None and (args.input is None or args.demographics is None):
        print(
            "Error: You must provide either --dataframe or both --input and --demographics.",
            file=sys.stderr
        )
        sys.exit(1)

    # Load dataframe from CSV if provided
    dataframe = None
    if args.dataframe is not None:
        dataframe = pd.read_csv(args.dataframe)

    pipeline = Pipeline(
        save_all = args.save_all,
        save_figures = args.save_figures,
        figures_dir = args.figures_dir
    )

    pipeline.run_full_pipeline(
        input_dir = args.input,
        demographic_file = args.demographics,
        dataframe = dataframe,
        output_pdf = args.output,
        groups = args.groups,
    )

    print(f"Pipeline completed successfully! Report saved to: {args.output}")
    if args.save_figures:
        print(f"Individual figures saved to: {args.figures_dir}")

def handle_from_vectors(args):
    """Handle pipeline from vectors"""
    import pickle
    # Load vectors — accepts the .pkl file saved by the pipeline (strength_maps.pkl)
    if args.vectors.endswith('.pkl'):
        with open(args.vectors, 'rb') as f:
            vectors = pickle.load(f)
    elif args.vectors.endswith('.csv'):
        # Convenience: CSV with columns [region, beta, comparison] or wide format
        # For robustness require pkl; CSV support is best-effort
        vectors = pd.read_csv(args.vectors)
    else:
        raise ValueError("Vectors file must be .pkl or .csv format")
    
    pipeline = Pipeline(
        save_all = args.save_all,
        save_figures = args.save_figures,
        figures_dir = args.figures_dir
    )
    
    pipeline.from_vectors(
        vectors = vectors,
        output_pdf = args.output,
    )
    
    print(f"Pipeline completed successfully! Report saved to: {args.output}")
    if args.save_figures:
        print(f"Individual figures saved to: {args.figures_dir}")

def handle_from_pls(args):
    """Handle pipeline from PLS results"""
    import pickle
    
    # Load PLS results
    if args.pls_results.endswith('.pkl'):
        with open(args.pls_results, 'rb') as f:
            pls_results = pickle.load(f)
    else:
        raise ValueError("PLS results must be .pkl format")
    
    pipeline = Pipeline(
        save_all = args.save_all,
        save_figures = args.save_figures,
        figures_dir = args.figures_dir
    )
    
    pipeline.from_pls_results(
        pls_results = pls_results,
        output_pdf = args.output,
    )
    
    print(f"Pipeline completed successfully! Report saved to: {args.output}")
    if args.save_figures:
        print(f"Individual figures saved to: {args.figures_dir}")

def handle_from_enrichment(args):
    """Handle report generation from enrichment results"""
    import pickle
    
    # Load enrichment results
    with open(args.enrichment_results, 'rb') as f:
        enrichment_results = pickle.load(f)
    
    pipeline = Pipeline(
        save_figures = args.save_figures,
        figures_dir = args.figures_dir
    )
    
    pipeline.from_enrichment_results(
        enrichment_results = enrichment_results,
        output_pdf = args.output
    )
    
    print(f"Report generated successfully! Saved to: {args.output}")
    if args.save_figures:
        print(f"Individual figures saved to: {args.figures_dir}")

if __name__ == "__main__":
    main()