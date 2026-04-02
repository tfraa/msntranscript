"""
Generate final PDF report with all graphics and results
"""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Creates final PDF report with all graphics and data summaries"""
    
    def __init__(self, figures, data_summaries):
        """
        Initialize the report generator
        
        Args:
            figures: List of matplotlib figures to include
            data_summaries: Dictionary containing data summaries
        """
        self.figures = figures
        self.data_summaries = data_summaries
    
    def generate_pdf(self, output_path):
        """
        Generate the PDF report
        
        Args:
            output_path: Path to save the PDF
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        
        logger.info(f"Generating PDF report: {output_path}")
        
        savepath = output_path / "Report.pdf"

        with PdfPages(savepath) as pdf:
            # Add title page
            self._add_title_page(pdf)
            
            # Add summary statistics page
            self._add_summary_page(pdf)
            
            # Add all figures
            for i, fig in enumerate(self.figures):
                logger.debug(f"Adding figure {i+1}/{len(self.figures)}")
                pdf.savefig(fig, bbox_inches = 'tight')
                plt.close(fig)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Imaging Transcriptomics Analysis Report'
            d['Author'] = 'MSN pipeline - Fra'
            d['Subject'] = 'MSN and Imaging Transcriptomics Analysis'
            d['Keywords'] = 'Transcriptomics, Imaging, FreeSurfer, PLS, Enrichment'
        
        logger.info(f"PDF report generated successfully: {output_path}")
    
    def _add_title_page(self, pdf):
        """
        Add a title page to the PDF
        
        Args:
            pdf: PdfPages object
        """
        fig = plt.figure(figsize = (8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Title
        ax.text(
            0.5, 0.7,
            'Morphometric similarity networks and imaging transcriptomics\nAnalysis Report',
            ha = 'center',
            va = 'center',
            fontsize = 24,
            fontweight = 'bold'
        )
        
        # Date
        from datetime import datetime
        date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ax.text(
            0.5, 0.4,
            f'Generated: {date_str}',
            ha = 'center',
            va = 'center',
            fontsize = 12
        )
        
        # Sample info
        if 'merged_data' in self.data_summaries:
            n_patients = len(self.data_summaries['merged_data'])
            ax.text(
                0.5, 0.3,
                f'Number of Patients: {n_patients}',
                ha = 'center',
                va = 'center',
                fontsize = 12
            )
        
        pdf.savefig(fig, bbox_inches ='tight')
        plt.close(fig)
    
    def _add_text_page(self, pdf, title, text):
        """
        Render a single page with a bold title and monospace body text.

        Args:
            pdf: PdfPages object
            title: Section title displayed at the top of the page
            text: Body text (monospace)
        """
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.text(0.05, 0.97, title, ha='left', va='top',
                fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.92, text, ha='left', va='top',
                fontsize=9, family='monospace', transform=ax.transAxes,
                wrap=False)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _add_summary_page(self, pdf):
        """
        Add paginated summary statistics — one section per PDF page.

        Args:
            pdf: PdfPages object
        """
        # ── Section 1: Demographics ─────────────────────────────────────────
        demo_text = ""
        if 'merged_data' in self.data_summaries:
            data = self.data_summaries['merged_data']
            demo_text += f"Total subjects: {len(data)}\n\n"

            if 'group' in data.columns:
                groups = sorted(data['group'].unique())
                demo_text += "Demographics by Group:\n\n"
                for group in groups:
                    gd = data[data['group'] == group]
                    n = len(gd)
                    demo_text += f"  Group {group}:  n = {n}  ({n / len(data) * 100:.1f}%)\n"
                    if 'age' in data.columns:
                        demo_text += f"    Age:  {gd['age'].mean():.1f} ± {gd['age'].std():.1f} years\n"
                    if 'sex' in data.columns:
                        for sex, cnt in gd['sex'].value_counts().items():
                            demo_text += f"    {sex}: {cnt} ({cnt / n * 100:.1f}%)\n"
                    demo_text += "\n"
                demo_text += "Overall:\n"
                if 'age' in data.columns:
                    demo_text += f"  Age: {data['age'].mean():.1f} ± {data['age'].std():.1f} years\n"
                if 'sex' in data.columns:
                    for sex, cnt in data['sex'].value_counts().items():
                        demo_text += f"  {sex}: {cnt} ({cnt / len(data) * 100:.1f}%)\n"
            else:
                if 'age' in data.columns:
                    demo_text += f"Age: {data['age'].mean():.1f} ± {data['age'].std():.1f}\n"
                if 'sex' in data.columns:
                    for sex, cnt in data['sex'].value_counts().items():
                        demo_text += f"  {sex}: {cnt} ({cnt / len(data) * 100:.1f}%)\n"

        if demo_text:
            self._add_text_page(pdf, "Summary — Demographics", demo_text)

        # ── Section 2: Mass Univariate (betas / vectors) ─────────────────────
        if 'vectors' in self.data_summaries:
            for comparison, df in self.data_summaries['vectors'].items():
                beta_text = f"Comparison: {comparison}\n\n"
                for metric in df['metric'].unique():
                    subset = df[df['metric'] == metric].copy()
                    sig = subset[subset['significant'] == True].sort_values('p_fdr')
                    beta_text += f"{metric.capitalize()}: {len(sig)} significant regions\n"
                    for _, row in sig.iterrows():
                        beta_text += f"  • {row['region']}  β = {row['beta']:.3f}  FDR = {row['p_fdr']:.4f}\n"
                    if len(sig) == 0:
                        beta_text += "  (none)\n"
                    beta_text += "\n"
                self._add_text_page(pdf, "Summary — Mass Univariate (Betas)", beta_text)

        # ── Section 3: Morphometric Similarity Network Strength ───────────────
        if 'strength_vec' in self.data_summaries:
            for comparison, df in self.data_summaries['strength_vec'].items():
                st_text = f"Comparison: {comparison}\n\n"
                sig = df[df['significant'] == True].sort_values('p_fdr')
                st_text += f"{len(sig)} significant regions\n\n"
                for _, row in sig.iterrows():
                    st_text += f"  • {row['region']}  t = {row['t_value']:.3f}  FDR = {row['p_fdr']:.4f}\n"
                if len(sig) == 0:
                    st_text += "  (none)\n"
                self._add_text_page(pdf, "Summary — MSN Strength Differences", st_text)

        # ── Section 4: PLS / Gene Results ─────────────────────────────────────
        if 'pls_results' in self.data_summaries:
            for comparison, result in self.data_summaries['pls_results'].items():
                n_sig = result.get('n_significant', 0)
                gene_df = result.get('gene_df')
                pls_text = f"Comparison: {comparison}\n"
                pls_text += f"Significant genes (FDR < 0.05): {n_sig}\n\n"
                if gene_df is not None and len(gene_df) > 0:
                    sorted_df = gene_df.sort_values('Z-score', ascending=False)
                    pls_text += "Top 10 genes (highest Z-score):\n"
                    for i, (_, row) in enumerate(sorted_df.head(10).iterrows(), 1):
                        pls_text += f"  {i:2d}. {row['Gene']:<20} Z = {row['Z-score']:>7.3f}  FDR = {row['fdr']:>8.4f}\n"
                    pls_text += "\nTop 10 genes (lowest Z-score):\n"
                    for i, (_, row) in enumerate(sorted_df.tail(10).iloc[::-1].iterrows(), 1):
                        pls_text += f"  {i:2d}. {row['Gene']:<20} Z = {row['Z-score']:>7.3f}  FDR = {row['fdr']:>8.4f}\n"
                else:
                    pls_text += "  No gene data available.\n"
                self._add_text_page(pdf, "Summary — PLS Gene Results", pls_text)

        # ── Section 5: Enrichment Results ─────────────────────────────────────
        if 'enrichment_results' in self.data_summaries:
            for comparison, gene_set_results in self.data_summaries['enrichment_results'].items():
                enr_text = f"Comparison: {comparison}\n\n"
                for gene_set_name, terms_df in gene_set_results.items():
                    if terms_df is None or len(terms_df) == 0:
                        enr_text += f"  {gene_set_name}: no results\n\n"
                        continue
                    sig_terms = terms_df[terms_df['FDR q-val'] < 0.05].sort_values('FDR q-val')
                    enr_text += f"  {gene_set_name}: {len(sig_terms)} significant terms\n"
                    for _, row in sig_terms.iterrows():
                        enr_text += f"    • {row['Term']}  NES = {row['NES']:>7.3f}  FDR = {row['FDR q-val']:.4f}\n"
                    enr_text += "\n"
                self._add_text_page(pdf, "Summary — Gene Set Enrichment", enr_text)