"""
Generate plots for each analysis phase
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting, datasets
import nibabel as nib
from scipy.spatial.distance import squareform
from pathlib import Path
from ..processing.data_processor import N_REGIONS, N_LH_REGIONS

logger = logging.getLogger(__name__)

class Visualizer:
    """Generates all plots for each analysis phase"""
    
    def __init__(self):
        """Initialize the visualizer"""
        # Set default style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

        current_dir = Path(__file__).parent
        self.left_annot_path = current_dir.parent / 'data' / 'lh.aparc.annot'
        self.right_annot_path = current_dir.parent / 'data' / 'rh.aparc.annot'
        self.region_labels = None  
    
    def plot_demographics(self, data):
        """
        Generate demographic summary plots (age distribution, sex distribution, group sizes).

        Args:
            data: Merged DataFrame containing at least 'group', 'age', and 'sex' columns.

        Returns:
            List of matplotlib figure objects.
        """
        figures = []

        if 'group' not in data.columns:
            logger.warning("No 'group' column found; skipping demographics plots.")
            return figures

        groups = sorted(data['group'].unique())
        palette = sns.color_palette("Set2", len(groups))

        # ── Figure 1: Group sizes ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = data['group'].value_counts().sort_index()
        bars = ax.bar([str(g) for g in counts.index], counts.values,
                      color=palette, edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xlabel('Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
        ax.set_title('Sample Size per Group', fontsize=14, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        figures.append(fig)

        # ── Figure 2: Age distribution per group ─────────────────────────────
        if 'age' in data.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            for group, color in zip(groups, palette):
                ages = data.loc[data['group'] == group, 'age'].dropna()
                ax.hist(ages, bins=15, alpha=0.6, label=f'Group {group}', color=color, edgecolor='white')
            ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Age Distribution by Group', fontsize=14, fontweight='bold')
            ax.legend(title='Group', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            figures.append(fig)

        # ── Figure 3: Sex distribution per group ─────────────────────────────
        if 'sex' in data.columns:
            sex_counts = data.groupby(['group', 'sex']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(7, 4))
            sex_counts.plot(kind='bar', ax=ax, edgecolor='black', linewidth=0.8)
            ax.set_xlabel('Group', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('Sex Distribution by Group', fontsize=14, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(title='Sex', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            figures.append(fig)

        logger.info(f"Generated {len(figures)} demographics plots.")
        return figures
    
    def plot_zscores(self, data, groups = None):
        """
        Generate plots of raw features data.
        
        Args:
            data: DataFrame with all data. 
            
        Returns:
            List of (figure, name) tuples
        """
        figures = []

        plt.rcParams.update({
            "font.family":      "sans-serif",
            "font.sans-serif":  ["Helvetica Neue", "Helvetica", "Arial",
                                "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype":     42,
        })

        if groups is not None:
            merged_rows = data[data['group'].isin(groups)].copy()
            merged_rows['group'] = 'merged_groups'
            data = pd.concat([data, merged_rows], ignore_index = True)

        logger.info("Generating raw features plots.")

        # Feature columns are those that follow the pattern hemi_region_metric
        # (exclude known non-feature columns)
        non_feature = {'patient_id', 'participant_id', 'age', 'sex', 'tiv', 'group'}
        feature_cols = pd.Index([
            c for c in data.columns
            if c not in non_feature and not c.startswith('scanner_')
        ])
        n_metrics = len(feature_cols) // N_REGIONS  # should equal N_METRICS

        for i in range(n_metrics):
            metric_cols = feature_cols[i::n_metrics]
            lh_cols = metric_cols[:N_LH_REGIONS]
            rh_cols = metric_cols[N_LH_REGIONS:N_REGIONS]

            metric_name = lh_cols[0].split('_')[2]
            lh_names = lh_cols.str.split('_').str[1]
            rh_names = rh_cols.str.split('_').str[1]
    
            # Compute means per group
            lh_data = data.groupby('group')[lh_cols].mean()
            lh_data.columns = lh_names
            
            rh_data = data.groupby('group')[rh_cols].mean()
            rh_data.columns = rh_names

            # Create figure with two subplots
            fig, axes = plt.subplots(2, 1, figsize = (22, 3 + 1.5*len(lh_data)))
            
            # Left hemisphere heatmap
            sns.heatmap(lh_data, 
                        cmap = 'RdYlBu_r',
                        annot = False,
                        cbar_kws = {'label': f'{metric_name}'},
                        linewidths = 0.8,
                        linecolor = 'white',
                        ax = axes[0])
            
            axes[0].set_title('LEFT HEMISPHERE', fontsize = 13, fontweight = 'bold', loc = 'left', pad = 10)
            axes[0].set_ylabel('Group', fontsize = 11, fontweight = 'bold')
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation = 90, fontsize = 8, ha = 'right')
            axes[0].set_xlabel('')
            
            # Right hemisphere heatmap
            sns.heatmap(rh_data, 
                        cmap='RdYlBu_r',
                        annot = False,
                        cbar_kws = {'label': f'{metric_name}'},
                        linewidths = 0.8,
                        linecolor = 'white',
                        ax = axes[1])
            
            axes[1].set_title('RIGHT HEMISPHERE', fontsize = 13, fontweight = 'bold', loc = 'left', pad = 10)
            axes[1].set_ylabel('Group', fontsize = 11, fontweight = 'bold')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation = 90, fontsize = 8, ha = 'right')
            axes[1].set_xlabel('Brain Region', fontsize = 11, fontweight = 'bold')
            
            # Main title
            plt.suptitle(f'Mean {metric_name} Across Brain Regions by Hemisphere', fontsize = 15, fontweight = 'bold', y = 0.998)
            plt.tight_layout()
            figures.append(fig)
        
        logger.info(f"Generated {len(figures)} heatmaps from raw data.")
        
        return figures

    def plot_group_comparisons(self, data, groups): # Plot of difference-maps (the ones obtained with the betas) both as plots and as brain maps (brain plots)
        """
        Generate pairwise group comparison plots
        
        Args:
            data: DataFrame with data
            groups: Series with group labels
            
        Returns:
            List of (figure, name) tuples
        """
        figures = []
        
        logger.info("Generating group comparison plots")
        
        for comparison, df in data.items():
            logger.info(f"Building bar plots and brain maps plots for {comparison}...")
            metrics = df['metric'].unique()
            for metric in metrics:
                subset = df[df['metric'] == metric].copy()
                if self.region_labels is not None:
                    subset['region'] = pd.Categorical(subset['region'], categories = self.region_labels, ordered = True)
                subset = subset.sort_values('region')
                # Extract the vector of betas and the corresponding region names
                betas = subset['t_value'].values # beta 
                p_corr = subset['p_fdr'].values
                
                # Visualization on brain surface
                figures.extend(self._surface_map_visualization(betas, f'Map of t-values for {metric}, {comparison}', mesh_type = 'infl'))
                
                # Visualization on bar plot
                figures.append(self._bar_plot_visualization(betas, p_corr, f'Map of t-values for {metric}, {comparison}', subset['region'].values))
                

        logger.info(f"Generated difference maps plots.")
        
        return figures
    
    def _bar_plot_visualization(self, betas, p_corr, plot_title, region_names):
        """ Visualizes the difference map as a bar plot. 
        
        Parameters 
        ---------- 
        betas :
            The beta values to plot.
        p_corr :
            The corrected p-values for significance.
        plot_title : str 
            The title of the plot.
        region_names : 
            The names of the brain regions corresponding to each beta value.

        Returns 
        -------     
        figs :
            List of figures generated.
            The beta values to plot.
        p_corr :
            The corrected p-values for significance.
        plot_title : str 
            The title of the plot.

        Returns 
        -------     
        figs :
            List of figures generated.
        """
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize = (12, 10), sharey = True)
        fig.suptitle(plot_title, fontsize = 16, weight = 'bold')
        reg_names = [name.split('_')[-1] for name in region_names]
        y_pos = np.arange(N_REGIONS // 2)
        titles = ['Left Hemisphere', 'Right Hemisphere']
        
        cmap = plt.cm.RdBu_r  

        max_abs_beta = max(abs(betas.min()), abs(betas.max()))
        xlim = max(6, np.ceil(max_abs_beta * 1.1))  # Minimum 1.5 or 6, otherwise scale up
        norm = plt.matplotlib.colors.Normalize(vmin = -xlim, vmax = xlim)

        for i, ax in enumerate(axes):
            start = i * (N_REGIONS // 2)
            end = (i + 1) * (N_REGIONS // 2)
            hemi_data = betas[start:end]
            hemi_p = p_corr[start:end]
            hemi_names = reg_names[start:end] 
            
            ax.barh(y_pos, hemi_data, align = 'center', alpha = 0.7)
            ax.yaxis.grid(False)
            ax.axvline(0, color = 'black', linewidth = 1.2, linestyle = '-', alpha = 0.8)
            ax.set_xlim(-xlim, xlim)
            ax.set_title(titles[i])
            ax.set_xlabel('t-values', fontsize = 11)

            # Spine styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)

            if i == 0: # Shared axis label
                ax.set_yticks(y_pos)
                ax.set_yticklabels(hemi_names, fontsize = 10)
                ax.invert_yaxis()
            else:
                # Right hemisphere - also show labels for clarity
                ax.set_yticks(y_pos)
                ax.set_yticklabels(hemi_names, fontsize=10)
                
            # Asterisks for significance
            for idx, (val, p) in enumerate(zip(hemi_data, hemi_p)):
                if p < 0.05: 
                    # Only if significant
                    if p < 0.001:
                        txt = '***' 
                        color = '#000000' 
                    elif p < 0.01:
                        txt = '**'
                        color = '#424242'
                    else:
                        txt = '*'
                        color = '#757575'
                    
                    # Position of the asterisk
                    offset = 0.1 if val >= 0 else -0.1
                    ha = 'left' if val >= 0 else 'right'
                    
                    ax.text(val + offset, y_pos[idx], txt, va = 'center', ha = ha, fontweight = 'bold', fontsize = 12, color = color)
        
        plt.tight_layout()
        
        return fig
    
    def _surface_map_visualization(self, reg_map, plot_title, mesh_type = 'infl', cmap = 'bwr'):
        """ Visualizes regional data on the fsaverage5 surface for both hemispheres. 
        
        Parameters 
        ---------- 
        reg_map :
            The input data map to plot on the cortical surface.
        plot_title : str 
            The title of the plot (It is also the base name of the output file).
        mesh_type : str, optional 
            The type of mesh to plot on. Options: 'infl' (inflated), 'pial'. Default is 'infl'. 
        
        Returns 
        ------- 
        None 
            Saves figures. 
        """

        plt.rcParams.update({
            "font.family":      "sans-serif",
            "font.sans-serif":  ["Helvetica Neue", "Helvetica", "Arial",
                                "Liberation Sans", "DejaVu Sans"],
            "pdf.fonttype":     42,
        })
        # Get left and right values of effect maps 
        value_map_l, value_map_r = self._surf_maps_transform(reg_map)
        
        # Fetching the fsaverage5 mesh, which has 10242 nodes
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')

        # Repeat for each hemisphere 
        hemis = [('left', self.left_annot_path, value_map_l), ('right', self.right_annot_path, value_map_r)] 
        figs = []
        
        for hemi_name, hemi_path, regional_map in hemis:

            # Load the annotation data to then link the nodes to their correct region in the Desikan Killiany parcellation
            # Obtain labels and names  
            labels, ctab, names = nib.freesurfer.read_annot(hemi_path)
            plot_data = np.zeros(labels.shape)
            plot_data = self._map_regions(plot_data, labels, names, regional_map)
            
            # Plot 
            t_stat_lim = 6
            display = plotting.plot_surf_stat_map(
                surf_mesh = fsaverage[f'{mesh_type}_{hemi_name}'],  # infl or pial
                stat_map = plot_data,
                hemi = hemi_name,
                view = 'medial', # lateral or medial   
                bg_map = fsaverage[f'sulc_{hemi_name}'], 
                bg_on_data = True,
                darkness = .5,
                title = f'{plot_title} {hemi_name}',
                cmap = cmap,
                output_file = None,
                vmax = t_stat_lim
                )
            figs.append((display.figure))
            
        return figs

    def _surf_maps_transform(self, map_vector):
        """
            Transforms a 68-region surface map into two 36-region maps (left and right hemispheres) for visualization. Transformation consists in separating the two 
            hemispheres data and in adding two areas with 0 value: "unknown" and "corpus callosum", which are required for visualization.  
            
            Parameters
            ----------
            map_vector
                A vector of 68 values coming from a map or effect values with each representing the cortical regions in the Desikan Killiany atlas. 

            Returns
            -------
            (map_left, map_right)
                Two lists (each of length 36) representing the transformed left and right hemisphere maps.

            Raises
            ------
            ValueError
                If the input does not contain exactly 68 elements or if the output 
                vectors are not length 36.
            """
        map_vector = np.array(map_vector)

        # Check that the map is made of N_REGIONS regions
        if len(map_vector) != N_REGIONS:
            raise ValueError(
                f'Input vector map has wrong number of regions! '
                f'Expected {N_REGIONS}, but got {len(map_vector)}.'
            )

        # Create the left and right maps
        map_left = map_vector[:N_LH_REGIONS]
        map_right = map_vector[N_LH_REGIONS:]

        # Insert the values for areas present in visualization but not in patient regions:
        # Corpus Callosum (index 3) and Unknown (index 0)
        map_left = np.insert(map_left, 3, 0)
        map_right = np.insert(map_right, 3, 0)
        map_left = np.insert(map_left, 0, 0)
        map_right = np.insert(map_right, 0, 0)

        # Maps should now be N_LH_REGIONS + 2 regions long
        expected_len = N_LH_REGIONS + 2
        if (len(map_left) != expected_len) or (len(map_right) != expected_len):
            raise ValueError(
                f'Generated maps have wrong length. '
                f'Expected {expected_len}, but got {len(map_left)} (L) and {len(map_right)} (R).'
            )

        return map_left, map_right

    def _map_regions(self, plot_data, labels, names, reg_map):
        """ Maps region-wise values (from effect maps, ...) to vertex-wise array for plotting. 
        
            Parameters 
            ---------- 
            plot_data : 
                A zero-filled array of shape (N_vertices,). 
            labels :
                The annotation array where each vertex has a region ID (0..N). 
            names : 
                The list of region names corresponding to the indices in 'labels'. 
            reg_map : 
                The region values to assign to each vertex for each region
            
            Returns 
            ------- 
            plot_data :
                The updated plot_data array with values assigned to each vertex. 
            """

        for name in names:

            # After searching for the index of the region, it sets all the vertices corresponding to that region to the input map value.  
            idx = names.index(name)
            plot_data[labels == idx] = reg_map[idx]

        return plot_data

    def plot_pls_results(self, results):
        """
        Plot PLS results: top-ranked genes by Z-score for each comparison.

        For each comparison a horizontal bar chart shows the top 20 genes with
        the highest and lowest Z-scores, with FDR-significant genes highlighted.

        Args:
            results: PLS results dict {comparison_name: {'gene_df': DataFrame, ...}}

        Returns:
            List of matplotlib figure objects.
        """
        figures = []

        for comparison_name, result in results.items():
            gene_df = result.get('gene_df')
            if gene_df is None or len(gene_df) == 0:
                logger.warning(f"No gene data for {comparison_name}; skipping PLS plot.")
                continue

            sorted_df = gene_df.sort_values('Z-score', ascending=False).reset_index(drop=True)
            top_high = sorted_df.head(10)
            top_low = sorted_df.tail(10).iloc[::-1]
            plot_df = pd.concat([top_low, top_high], ignore_index=True)
            plot_df['color'] = plot_df.apply(
                lambda r: '#E57373' if (r['Z-score'] > 0 and r['fdr'] < 0.05)
                else '#64B5F6' if (r['Z-score'] < 0 and r['fdr'] < 0.05)
                else '#BDBDBD',
                axis=1
            )

            fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.45)))
            ax.barh(range(len(plot_df)), plot_df['Z-score'],
                    color=plot_df['color'], edgecolor='black', linewidth=0.5, alpha=0.85)
            ax.set_yticks(range(len(plot_df)))
            ax.set_yticklabels(plot_df['Gene'], fontsize=9)
            ax.axvline(0, color='black', linewidth=1.2)
            ax.set_xlabel('Z-score', fontsize=12, fontweight='bold')
            ax.set_title(f'PLS Gene Scores — {comparison_name}\n'
                         f'(red/blue = FDR < 0.05, grey = not significant)',
                         fontsize=13, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            figures.append(fig)

        logger.info(f"Generated {len(figures)} PLS results plots.")
        return figures
    
    def plot_enrichment(self, results):
        """
        Generate one enrichment bar plot per gene library per comparison.
        Libraries with no significant terms (FDR q-val < 0.05) are skipped.

        Args:
            results: Enrichment results dict {comparison: {library: DataFrame}}

        Returns:
            List of matplotlib figure objects.
        """
        from matplotlib.patches import Patch

        figures = []
        top_n = 10

        # Colour palette: one distinct colour per library
        library_palette = [
            '#1B5E20',  # dark green
            '#0D47A1',  # dark blue
            '#4A148C',  # dark purple
            '#BF360C',  # dark orange
            '#37474F',  # dark grey-blue
        ]

        logger.info("Generating enrichment plots...")

        for comparison_name, gene_set_results in results.items():
            for lib_idx, (library_name, df) in enumerate(gene_set_results.items()):
                if df is None or len(df) == 0:
                    logger.info(f"No results for {library_name} in {comparison_name}; skipping.")
                    continue

                # Keep only significant terms
                sig_df = df[df['FDR q-val'] < 0.05].copy()
                if sig_df.empty:
                    logger.info(f"No significant enrichment terms for {library_name} in {comparison_name}.")
                    continue

                sig_df['NES'] = pd.to_numeric(sig_df['NES'], errors='coerce')
                sig_df = sig_df.dropna(subset=['NES'])

                pos_df = sig_df[sig_df['NES'] > 0]
                neg_df = sig_df[sig_df['NES'] < 0]

                logger.info(
                    f"{library_name} / {comparison_name}: "
                    f"{len(pos_df)} positive, {len(neg_df)} negative significant terms"
                )

                # Top N from each direction
                top_pos = pos_df.nlargest(min(top_n, len(pos_df)), 'NES') if not pos_df.empty else pd.DataFrame()
                top_neg = neg_df.nsmallest(min(top_n, len(neg_df)), 'NES') if not neg_df.empty else pd.DataFrame()

                plot_df = pd.concat([top_neg, top_pos], ignore_index=True)
                if plot_df.empty:
                    continue

                # Sort ascending so positive bars point right at the bottom
                plot_df = plot_df.sort_values('NES', ascending=True).reset_index(drop=True)

                bar_colors = ['#64B5F6' if nes < 0 else '#E57373' for nes in plot_df['NES']]
                lib_color = library_palette[lib_idx % len(library_palette)]

                fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.45)))

                ax.barh(
                    y=range(len(plot_df)),
                    width=plot_df['NES'],
                    color=bar_colors,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.88,
                )

                # Term labels on the y-axis, coloured by library
                ax.set_yticks(range(len(plot_df)))
                ax.set_yticklabels(plot_df['Term'], fontsize=9, color=lib_color, fontweight='bold')

                ax.axvline(x=0, color='black', linewidth=1.2)
                nes_max = max(abs(plot_df['NES'].max()), abs(plot_df['NES'].min()), 1.0)
                xlim = round(nes_max * 1.25, 1)
                ax.set_xlim(-xlim, xlim)

                ax.set_xlabel('Normalized Enrichment Score (NES)', fontsize=12, fontweight='bold')
                ax.set_title(
                    f'{library_name}\n{comparison_name}',
                    fontsize=14, fontweight='bold', color=lib_color
                )

                # Significance asterisks
                for idx, (nes, fdr) in enumerate(zip(plot_df['NES'], plot_df['FDR q-val'])):
                    if fdr < 0.001:
                        txt, col = '***', '#000000'
                    elif fdr < 0.01:
                        txt, col = '**', '#424242'
                    elif fdr < 0.05:
                        txt, col = '*', '#757575'
                    else:
                        continue
                    offset = 0.05 * xlim if nes >= 0 else -0.05 * xlim
                    ha = 'left' if nes >= 0 else 'right'
                    ax.text(nes + offset, idx, txt,
                            va='center', ha=ha, fontweight='bold', fontsize=11, color=col)

                ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_axisbelow(True)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)

                legend_elements = [
                    Patch(facecolor='#E57373', edgecolor='black', alpha=0.88, label='Positive NES'),
                    Patch(facecolor='#64B5F6', edgecolor='black', alpha=0.88, label='Negative NES'),
                ]
                ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

                plt.tight_layout()
                figures.append(fig)

                logger.info(
                    f"Created enrichment plot for {library_name} / {comparison_name}: "
                    f"{len(plot_df)} terms ({len(top_pos)} pos, {len(top_neg)} neg)"
                )

        logger.info(f"Generated {len(figures)} enrichment plots.")
        return figures

    def plot_similarity_matrix(self, similarity_df, region_labels):
        """
        Visualize the average similarity matrix for each group and for the combined patient group.
        """
        figures = []
        
        n_pairs = N_REGIONS * (N_REGIONS - 1) // 2
        matrix_cols = similarity_df.iloc[:, 1:n_pairs + 1]
        groups = similarity_df['group'].unique()
        groups = sorted(groups)

        for group_id in groups:
            group_data = matrix_cols[similarity_df['group'] == group_id]
            
            # Calculate average across all subjects in this group and build matrix
            avg_condensed = group_data.mean(axis = 0).values
            avg_matrix = squareform(avg_condensed)
            
            # Create figure
            fig, ax = plt.subplots(figsize = (10, 8))
            im = ax.imshow(avg_matrix, cmap = 'viridis', vmax = 1, vmin = 0, aspect = 'auto')
            ax.grid(False)
            ax.set_title(f'Group {group_id} - Average similarity Matrix')
            ax.set_xlabel('Region')
            ax.set_ylabel('Region')
            tick_idx = np.arange(0, len(region_labels), 2)
            ax.set_xticks(tick_idx)
            ax.set_yticks(tick_idx)
            ax.set_xticklabels([region_labels[i] for i in tick_idx], rotation = 45, ha = 'right', fontsize = 8, fontweight = 'bold')
            ax.set_yticklabels([region_labels[i] for i in tick_idx], fontsize = 8, fontweight = 'bold')
            plt.colorbar(im, ax = ax, label = 'Similarity')
            figures.append(fig)

        # All patients combined
        if len(groups) > 2:
            all_patients_data = matrix_cols[similarity_df['group'] != 0]
            avg_condensed = all_patients_data.mean(axis = 0).values
            avg_matrix = squareform(avg_condensed)

            fig, ax = plt.subplots(figsize = (10, 8))
            im = ax.imshow(avg_matrix, cmap = 'viridis', vmax = 1, vmin = 0, aspect = 'auto')
            ax.grid(False)
            ax.set_title(f'All patient groups - Average similarity Matrix')
            ax.set_xlabel('Region')
            ax.set_ylabel('Region')
            tick_idx = np.arange(0, len(region_labels), 2)
            ax.set_xticks(tick_idx)
            ax.set_yticks(tick_idx)
            ax.set_xticklabels([region_labels[i] for i in tick_idx], rotation = 45, ha = 'right', fontsize = 8, fontweight = 'bold')
            ax.set_yticklabels([region_labels[i] for i in tick_idx], fontsize = 8, fontweight = 'bold')
            plt.colorbar(im, ax = ax, label = 'Similarity')
            figures.append(fig) 
        
        return figures

    def plot_strength_maps(self, strength_df):
        """
        Visualize the average strength map for each group and for the combined patient group both as a heatmap and as brain maps
        """
        figures = []

        region_names = strength_df.columns[1:N_REGIONS + 1].tolist()
        group_means = strength_df.groupby('group')[region_names].mean()

        # Mean for all patients combined (everything except group 0)
        union_mask = strength_df['group'] != 0
        union_mean_series = strength_df.loc[union_mask, region_names].mean()
        union_row = pd.DataFrame(union_mean_series).T
        union_row.index = ['All patients']
        combined_means = pd.concat([group_means, union_row])
        combined_means.index = combined_means.index.astype(str)

        # First heatmap 
        fig = plt.figure(figsize = (24, 5))
        ax = sns.heatmap(combined_means, cmap = 'viridis', annot = False, cbar_kws = {'label': 'Strength'}, xticklabels = True, square = True)
        plt.title('Average regional strength: All Groups')
        plt.ylabel('Group', fontweight = 'bold')
        plt.xlabel('Brain Regions', fontweight = 'bold')
        plt.xticks(rotation = 45, ha = 'right', fontsize = 8, fontweight = 'bold') 
        plt.tight_layout()
        figures.append(fig)

        # Second heatmap 
        subset_rows = ['0', 'All patients']
        subset_means = combined_means.loc[combined_means.index.isin(subset_rows)]

        fig = plt.figure(figsize = (24, 4))
        ax = sns.heatmap(subset_means, cmap = 'viridis', annot = False, cbar_kws = {'label': 'Strength'}, xticklabels = True)
        plt.title('Average regional strength: HC (Group 0) vs Patients')
        plt.ylabel('Group')
        plt.xlabel('Brain Regions')
        plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
        plt.tight_layout()
        figures.append(fig)

        # Strength maps on brain surface
        # for idx, row in combined_means.iterrows():
        #     figures.extend(self._surface_map_visualization(row, f'Strength map average for group {idx}.', mesh_type = 'infl', cmap = 'viridis')) 

        return figures
    
    def plot_strength_diff_maps(self, strength_dict):
        """
        Visualize the difference maps (vs HC) with betas, for each group and for the combined patient group.
        """
        figures = []
        
        for comparison, df in strength_dict.items():
            logger.info(f"Building bar plots and brain surface plots for {comparison}...")

            subset = df.copy()
            # Enforce original region ordering
            if self.region_labels is not None:
                subset['region'] = pd.Categorical(subset['region'], categories = self.region_labels, ordered = True)
            subset = subset.sort_values('region')
            
            # Extract the vector of betas and the corresponding region names
            betas = subset['t_value'].values # beta 
            p_corr = subset['p_fdr'].values
            
            # Visualization on brain surface
            figures.extend(self._surface_map_visualization(betas, f'Map of t-values for strength, {comparison}', mesh_type = 'infl'))
            
            # Visualization on brain surface : only significant
            betas_significant = betas.copy()
            betas_significant[p_corr >= 0.05] = 0  
            figures.extend(self._surface_map_visualization(betas_significant, f'Significant t-value map for strength, {comparison}', mesh_type = 'infl'))

            # Visualization on bar plot
            figures.append(self._bar_plot_visualization(betas, p_corr, f'Map of t-values for strength, {comparison}', subset['region'].values))

        logger.info(f"Generated difference maps plots.")

        return figures

    def get_all_figures(self):
        """
        Get all generated figures
        
        Returns:
            List of (figure, name) tuples
        """
        return []