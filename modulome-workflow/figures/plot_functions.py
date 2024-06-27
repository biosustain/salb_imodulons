"""
Plotting functions for iModulons
"""
import logging
import warnings
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import math
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from scipy import sparse, stats
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from os import path

from pymodulon.compare import convert_gene_index
from pymodulon.enrichment import parse_regulon_str
from pymodulon.util import _parse_sample, dima, explained_variance, mutual_info_distance

###################
# Import ica_data #
###################

from pymodulon.plotting import *
from pymodulon.io import *

#precise2_pcoli2_json_path_batch_corrected = path.join('data','precise1k','precise1k_pcoli2_curated.json.gz')
precise2_pcoli2_json_path_batch_corrected = path.join('data','precise1k','precise1k_pcoli2_batch_corrected.json.gz') # Enter precise json filename here.
ica_data = load_json_model(precise2_pcoli2_json_path_batch_corrected)


#######################
# Plotting parameters #
#######################

matplotlib.rcParams["font.size"] = 16
# format figures
plt.rcParams['figure.dpi'] = 120
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Also, if you want your axes lines to be true black, and not this weird dark gray:
matplotlib.rcParams["text.color"] = "black"
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"

# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams["font.sans-serif"] = "Arial"
# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams["font.family"] = "sans-serif"
# Also, "make fonts a bit larger"
#matplotlib.rcParams["font.size"] = 16
matplotlib.rcParams["font.size"] = 12

plt.rcParams.update({'legend.fontsize': 12})


#############
# Bar Plots #
#############


def barplot_pcoli2(
    values,
    sample_table,
    ylabel="",
    projects=None,
    highlight=None,
    ax=None,
    legend_kwargs=None,
):
    """
    Creates an overlaid scatter and barplot for a set of values (either gene
    expression levels or iModulon activities)
    Parameters
    ----------
    values: ~pandas.Series
        List of `values` to plot
    sample_table: ~pandas.DataFrame
        Sample table from :class:`~pymodulon.core.IcaData` object
    ylabel: str, optional
        Y-axis label
    projects: list or str, optional
        Name(s) of `projects` to show (default: show all)
    highlight: list or str, optional
        Project(s) to `highlight` (default: None)
    ax: ~matplotlib.axes.Axes, optional
        Axes object to plot on, otherwise use current Axes
    legend_kwargs: dict, optional
        Additional keyword arguments passed to :func:`matplotlib.pyplot.legend`
    Returns
    -------
    ax: ~matplotlib.axes.Axes
        :class:`~matplotlib.axes.Axes` containing the barplot
    """
    # Remove extra projects
    if isinstance(projects, str):
        projects = [projects]

    if projects is not None and len(projects) == 1:
        highlight = projects

    if projects is not None and "project" in sample_table:
        sample_table = sample_table[sample_table.project.isin(projects)]
        values = values[sample_table.index]

    if ax is None:
        figsize = (len(values) / 15 + 0.5, 2)
        fig, ax = plt.subplots(figsize=figsize)

    # Get ymin and max
    ymin = values.min()
    ymax = values.max()
    yrange = ymax - ymin
    ymax = max(1, max(ymax * 1.1, ymax + yrange * 0.1))
    ymin = min(-1, min(ymin * 1.1, ymin - yrange * 0.1))
    yrange = ymax - ymin


    # Add project-specific information
    if "project" in sample_table.columns and "condition" in sample_table.columns:

        # Sort data by project/condition to ensure replicates are together
        metadata = sample_table.loc[:, ["project", "condition"]]
        metadata["x_cr"] = [6,7,4,5,19,20,17,18,10,11,8,9,12,13,15,16,14,1,2,3,23,31,29,33,24,32,30,34,21,35,37,41,27,39,22,36,38,42,28,40,43,25,44,26]
        metadata = metadata.sort_values(["x_cr"])
        metadata["name"] = metadata.project + " - " + metadata.condition.astype(str)
        metadata["name_cr"] = [
        'c_Empty_1 (1)',
        'c_Empty_1_noind (1)',
        'c_Empty_1_noind (1)',
        'c_CC1 (1)',
        'c_CC1 (1)',
        'MBP (1)',
        'MBP (1)',
        'BLG (1)',
        'BLG (1)',
        'alactalbumin (1)',
        'alactalbumin (1)',
        'Ovalbumin (1)',
        'Ovalbumin (1)',
        'sfGFP (1)',
        'mfp (1)',
        'mfp (1)',
        'Brazzein (1)',
        'Brazzein (1)',
        'MNEI (1)',
        'MNEI (1)',
        'c_Empty (2)',
        'c_Empty (2)',
        'c_Empty_gly2 (2)',
        'c_Empty_gly2 (2)',
        'c_T7_pET-empty (2)',
        'c_T7_pET-empty (2)',
        'c_Empty_3HR (2)',
        'c_Empty_3HR (2)',
        'BLG_gly2 (2)',
        'BLG_gly2 (2)',
        'EGFP-BLG_gly2 (2)',
        'EGFP-BLG_gly2 (2)',
        'mfp5_gly2 (2)',
        'mfp5_gly2 (2)',
        'EGFP (2)',
        'EGFP (2)',
        'EGFP-BLG (2)',
        'EGFP-BLG (2)',
        'EGFP-BLG_3HR (2)',
        'EGFP-BLG_3HR (2)',
        'EGFP-alactalbumin (2)',
        'EGFP-alactalbumin (2)',
        'T7_pET-BLG (2)',
        'T7_pET-BLG (2)']
        metadata["name"] = metadata["name_cr"]

        
        # Coerce highlight to iterable
        if highlight is None:
            highlight = []
        elif isinstance(highlight, str):
            highlight = [highlight]

        # Get X and Y values for scatter points
        metadata["y"] = values
        metadata["x"] = np.cumsum(~metadata[["name"]].duplicated())
        # Get heights for barplot
        bar_vals = metadata.groupby("x").mean()

        # Add colors and names
        bar_vals["name"] = metadata.drop_duplicates("name").name.values
        bar_vals["project"] = metadata.drop_duplicates("name").project.values
        #print(bar_vals)





        # Plot bars for highlighted samples
        color_vals = bar_vals[bar_vals.project.isin(highlight)]
        color_cycle = [

            "tab:red",
            "tab:orange",
            "tab:green",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",

            # "black",        #1      'c_Empty_1 (1)',
            # "gray",         #2      'c_Empty_1_noind (1)',
            # "lightgray",    #3      'c_CC1 (1)',
            # "tab:purple",       #4      'MBP (1)',
            # "tab:red",        #5      'BLG (1)',
            # "tab:orange",         #6      'alactalbumin (1)',
            # "gold",         #7      'Ovalbumin (1)',
            # "tab:green",        #8      'sfGFP (1)',
            # "tab:cyan",         #9      'mfp (1)'
            # "tab:blue",          #10     'Brazzein (1)',
            # "navy",          #11     'MNEI (1)',
            # "black",        #12     'c_Empty (2)',
            # "dimgray",       #13     'c_Empty_gly2 (2)',
            # "gray",        #14     'c_T7_pET-empty (2)',
            # "lightgray",         #15     'c_Empty_3HR (2)',
            # "tab:red",         #16     'BLG_gly2 (2)',
            # "tab:green",        #17     'EGFP-BLG_gly2 (2)',
            # "tab:cyan",         #18     'mfp5_gly2 (2)',
            # "tab:green",         #19     'EGFP (2)',
            # "tab:red",         #20     'EGFP-BLG (2)',
            # "darkred",        #21     'EGFP-BLG_3HR (2)',
            # "tab:orange",         #22     'EGFP-alactalbumin (2)',
            # "darkred",         #23     'T7_pET-BLG (2)',
            
            # "gray",        #1      'c_Empty_1 (1)',
            # "gray",         #2      'c_Empty_1_noind (1)',
            # "gray",    #3      'c_CC1 (1)',
            # "gray",       #4      'MBP (1)',
            # "gray",        #5      'BLG (1)',
            # "gray",         #6      'alactalbumin (1)',
            # "gray",         #7      'Ovalbumin (1)',
            # "gray",        #8      'sfGFP (1)',
            # "gray",         #9      'mfp (1)'
            # "red",          #10     'Brazzein (1)',
            # "gray",          #11     'MNEI (1)',
            # "gray",        #12     'c_Empty (2)',
            # "gray",       #13     'c_Empty_gly2 (2)',
            # "gray",        #14     'c_T7_pET-empty (2)',
            # "gray",         #15     'c_Empty_3HR (2)',
            # "gray",         #16     'BLG_gly2 (2)',
            # "gray",        #17     'EGFP-BLG_gly2 (2)',
            # "gray",         #18     'mfp5_gly2 (2)',
            # "gray",         #19     'EGFP (2)',
            # "gray",         #20     'EGFP-BLG (2)',
            # "gray",        #21     'EGFP-BLG_3HR (2)',
            # "gray",         #22     'EGFP-alactalbumin (2)',
            # "gray",         #23     'T7_pET-BLG (2)',

        ]
        i = 0
        for name, group in color_vals.groupby("x"):
            


            if name in [1, 2, 3, 12, 13, 14, 15]:
                ax.bar(
                    group.index,
                    group.y,
                    color=color_cycle[i],
                    width=1,
                    linewidth=0,
                    align="edge",
                    zorder=1,
                    hatch = ['/'],
                    label=group.name,
                )
            else:
                ax.bar(
                    group.index,
                    group.y,
                    color=color_cycle[i],
                    width=1,
                    linewidth=0,
                    align="edge",
                    zorder=1,
                    label=group.name,
                )
            i = (i + 1) % len(color_cycle)

        # Plot bars for non-highlighted samples
        other_vals = bar_vals[~bar_vals.project.isin(highlight)]
        ax.bar(
            other_vals.index,
            other_vals.y,
            color="tab:blue",
            width=1,
            linewidth=0,
            align="edge",
            zorder=1,
            label=None,
        )
        ax.scatter(metadata.x + 0.5, metadata.y, color="k", zorder=2, s=10)
        # Get project names and sizes
        projects = metadata.project.drop_duplicates()
        md_cond = metadata.drop_duplicates(["name"])
        project_sizes = [len(md_cond[md_cond.project == proj]) for proj in projects]
        nbars = len(md_cond)

        # Draw lines to discriminate between projects
        proj_lines = np.cumsum([1] + project_sizes)
        ax.vlines(proj_lines, ymin, ymax, colors="lightgray", linewidth=1)
        ax.vlines(12, ymin, ymax, colors="lightgray", linewidth=1)

        # Add project names
        texts = []
        start = 2
        for proj, size in zip(projects, project_sizes):
            x = start + size / 2
            texts.append(
                ax.text(
                    x, ymin - yrange * 0.02, proj, ha="right", va="top", rotation=45
                )
            )
            start += size

        # Add legend
        if not color_vals.empty:
            kwargs = {
                "bbox_to_anchor": (1, 1),
                "ncol": len(color_vals.name.unique()) // 6 + 1,
            }

            if legend_kwargs is not None:
                kwargs.update(legend_kwargs)

            ax.legend(**kwargs)

    else:
        logging.warning("Missing `project` and `condition` columns in `sample_table.`")
        ax.bar(range(len(values)), values, width=1, align="edge")
        nbars = len(values)

    # Set axis limits
    xmin = -0.5
    xmax = nbars + 2.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Axis labels
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks([])

    # X-axis
    ax.hlines(0, xmin, xmax, color="k")

    return ax

def plot_pcoli2(
    ica_data, imodulon, projects="pcoli2", highlight="pcoli2", ax=None, legend_kwargs=None
):
    # Check that iModulon exists
    if imodulon in ica_data.A.index:
        values = ica_data.A.loc[imodulon]
    else:
        raise ValueError(f"iModulon does not exist: {imodulon}")
    label = "iModulon: {}".format(imodulon)

    return barplot_pcoli2(
        values=values,
        sample_table=ica_data.sample_table,
        ylabel=label,
        projects=projects,
        highlight=highlight,
        ax=ax,
        legend_kwargs=legend_kwargs,
    )

def plot_pcoli2_expression(
    ica_data, gene, projects="pcoli2", highlight="pcoli2", ax=None, legend_kwargs=None
):
    # Check that gene exists
    if gene in ica_data.X.index:
        values = ica_data.X.loc[gene]
        label = "{} Expression".format(gene)
    else:
        locus = ica_data.name2num(gene)
        values = ica_data.X.loc[locus]
        label = "${}$ Expression".format(gene)

    return barplot_pcoli2(
        values=values,
        sample_table=ica_data.sample_table,
        ylabel=label,
        projects=projects,
        highlight=highlight,
        ax=ax,
        legend_kwargs=legend_kwargs,
    )

##############
# Phaseplane #
##############

qualitative_colors = sns.color_palette("Set3",10)
qualitative_colors

def phaseplane(iM1 = "Fur-1", iM2 = "Fur-2", 
               sample1 = [None], sample2 = [None], sample3 = [None], sample4 = [None], sample5 = [None], sample6 = [None], sample7 = [None], sample8 = [None], sample9 = [None],
               color1 = qualitative_colors[3], color2 = qualitative_colors[4], color3 = qualitative_colors[0], color4 = qualitative_colors[1], color5 = qualitative_colors[2], 
               color6 = qualitative_colors[5], color7 = qualitative_colors[6], color8 = qualitative_colors[7], color9 = qualitative_colors[8],
               label1 = [None], label2 = [None], label3 = [None], label4 = [None], label5 = [None], label6 = [None], label7 = [None], label8 = [None], label9 = [None],
               legend_loc = "lower right"
               ):
    
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    
    #Plot all of Precise samples as gray scatters
    x = ica_data.A.loc[iM1]
    y = ica_data.A.loc[iM2]
    ax.scatter(x, y, color = "gray", alpha=1)

    sample_lst = [sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8, sample9]
    color_lst = [color1, color2, color3, color4, color5, color6, color7, color8, color9]
    label_lst = [label1, label2, label3, label4, label5, label6, label7, label8, label9]
    labels=[]


    for i in range(9):
        if sample_lst[i][0] != None:
            x = ica_data.A.loc[iM1, sample_lst[i]]
            y = ica_data.A.loc[iM2, sample_lst[i]]
            ax.scatter(x, y, alpha = 1, s = 75, ec = "k", color = color_lst[i], label="hi")
            labels.append(label_lst[i])
    
    label_dict = dict(zip(color_lst, labels))
    patches = []
    for color in list(label_dict.keys()):
        patches.append(mpatches.Patch(color=color, label=label_dict[color], ec= "black"))
    ax.legend(handles=patches, fontsize=10, frameon= False, loc=legend_loc)

    ax.set_xlabel(iM1, fontsize=15, fontweight="bold") 
    ax.set_ylabel(iM2, fontsize=15, fontweight="bold")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)


    plt.axhline(0, color='black', linewidth=.5)
    plt.axvline(0, color='black', linewidth=.5)

    return

###################################
# Other frequently used functions #
###################################

def list_genes_in_iM(imodulon = "RpoH"):
    '''
    Function that returns list of all genes in input iModulon.
    '''

    iM_select = [imodulon]
    bin_M = ica_data.M_binarized

    for _iM in iM_select:
        gene_id_list = []
        for gene in list(set(bin_M[_iM].loc[bin_M[_iM] == 1].index)):
            gene_id_list.append(gene)

    #Sort by weight in iM.
    gene_id_list = ica_data.M.loc[gene_id_list, "RpoH"].sort_values(ascending=False).index
    gene_list = list(ica_data.gene_table.loc[gene_id_list, "gene_name"])
    
    
    return gene_list

def df_for_filtering(iM1 = "RpoH", iM2 = "UC-9", genes_in_iM = None):
    df_full = pd.DataFrame(index=ica_data.A.columns)
    df_full["condition"] = ica_data.sample_table.condition
    df_full["experiment"] = ica_data.sample_table.project
    df_full[iM1] = ica_data.A.loc[iM1]
    df_full[iM2] = ica_data.A.loc[iM2]

    if genes_in_iM != None:
        for gene in list_genes_in_iM(genes_in_iM):
            df_full[gene] = ica_data.X.T[ica_data.gene_table[ica_data.gene_table["gene_name"] == gene].index[0]]

    df_full = df_full.sort_values(by=iM2, ascending=False)

    return df_full

def mean_df(df):
    '''Returns dataframe with mean values for every condition'''
    A_mean = pd.DataFrame(columns=df.columns.unique())
    for condition in A_mean.columns:    
        if type(df.loc[:,condition]) == pd.core.frame.DataFrame: #If there are more than one sample with the given condition,
            A_mean[condition] = df.loc[:,condition].mean(axis=1) #Get mean value. Treated as Dataframe.
        else:                                                    #If there are no more than one samples for the given condition,
            A_mean[condition] = df.loc[:,condition]              #Get mean value. Treated as Series.
    return A_mean




###################################
# Load frequently used Dataframes #
###################################


# Load heterologous gene expression data
df_gene_expression_path = path.join('data','other data','gene expression levels.csv')
df_gene_expression = pd.read_csv(df_gene_expression_path, index_col="Unnamed: 0")
df_gene_expression.tail(3)

# Create new index column with sample numbers
# Add sample number column corresponding to sample_id
sample_table_pcoli2 = ica_data.sample_table.query("project == 'pcoli2'")
# Add column to fill with sample numbers.
df_gene_expression["id"] = np.nan
for id in sample_table_pcoli2.index:
    for sample_id in df_gene_expression.index:
        if sample_table_pcoli2.loc[id, "sample_id"] == sample_id:
            df_gene_expression.loc[sample_id, "id"] = id

# sort by id
df_gene_expression = df_gene_expression.sort_values(by="id", ascending=True)
# set id as new index col
df_gene_expression.index = df_gene_expression["id"]


metadata_edit = pd.read_csv("data/other data/metadata_pcoli2_edit.csv", index_col="Unnamed: 0")
for sample in metadata_edit.index: #Remove samples that didn't pass QC from metadata.
    if sample not in ica_data.sample_names:
        metadata_edit = metadata_edit.drop(sample)
metadata_edit["counts/length / tot_counts/lengths"] = df_gene_expression["counts/length / tot_counts/lengths"]
metadata_edit["counts / tot_counts"] = df_gene_expression["counts / tot_counts"]
metadata_edit = metadata_edit.sort_values(by="OD", ascending=False)