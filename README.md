# Vasculature_neighborhoods
This repository contains code developed for the analysis of endothelial cell-specific neighborhoods from spatial-omics datasets. The purpose of this project is to identify, analyze, and visualize the unique neighborhoods of endothelial cells within tissue samples using cutting-edge spatial-omics technologies such as CODEX, MERFISH, and others.

[20250820_SingleCell_NeighborhoodAnalysis_Endothelial_HuBMAP.ipynb](https://github.com/HickeyLab/Vasculature_neighborhoods/blob/jj413/20250820_SingleCell_NeighborhoodAnalysis_Endothelial_HuBMAP.ipynb) contains code for generating endothelial cell-specific neighborhoods using spatially-resolved single-cell data from the HuBMAP human intestine dataset. The notebook includes preprocessing steps, spatial clustering, and visualization tools such as catplots, area plots, and heatmaps to explore neighborhood composition and variation across intestinal regions. The analysis highlights spatial relationships between endothelial cells and surrounding immune or stromal populations, enabling insight into tissue-specific vascular architecture and potential functional niches.

The notebook is organized into three main sections:

1. **Endothelial Cell–Specific Neighborhoods**  
   Define neighborhoods centered on endothelial cells and explore their cellular composition and diversity.

2. **Rings Analysis**  
   Examine how cellular composition changes radially outward from endothelial cells using concentric distance-based bins.

3. **Donor-Specific Rings Analysis**  
   Stratify data by donors to analyze endothelial cell neighborhood enrichment and highlight variability across samples.
   
Each section includes basic preprocessing, neighborhood or ring construction, and visualization steps to summarize spatial relationships.
