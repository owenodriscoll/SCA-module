# DopSCA Uncertainty Analysis Notebooks

## Overview
This collection of Jupyter notebooks analyzes and quantifies various sources of retrieval uncertainty in the Doppler Scatterometer (DopSCA) system across different case studies and uncertainty sources.

---

## Notebooks

### 0. **SCA_summary___retrieval_uncertainty.ipynb**
**Objective**: Provide a comprehensive overview of total retrieval uncertainties across all sources for the California case study specifically.

### 1. **SCA_uncertainty___monte_carlo_California.ipynb**
**Objective**: Quantify retrieval uncertainties for the California case study through Monte Carlo analysis.
- Performs 50 iterations for each case study to assess noise-driven uncertainty distributions
- Crops input data to match Mediterranean case extent for fair inter-region comparison
- Analyzes the distribution of uncertainties stemming from noise variability
- Summarizes RMSE values for key error sources

### 2. **SCA_uncertainty___monte_carlo_MED.ipynb**
**Objective**: Quantify retrieval uncertainties for two Mediterranean case studies through Monte Carlo analysis.
- Analyzes two separate SWB  runs over the Mediterranean region
- Performs 50 iterations for each case study to assess noise-driven uncertainty distributions
- Filters data to avoid regions with invalid NRCS values
- Provides per-case-study uncertainty summaries and statistical distributions

### 3. **SCA_uncertainty___residual_leakage.ipynb**
**Objective**: Characterize and quantify residual leakage errors after correction attempts.
- Estimates the incurred leakage signal
- Re-estimates leakage using SCA's own simulated NRCS to assess correction effectiveness
- Calculates net residual leakage (incurred minus estimated)

### 4. **SCA_uncertainty___IWA.ipynb**
**Objective**: Assess velocity estimation uncertainty due to inverse-wave-age (IWA) variability.
- Analyzes sensitivity of velocity retrievals to variations in wave age using lookup tables
- Evaluates velocity spread as a function of wind speed and direction
- Demonstrates how IWA ambiguity impacts wind vector retrieval accuracy
- Quantifies directional dependencies of velocity uncertainty

### 5. **SCA_uncertainty___wave_Doppler.ipynb**
**Objective**: Evaluate uncertainty in Doppler velocity estimates related to the use of a wave-Doppler GMF.
- Analyzes wave-Doppler-induced errors in the retrieval chain
- Quantifies how wave spectral properties propagate into measurement uncertainty



---


