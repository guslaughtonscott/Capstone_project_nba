# Capstone_project_nba
# Modelling NBA Position(less) Basketball

This project applies advanced NBA statistics and machine learning to identify **positionless players** — athletes whose on-court profiles do not fit cleanly into traditional Guard, Wing, or Big roles. Using stacked ensemble models and entropy-based analysis, the project aims to quantify positional ambiguity and explore its relationship with team success.


## Project Overview

Traditional basketball positions are becoming increasingly blurred in the modern NBA. Many elite players operate across multiple roles, contributing to a more fluid, positionless style of play. This project investigates whether machine learning models can:

- Accurately classify NBA player positions
- Quantify positional ambiguity using prediction entropy
- Identify positionless players
- Examine whether positionless players are disproportionately associated with winning teams

The core hypothesis is that **positionless players provide a competitive advantage**, and that this can be detected through model uncertainty.


## Data Sources

Player data was collected using a combination of:

- **NBA API**
- **Web scraping from Basketball Reference**

The dataset includes multiple seasons of NBA player statistics and advanced metrics, covering performance, usage, and role indicators.

Target variable:
- **Position** (Guard, Wing, Big)


## Methodology

The project follows a full end-to-end machine learning pipeline:

1. Data cleaning and validation  
2. Feature engineering using advanced basketball metrics    
3. Model training with cross-validation  
4. Hyperparameter optimisation using Optuna  
5. Ensemble modelling via stacking  
6. Entropy-based analysis to quantify positional ambiguity  

A key methodological contribution is the use of **Shannon entropy** on class probability outputs to measure how positionally ambiguous each player is.


## Models

### Base Learners (Level-0 Models)
- Logistic Regression  
- Support Vector Machine (SVM)  
- LightGBM  

### Meta-Learner (Level-1 Model)
- Logistic Regression

The base learners generate out-of-fold predictions which are then used as inputs to the meta-learner in a **stacked ensemble** architecture.

---

## Positionless Player Identification

For each player, the model outputs probabilities for:

- Guard  
- Wing  
- Big  

Shannon entropy is calculated over these probabilities:

\[
H = -\sum p_i \log(p_i)
\]

High entropy indicates:
> **High positional ambiguity → positionless profile**

Low entropy indicates:
> **Clear positional identity**

This allows positionlessness to be quantified rather than subjectively defined.

---

## Results

- **Final model accuracy:** 78.8%  
- **Key finding:** ~50% of identified positionless players were NBA champions  
- **Context:** This is significantly above the ~12% league-average championship probability

This supports the hypothesis that positionless players are overrepresented among championship-winning teams. Less notable players at top of the entropy list
can be used for uncovering potential


## Custom Python Library: `gus_nba_tools`

As part of this project, a custom public Python package was developed: **`gus_nba_tools`**

This library contains reusable tools for:

- Feature engineering
- Metric calculation
- Data cleaning
- Position analysis utilities

The library is published and versioned independently, and is used throughout the modelling pipeline to ensure clean, modular, and reproducible code.

This reflects a production-style workflow, separating reusable logic from analysis notebooks.


