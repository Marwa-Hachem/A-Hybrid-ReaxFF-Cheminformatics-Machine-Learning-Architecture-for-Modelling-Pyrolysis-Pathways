# A-Machine-Learning-Architecture-for-Modelling-Pyrolysis-Pathways
Hybrid ReaxFF–cheminformatics–machine learning workflow that converts reactive MD event logs into labeled pyrolysis mechanisms, reaction pathways, and Arrhenius parameters. Includes code, labelled datasets, and sample ReaxFF data for cyclohexanone and vegetable glycerin.

Step-by-step protocol: https://www.protocols.io/private/BED06CE0C31411F1AC5C0A58A9FEAC02 

# Scripts
1. Reaction Validation

This script is the first step of the workflow. It processes the raw reaction outputs generated from the ReaxFF/ChemTraYzer simulations and filters out chemically invalid reactions. The validation stage ensures that molecules can be correctly parsed, that valence and bonding are chemically reasonable, and that malformed reactions are removed before further analysis.

2. Create IDs

After validation, this script generates unique identifiers for molecular species and reaction events. Each species is assigned an ID based on its canonical molecular representation and atom indices information, allowing identical molecules originating from different atom sets to be distinguished. 

3. Track based on IDs

This script reconstructs reaction pathways and tracks the evolution of species throughout the simulation using the previously generated IDs. By following reactant–product relationships over time, it identifies dominant pathways, intermediates, branching events, and product formation routes. 

4. Statistics of pathways

This script performs statistical analysis on the reconstructed and labeled reaction pathways. It calculates quantities such as product frequencies, pathway abundances, reaction-class distributions, dominant intermediates, and other descriptive statistics. 

5. Feature engineering

This script converts the validated reactions into machine-learning-ready numerical representations. It computes molecular fingerprints such as Morgan and BRICS fingerprints for reactants and products, along with engineered scalar descriptors related to bond changes, ring transformations, radicals, aromaticity, oxygen environments, and other chemical properties. The script also organizes labels and saves the processed features into structured arrays and .npz files that are later used for machine-learning training and prediction.

6. ML – PCA & CV

This script constitutes the main machine-learning training and evaluation framework. It loads the engineered features, performs preprocessing and scaling, applies PCA dimensionality reduction to the fingerprint space, and trains multiple classification models including Linear SVC, Random Forest, Extra Trees, XGBoost, MLP, and Naive Bayes. Stratified cross-validation is then used to evaluate model performance using metrics such as accuracy, macro precision, macro recall, and macro F1-score. The script also computes 95% confidence intervals and automatically identifies the best-performing model based on cross-validation performance. 

7. Label from best model in CV

This script uses the best model selected from the cross-validation framework to classify reactions in an external or unseen dataset. The selected model is retrained using all labeled training data, after which the same preprocessing steps, scaling transformations, and PCA projections are applied to the external dataset. The model then predicts mechanistic transformation labels for unseen reactions and saves the resulting labeled dataset and evaluation metrics. 

8. Kinetics

This script performs kinetic analysis on validated reaction data from the isothermal ReaxFF simulations: computes pseudo-first-order rate constants from ln N(t) versus time, averages replicates at each temperature, and extracts the activation energy and pre-exponential factor from an Arrhenius fit.

# Requirements
Python 3.13
RDKit 2024.9.5
pandas 2.2.3
NetworkX 3.4.2
scikit-learn 1.6.1
XGBoost 3.1.1

# Data availability

This repository contains the analysis code. The complete ReaxFF trajectories, reaction event logs, and datasets will be deposited on Zenodo upon publication.

# Citation

If you use this code or data, please cite the article above (citation will be updated on publication).
