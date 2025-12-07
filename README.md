# CS 410 Language Structure Analysis

Pipeline:

- Extract features from `data/*.conllu`:
  - `python main.py`
- Run clustering + visuals (auto k or set k):
  - `python language_analysis.py`
  - `python language_analysis.py 6`

New features: morphological entropy, dependency directionality, verb-order ratios.
Additional clustering: GMM and DBSCAN (reported alongside k-means/hierarchical).

Outputs: `language_features_raw.csv`, `language_features_normalized.csv`, `language_analysis_comprehensive.png`

Dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn umap-learn`
