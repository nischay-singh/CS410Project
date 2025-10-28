# CS 410 Language Structure Analysis

A comprehensive system for analyzing structural patterns across languages using machine learning, implementing the complete CS 410 proposal.

## Quick Start

### 1. Extract Features

```bash
python main.py
```

### 2. Run Complete Analysis

```bash
python language_analysis.py
```

## Project Structure

### Core Files

- `main.py` - Feature extraction from CoNLL-U files
- `language_analysis.py` - Complete clustering and visualization analysis
- `conllu_parser.py` - CoNLL-U format parser
- `feature_extractor.py` - Main feature extraction coordinator
- `linguistic_features.py` - Advanced linguistic feature extractor

### Data

- `data/` - CoNLL-U files for 20 languages
- `language_features_raw.csv` - Extracted features
- `language_features_normalized.csv` - Normalized features

### Output

- `feature_importance.png` - Feature importance analysis
- `language_analysis_comprehensive.png` - Complete visualization suite

## Features

### Linguistic Features (42 total)

- **Syntactic**: POS distributions, word order patterns, dependency structures
- **Morphological**: Word length, morphological complexity, inflection patterns
- **Lexical**: Type-token ratios, vocabulary richness, word frequency
- **Structural**: Sentence length, punctuation patterns, character ratios

### Analysis Components

- **Clustering**: K-means and hierarchical clustering
- **Dimensionality Reduction**: PCA and UMAP
- **Visualization**: Language similarity maps and feature analysis
- **Classification**: Language identification model

## Results

The system successfully clusters 20 languages into meaningful groups based on structural similarities, revealing patterns that both support and challenge traditional linguistic classifications.

## Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn
```
