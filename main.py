import argparse
import sys
import subprocess
import importlib
from pathlib import Path

def ensure_dependencies():
    deps = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("umap-learn", "umap"),
    ]
    for pkg, module_name in deps:
        try:
            importlib.import_module(module_name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure_dependencies()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
import umap

from feature_extractor import LanguageFeatureExtractor


def normalize_features(df: pd.DataFrame, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ["language"]
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()
    metadata = df[exclude_cols] if any(col in df.columns for col in exclude_cols) else None
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if not feature_cols:
        return df.copy()
    features_numeric = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features_numeric)
    df_normalized = pd.DataFrame(normalized_features, columns=feature_cols, index=df.index)
    if metadata is not None:
        for col in exclude_cols:
            if col in df.columns:
                df_normalized[col] = df[col].values
    return df_normalized


def extract_features(data_dir: str, max_tokens: int = 100000) -> pd.DataFrame:
    extractor = LanguageFeatureExtractor(max_tokens=max_tokens)
    language_mapping = {
        "english.conllu": "English",
        "spanish.conllu": "Spanish",
        "french.conllu": "French",
        "hindi.conllu": "Hindi",
        "arabic.conllu": "Arabic",
        "chinese.conllu": "Mandarin",
        "vietnamese.conllu": "Vietnamese",
        "thai.conllu": "Thai",
        "german.conllu": "German",
        "sanskrit.conllu": "Sanskrit",
        "tamil.conllu": "Tamil",
        "telugu.conllu": "Telugu",
        "marathi.conllu": "Marathi",
        "swedish.conllu": "Swedish",
        "japanese.conllu": "Japanese",
        "korean.conllu": "Korean",
        "turkish.conllu": "Turkish",
        "uzbek.conllu": "Uzbek",
        "kazakh.conllu": "Kazakh",
        "sindhi.conllu": "Sindhi",
        "malayalam.conllu": "Malayalam",
        "portugal.conllu": "Portugal",
        "finnish.conllu": "Finnish",
        "punjabi.conllu": "Punjabi",
        "indonesian.conllu": "Indonesian",
        "romanian.conllu": "Romanian",
        "russian.conllu": "Russian",
    }
    directory = Path(data_dir)
    if not directory.exists():
        return None
    df = extractor.extract_from_directory(directory, language_mapping)
    if df is None or getattr(df, "empty", True):
        return None
    df.to_csv("language_features_raw.csv", index=False)
    df_norm = normalize_features(df, exclude_cols=["language"])
    if df_norm is not None and not df_norm.empty:
        df_norm.to_csv("language_features_normalized.csv", index=False)
    return df_norm


def filter_features(df: pd.DataFrame):
    allowlist = [
        "morph_entropy",
        "case_entropy",
        "tense_entropy",
        "mood_entropy",
        "avg_sentence_length",
        "std_sentence_length",
        "std_word_length",
        "pos_NOUN",
        "pos_VERB",
        "pos_ADJ",
        "pos_ADV",
        "pos_PRON",
        "pos_DET",
        "pos_ADP",
        "noun_verb_ratio",
        "adj_noun_ratio",
        "svo_ratio",
        "vso_ratio",
        "verb_final_ratio",
        "verb_initial_ratio",
        "subj_before_verb_ratio",
        "obj_before_verb_ratio",
        "dep_left_ratio",
        "dep_left_ratio_verbs",
        "dep_left_ratio_nouns",
        "dep_left_ratio_adp",
        "amod_left_ratio",
        "avg_dependency_length",
        "std_dependency_length",
        "dep_len_p50",
        "dep_len_p90",
        "long_dependency_ratio",
        "dependency_diversity",
        "word_ratio",
    ]
    feature_df = df.drop("language", axis=1)
    feature_df = feature_df[[c for c in allowlist if c in feature_df.columns]]
    selector = VarianceThreshold(threshold=0.01)
    feature_array = selector.fit_transform(feature_df.values)
    selected_features = feature_df.columns[selector.get_support()].tolist()
    filtered = pd.DataFrame(feature_array, columns=selected_features)
    corr = filtered.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    remove = set()
    for col in upper.columns:
        high = upper.index[upper[col] > 0.95].tolist()
        for h in high:
            if col in filtered.columns and h in filtered.columns:
                var1 = filtered[col].var()
                var2 = filtered[h].var()
                remove.add(col if var1 < var2 else h)
    final_cols = [c for c in selected_features if c not in remove]
    return filtered[final_cols].values, final_cols


def find_optimal_k(features, k_min=3, k_max=8):
    scores = []
    k_range = range(k_min, k_max + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(features)
        score = silhouette_score(features, labels)
        scores.append(score)
    return k_range[np.argmax(scores)]


def run_clustering(features, languages, n_clusters=None):
    if n_clusters is None:
        n_clusters = find_optimal_k(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    km_labels = km.fit_predict(features)
    hier = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    hier_labels = hier.fit_predict(features)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    gmm_labels = gmm.fit_predict(features)
    db = DBSCAN(eps=0.8, min_samples=max(3, int(0.1 * len(features))), metric="euclidean")
    db_labels = db.fit_predict(features)
    results = pd.DataFrame(
        {
            "Language": languages,
            "KMeans_Cluster": km_labels,
            "Hierarchical_Cluster": hier_labels,
            "GMM_Cluster": gmm_labels,
            "DBSCAN_Cluster": db_labels,
        }
    )
    return results, km_labels, hier_labels


def reduce_and_plot(features, languages, clustering_results, feature_names):
    pca = PCA(n_components=2)
    pca_feat = pca.fit_transform(features)
    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=min(8, len(languages) - 1),
        min_dist=0.05,
        metric="cosine",
        spread=1.2,
    )
    umap_feat = reducer.fit_transform(features)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    n_k = len(set(clustering_results["KMeans_Cluster"]))
    colors_k = plt.cm.Set3(np.linspace(0, 1, n_k))
    for cid in range(n_k):
        mask = clustering_results["KMeans_Cluster"] == cid
        axes[0, 0].scatter(
            pca_feat[mask, 0],
            pca_feat[mask, 1],
            c=[colors_k[cid]],
            s=120,
            alpha=0.75,
            label=f"C{cid}",
            edgecolors="black",
            linewidth=1,
        )
    for i, lang in enumerate(languages):
        axes[0, 0].annotate(lang, (pca_feat[i, 0], pca_feat[i, 1]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[0, 0].set_title("PCA (K-means)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")

    for cid in range(n_k):
        mask = clustering_results["KMeans_Cluster"] == cid
        axes[0, 1].scatter(
            umap_feat[mask, 0],
            umap_feat[mask, 1],
            c=[colors_k[cid]],
            s=120,
            alpha=0.75,
            label=f"C{cid}",
            edgecolors="black",
            linewidth=1,
        )
    for i, lang in enumerate(languages):
        axes[0, 1].annotate(lang, (umap_feat[i, 0], umap_feat[i, 1]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[0, 1].set_title("UMAP (K-means)")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8, loc="best")

    n_h = len(set(clustering_results["Hierarchical_Cluster"]))
    colors_h = plt.cm.Set3(np.linspace(0, 1, n_h))
    for cid in range(n_h):
        mask = clustering_results["Hierarchical_Cluster"] == cid
        axes[1, 0].scatter(
            pca_feat[mask, 0],
            pca_feat[mask, 1],
            c=[colors_h[cid]],
            s=120,
            alpha=0.75,
            label=f"C{cid}",
            edgecolors="black",
            linewidth=1,
        )
    for i, lang in enumerate(languages):
        axes[1, 0].annotate(lang, (pca_feat[i, 0], pca_feat[i, 1]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1, 0].set_title("PCA (Hierarchical)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8, loc="best")

    for cid in range(n_h):
        mask = clustering_results["Hierarchical_Cluster"] == cid
        axes[1, 1].scatter(
            umap_feat[mask, 0],
            umap_feat[mask, 1],
            c=[colors_h[cid]],
            s=120,
            alpha=0.75,
            label=f"C{cid}",
            edgecolors="black",
            linewidth=1,
        )
    for i, lang in enumerate(languages):
        axes[1, 1].annotate(lang, (umap_feat[i, 0], umap_feat[i, 1]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[1, 1].set_title("UMAP (Hierarchical)")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig("language_analysis_comprehensive.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pca_feat, umap_feat, pca.explained_variance_ratio_


def main():
    parser = argparse.ArgumentParser(description="Run extraction and clustering pipeline.")
    parser.add_argument("--data_dir", default="data", help="Path to directory with .conllu files")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters (optional, else auto)")
    args = parser.parse_args()

    df_norm = extract_features(args.data_dir)
    if df_norm is None or df_norm.empty:
        print("No features extracted. Ensure data directory contains .conllu files.")
        sys.exit(1)

    features, feature_names = filter_features(df_norm)
    languages = df_norm["language"].values
    clustering_results, km_labels, hier_labels = run_clustering(features, languages, n_clusters=args.k)
    pca_feat, umap_feat, pca_var = reduce_and_plot(features, languages, clustering_results, feature_names)

    clustering_results.to_csv("language_clusters.csv", index=False)
    print("Done. Outputs:")
    print("- language_features_raw.csv")
    print("- language_features_normalized.csv")
    print("- language_clusters.csv")
    print("- language_analysis_comprehensive.png")


if __name__ == "__main__":
    main()

