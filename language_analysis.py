import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from scipy.cluster.hierarchy import dendrogram, linkage
import umap
import warnings
warnings.filterwarnings('ignore')

class LanguageStructureAnalyzer:
    def __init__(self, features_file='language_features_normalized.csv'):
        self.features_file = features_file
        self.df = None
        self.features = None
        self.languages = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.pca = None
        self.umap_reducer = None
        self.kmeans_model = None
        self.hierarchical_model = None
        
    def load_data(self):
        self.df = pd.read_csv(self.features_file)
        
        self.languages = self.df['language'].values
        features_to_exclude = ['unique_words', 'total_words']
        feature_df = self.df.drop('language', axis=1)
        allowlist = [
            'morph_entropy', 'case_entropy', 'tense_entropy', 'mood_entropy',
            'avg_sentence_length', 'std_sentence_length', 'std_word_length',
            'pos_NOUN', 'pos_VERB', 'pos_ADJ', 'pos_ADV', 'pos_PRON', 'pos_DET', 'pos_ADP',
            'noun_verb_ratio', 'adj_noun_ratio',
            'svo_ratio', 'sov_ratio', 'vso_ratio',
            'verb_final_ratio', 'verb_initial_ratio', 'subj_before_verb_ratio', 'obj_before_verb_ratio',
            'dep_left_ratio', 'dep_left_ratio_verbs', 'dep_left_ratio_nouns', 'dep_left_ratio_adp',
            'amod_left_ratio', 'nmod_left_ratio',
            'avg_dependency_length', 'std_dependency_length', 'dep_len_p50', 'dep_len_p90', 'long_dependency_ratio',
            'word_order_diversity', 'dependency_diversity',
            'word_ratio'
        ]
        keep_cols = [c for c in allowlist if c in feature_df.columns]
        if keep_cols:
            feature_df = feature_df[keep_cols]
            pass
        
        for feat in features_to_exclude:
            if feat in feature_df.columns:
                feature_df = feature_df.drop(feat, axis=1)
                pass
        
        selector = VarianceThreshold(threshold=0.01)
        feature_array = selector.fit_transform(feature_df.values)
        selected_features = feature_df.columns[selector.get_support()].tolist()
        
        feature_df_filtered = pd.DataFrame(feature_array, columns=selected_features)
        corr_matrix = feature_df_filtered.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = []
        for col in upper_tri.columns:
            high_corr = upper_tri.index[upper_tri[col] > 0.95].tolist()
            if high_corr:
                high_corr_pairs.extend([(col, feat) for feat in high_corr])
        
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            if feat1 in feature_df_filtered.columns and feat2 in feature_df_filtered.columns:
                var1 = feature_df_filtered[feat1].var()
                var2 = feature_df_filtered[feat2].var()
                if var1 < var2:
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        final_features = [f for f in selected_features if f not in features_to_remove]
        self.features = feature_df_filtered[final_features].values
        self.feature_names = final_features
        return self
    
    
    def find_optimal_clusters(self, max_k=8):
        silhouette_scores = []
        k_range = range(3, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.features)
            score = silhouette_score(self.features, labels)
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        return optimal_k
    
    def perform_clustering(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters()
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        kmeans_labels = self.kmeans_model.fit_predict(self.features)
        
        self.hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        hierarchical_labels = self.hierarchical_model.fit_predict(self.features)
        
        kmeans_silhouette = silhouette_score(self.features, kmeans_labels)
        hierarchical_silhouette = silhouette_score(self.features, hierarchical_labels)
        
        clustering_results = pd.DataFrame({
            'Language': self.languages,
            'KMeans_Cluster': kmeans_labels,
            'Hierarchical_Cluster': hierarchical_labels
        })
        
        return clustering_results, kmeans_labels, hierarchical_labels

    def perform_additional_clustering(self, n_clusters):
        results = {}
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        gmm_labels = gmm.fit_predict(self.features)
        results['GMM_Cluster'] = gmm_labels
        eps = 0.8
        min_samples = max(3, int(0.1 * len(self.features)))
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        db_labels = db.fit_predict(self.features)
        results['DBSCAN_Cluster'] = db_labels
        return results
    
    def perform_dimensionality_reduction(self):
        pca_full = PCA()
        pca_full.fit(self.features)
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        n_components_90 = np.argmax(cumsum_variance >= 0.90) + 1
        
        self.pca = PCA(n_components=2)
        pca_features = self.pca.fit_transform(self.features)
        
        self.umap_reducer = umap.UMAP(
            n_components=2, 
            random_state=42,
            n_neighbors=min(8, len(self.languages) - 1),
            min_dist=0.05,
            metric='cosine',
            spread=1.2
        )
        umap_features = self.umap_reducer.fit_transform(self.features)
        
        return pca_features, umap_features
    
    def visualize_language_similarity(self, pca_features, umap_features, clustering_results):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        n_clusters = len(set(clustering_results['KMeans_Cluster']))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for cluster_id in range(n_clusters):
            mask = clustering_results['KMeans_Cluster'] == cluster_id
            axes[0, 0].scatter(pca_features[mask, 0], pca_features[mask, 1], 
                              c=[colors[cluster_id]], s=150, alpha=0.7, 
                              label=f'Cluster {cluster_id}', edgecolors='black', linewidth=1)
        
        for i, lang in enumerate(self.languages):
            axes[0, 0].annotate(lang, (pca_features[i, 0], pca_features[i, 1]), 
                              xytext=(3, 3), textcoords='offset points', fontsize=7)
        axes[0, 0].set_title('PCA: Language Similarity (K-means clusters)', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8, loc='best')
        
        for cluster_id in range(n_clusters):
            mask = clustering_results['KMeans_Cluster'] == cluster_id
            axes[0, 1].scatter(umap_features[mask, 0], umap_features[mask, 1], 
                              c=[colors[cluster_id]], s=150, alpha=0.7, 
                              label=f'Cluster {cluster_id}', edgecolors='black', linewidth=1)
        
        for i, lang in enumerate(self.languages):
            axes[0, 1].annotate(lang, (umap_features[i, 0], umap_features[i, 1]), 
                              xytext=(3, 3), textcoords='offset points', fontsize=7)
        axes[0, 1].set_title('UMAP: Language Similarity (K-means clusters)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('UMAP 1')
        axes[0, 1].set_ylabel('UMAP 2')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(fontsize=8, loc='best')
        
        n_hier_clusters = len(set(clustering_results['Hierarchical_Cluster']))
        hier_colors = plt.cm.Set3(np.linspace(0, 1, n_hier_clusters))
        
        for cluster_id in range(n_hier_clusters):
            mask = clustering_results['Hierarchical_Cluster'] == cluster_id
            axes[1, 0].scatter(pca_features[mask, 0], pca_features[mask, 1], 
                              c=[hier_colors[cluster_id]], s=150, alpha=0.7, 
                              label=f'Cluster {cluster_id}', edgecolors='black', linewidth=1)
        
        for i, lang in enumerate(self.languages):
            axes[1, 0].annotate(lang, (pca_features[i, 0], pca_features[i, 1]), 
                              xytext=(3, 3), textcoords='offset points', fontsize=7)
        axes[1, 0].set_title('PCA: Language Similarity (Hierarchical clusters)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(fontsize=8, loc='best')
        
        for cluster_id in range(n_hier_clusters):
            mask = clustering_results['Hierarchical_Cluster'] == cluster_id
            axes[1, 1].scatter(umap_features[mask, 0], umap_features[mask, 1], 
                              c=[hier_colors[cluster_id]], s=150, alpha=0.7, 
                              label=f'Cluster {cluster_id}', edgecolors='black', linewidth=1)
        
        for i, lang in enumerate(self.languages):
            axes[1, 1].annotate(lang, (umap_features[i, 0], umap_features[i, 1]), 
                              xytext=(3, 3), textcoords='offset points', fontsize=7)
        axes[1, 1].set_title('UMAP: Language Similarity (Hierarchical clusters)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('UMAP 1')
        axes[1, 1].set_ylabel('UMAP 2')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=8, loc='best')
        
        plt.tight_layout()
        plt.savefig('language_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        print("Visualization saved to language_analysis_comprehensive.png")
        plt.close()
        
        return None
    
    
    def run_complete_analysis(self, n_clusters=None):
        print("Implementing the complete proposal pipeline...")
        
        self.load_data()
        clustering_results, kmeans_labels, hierarchical_labels = self.perform_clustering(n_clusters=n_clusters)
        extra_clusters = self.perform_additional_clustering(n_clusters if n_clusters else len(set(kmeans_labels)))
        for name, labels in extra_clusters.items():
            clustering_results[name] = labels
        pca_features, umap_features = self.perform_dimensionality_reduction()
        self.visualize_language_similarity(pca_features, umap_features, clustering_results)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Final feature set ({len(self.feature_names)} features):")
        print(", ".join(self.feature_names))
        
        return {
            'clustering_results': clustering_results,
            'pca_features': pca_features,
            'umap_features': umap_features,
            'feature_names': self.feature_names
        }


if __name__ == "__main__":
    import sys
    analyzer = LanguageStructureAnalyzer()
    
    # Allow manual specification of number of clusters
    n_clusters = None
    if len(sys.argv) > 1:
        try:
            n_clusters = int(sys.argv[1])
            print(f"Using manually specified k={n_clusters}")
        except ValueError:
            print(f"Invalid cluster number: {sys.argv[1]}. Using automatic selection.")
    
    results = analyzer.run_complete_analysis(n_clusters=n_clusters)
