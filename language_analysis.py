import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import umap
import warnings
warnings.filterwarnings('ignore')

class LanguageStructureAnalyzer:
    def __init__(self, features_file='language_features_normalized.csv'):
        self.features_file = features_file
        self.df = None
        self.features = None
        self.languages = None
        self.scaler = StandardScaler()
        self.pca = None
        self.umap_reducer = None
        self.kmeans_model = None
        self.hierarchical_model = None
        
    def load_data(self):
        print("Loading language features...")
        self.df = pd.read_csv(self.features_file)
        
        self.languages = self.df['language'].values
        self.features = self.df.drop('language', axis=1).values
        
        print(f"Loaded {len(self.languages)} languages with {self.features.shape[1]} features")
        print(f"Languages: {', '.join(self.languages)}")
        
        return self
    
    
    def perform_clustering(self, n_clusters=5):
        print(f"\n=== CLUSTERING ANALYSIS (k={n_clusters}) ===")
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = self.kmeans_model.fit_predict(self.features)
        
        self.hierarchical_model = AgglomerativeClustering(n_clusters=n_clusters)
        hierarchical_labels = self.hierarchical_model.fit_predict(self.features)
        
        kmeans_silhouette = silhouette_score(self.features, kmeans_labels)
        hierarchical_silhouette = silhouette_score(self.features, hierarchical_labels)
        
        print(f"K-means silhouette score: {kmeans_silhouette:.3f}")
        print(f"Hierarchical silhouette score: {hierarchical_silhouette:.3f}")
        
        clustering_results = pd.DataFrame({
            'Language': self.languages,
            'KMeans_Cluster': kmeans_labels,
            'Hierarchical_Cluster': hierarchical_labels
        })
        
        print("\nClustering Results:")
        print(clustering_results)
        
        print("\nK-means Cluster Analysis:")
        for cluster_id in range(n_clusters):
            cluster_langs = clustering_results[clustering_results['KMeans_Cluster'] == cluster_id]['Language'].tolist()
            print(f"Cluster {cluster_id}: {', '.join(cluster_langs)}")
        
        return clustering_results, kmeans_labels, hierarchical_labels
    
    def perform_dimensionality_reduction(self):
        print("\n=== DIMENSIONALITY REDUCTION ===")
        
        self.pca = PCA(n_components=2)
        pca_features = self.pca.fit_transform(self.features)
        
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Total variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        self.umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_features = self.umap_reducer.fit_transform(self.features)
        
        return pca_features, umap_features
    
    def visualize_language_similarity(self, pca_features, umap_features, clustering_results):
        print("\n=== CREATING VISUALIZATIONS ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        scatter = axes[0, 0].scatter(pca_features[:, 0], pca_features[:, 1], 
                                   c=clustering_results['KMeans_Cluster'], 
                                   cmap='tab10', s=100, alpha=0.7)
        for i, lang in enumerate(self.languages):
            axes[0, 0].annotate(lang, (pca_features[i, 0], pca_features[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 0].set_title('PCA: Language Similarity (K-means clusters)')
        axes[0, 0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        
        scatter = axes[0, 1].scatter(umap_features[:, 0], umap_features[:, 1], 
                                   c=clustering_results['KMeans_Cluster'], 
                                   cmap='tab10', s=100, alpha=0.7)
        for i, lang in enumerate(self.languages):
            axes[0, 1].annotate(lang, (umap_features[i, 0], umap_features[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_title('UMAP: Language Similarity (K-means clusters)')
        axes[0, 1].set_xlabel('UMAP 1')
        axes[0, 1].set_ylabel('UMAP 2')
        
        scatter = axes[1, 0].scatter(pca_features[:, 0], pca_features[:, 1], 
                                   c=clustering_results['Hierarchical_Cluster'], 
                                   cmap='tab10', s=100, alpha=0.7)
        for i, lang in enumerate(self.languages):
            axes[1, 0].annotate(lang, (pca_features[i, 0], pca_features[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_title('PCA: Language Similarity (Hierarchical clusters)')
        axes[1, 0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1, 0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        
        scatter = axes[1, 1].scatter(umap_features[:, 0], umap_features[:, 1], 
                                   c=clustering_results['Hierarchical_Cluster'], 
                                   cmap='tab10', s=100, alpha=0.7)
        for i, lang in enumerate(self.languages):
            axes[1, 1].annotate(lang, (umap_features[i, 0], umap_features[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_title('UMAP: Language Similarity (Hierarchical clusters)')
        axes[1, 1].set_xlabel('UMAP 1')
        axes[1, 1].set_ylabel('UMAP 2')
        
        plt.tight_layout()
        plt.savefig('language_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return None
    
    
    def run_complete_analysis(self):
        print("Implementing the complete proposal pipeline...")
        
        self.load_data()
        clustering_results, kmeans_labels, hierarchical_labels = self.perform_clustering(n_clusters=5)
        pca_features, umap_features = self.perform_dimensionality_reduction()
        self.visualize_language_similarity(pca_features, umap_features, clustering_results)
        
        print("\n=== ANALYSIS COMPLETE ===")
        
        return {
            'clustering_results': clustering_results,
            'pca_features': pca_features,
            'umap_features': umap_features
        }


if __name__ == "__main__":
    analyzer = LanguageStructureAnalyzer()
    results = analyzer.run_complete_analysis()
