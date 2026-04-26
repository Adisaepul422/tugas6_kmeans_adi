"""
MODEL K-MEANS CLUSTERING UNTUK SEGMENTASI PELANGGAN GROSIR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load dataset dan preprocessing"""
    df = pd.read_csv(filepath)
    
    # Pilih fitur untuk clustering (6 fitur pengeluaran)
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    
    # Cek apakah kolom sesuai dengan dataset
    available_features = [f for f in features if f in df.columns]
    if 'Channel' in df.columns and 'Region' in df.columns:
        metadata = df[['Channel', 'Region']].copy()
    else:
        metadata = pd.DataFrame(index=df.index)
    
    X = df[available_features].copy()
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())
    
    return X, metadata, available_features, df

def normalize_data(X):
    """Normalisasi data dengan StandardScaler"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def find_optimal_k(X_scaled, max_k=10):
    """Menentukan K optimal dengan Elbow Method"""
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Plot Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Inertia')
    plt.title('Metode Elbow untuk Menentukan K Optimal')
    plt.xticks(K_range)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/elbow_plot.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # K optimal biasanya di elbow (K=3 atau K=4)
    optimal_k = 3
    
    return optimal_k, inertias

def perform_clustering(X_scaled, n_clusters=3):
    """Melakukan clustering K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    return kmeans, clusters

def visualize_clusters_pca(X_scaled, clusters, n_clusters):
    """Visualisasi dengan PCA 2D"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i in range(n_clusters):
        mask = clusters == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'Cluster {i}', 
                   s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Visualisasi Hasil Clustering (PCA 2D) dengan K={n_clusters}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/cluster_visualization.png', dpi=100, bbox_inches='tight')
    plt.close()

def get_cluster_profiles(df, clusters, features):
    """Analisis profil setiap cluster"""
    df_result = df.copy()
    df_result['Cluster'] = clusters
    
    cluster_stats = {}
    for i in range(len(df_result['Cluster'].unique())):
        cluster_data = df_result[df_result['Cluster'] == i]
        cluster_stats[f'Cluster {i}'] = {
            'jumlah': len(cluster_data),
            'persentase': round(len(cluster_data) / len(df_result) * 100, 2),
            'Fresh': round(cluster_data['Fresh'].mean(), 2) if 'Fresh' in cluster_data.columns else 0,
            'Milk': round(cluster_data['Milk'].mean(), 2) if 'Milk' in cluster_data.columns else 0,
            'Grocery': round(cluster_data['Grocery'].mean(), 2) if 'Grocery' in cluster_data.columns else 0,
            'Frozen': round(cluster_data['Frozen'].mean(), 2) if 'Frozen' in cluster_data.columns else 0,
            'Detergents_Paper': round(cluster_data['Detergents_Paper'].mean(), 2) if 'Detergents_Paper' in cluster_data.columns else 0,
            'Delicassen': round(cluster_data['Delicassen'].mean(), 2) if 'Delicassen' in cluster_data.columns else 0
        }
    
    return df_result, cluster_stats

def run_full_clustering(filepath, n_clusters=3):
    """Menjalankan seluruh proses clustering"""
    X, metadata, features, df_original = load_and_preprocess_data(filepath)
    X_scaled, scaler = normalize_data(X)
    kmeans, clusters = perform_clustering(X_scaled, n_clusters)
    visualize_clusters_pca(X_scaled, clusters, n_clusters)
    df_result, cluster_stats = get_cluster_profiles(df_original, clusters, features)
    
    return {
        'data': df_result,
        'cluster_stats': cluster_stats,
        'optimal_k': n_clusters,
        'features': features,
        'cluster_labels': clusters
    }