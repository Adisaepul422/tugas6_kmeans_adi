from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

app = Flask(__name__)
app.secret_key = 'kunci-rahasia-2024'

# Pastikan folder static dan templates ada
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)  # Penting untuk Railway!

# Path file CSV
def find_csv_file():
    # Coba beberapa kemungkinan path di Railway
    possible_paths = [
        'wholesale_customers.csv',  # Langsung di root
        'data/wholesale_customers.csv',
        '/app/wholesale_customers.csv',
        'Wholesale_customers_data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_data():
    csv_path = find_csv_file()
    if csv_path:
        df = pd.read_csv(csv_path)
        return df
    else:
        # Buat data dummy jika file tidak ditemukan
        print("WARNING: CSV file not found, creating dummy data")
        np.random.seed(42)
        n_samples = 100
        data = {
            'Fresh': np.random.randint(1000, 60000, n_samples),
            'Milk': np.random.randint(1000, 50000, n_samples),
            'Grocery': np.random.randint(1000, 100000, n_samples),
            'Frozen': np.random.randint(1000, 40000, n_samples),
            'Detergents_Paper': np.random.randint(1000, 50000, n_samples),
            'Delicassen': np.random.randint(1000, 20000, n_samples)
        }
        return pd.DataFrame(data)

@app.route('/')
def index():
    df = load_data()
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    available_features = [f for f in features if f in df.columns]
    
    return render_template('index.html', 
                         features=available_features,
                         total_data=len(df))

@app.route('/cluster/<int:k>')
def cluster(k):
    df = load_data()
    
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].copy()
    X = X.fillna(X.median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    cluster_stats = {}
    df['Cluster'] = clusters
    for i in range(k):
        cluster_data = df[df['Cluster'] == i]
        cluster_stats[f'Cluster {i}'] = {
            'jumlah': len(cluster_data),
            'persentase': round(len(cluster_data) / len(df) * 100, 2),
            'Fresh': round(cluster_data['Fresh'].mean(), 2) if 'Fresh' in cluster_data.columns else 0,
            'Milk': round(cluster_data['Milk'].mean(), 2) if 'Milk' in cluster_data.columns else 0,
            'Grocery': round(cluster_data['Grocery'].mean(), 2) if 'Grocery' in cluster_data.columns else 0,
            'Frozen': round(cluster_data['Frozen'].mean(), 2) if 'Frozen' in cluster_data.columns else 0,
            'Detergents_Paper': round(cluster_data['Detergents_Paper'].mean(), 2) if 'Detergents_Paper' in cluster_data.columns else 0,
            'Delicassen': round(cluster_data['Delicassen'].mean(), 2) if 'Delicassen' in cluster_data.columns else 0
        }
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for i in range(k):
        mask = clusters == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'Cluster {i}', s=50, alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Visualisasi Clustering (K={k})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/cluster_result.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    result_data = df.head(50).to_dict('records')
    
    return render_template('result.html', 
                         clusters=cluster_stats,
                         n_clusters=k,
                         total_data=len(df),
                         result_data=result_data,
                         features=available_features)

# Untuk running di Railway
import os

if __name__ == '__main__':
    # Railway akan mengisi variabel PORT secara otomatis
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)