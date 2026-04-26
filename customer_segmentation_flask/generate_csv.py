import pandas as pd
import numpy as np
import os  # ← Tambahkan ini!

np.random.seed(42)
n_samples = 440

data = {
    'Fresh': np.random.randint(1000, 100000, n_samples),
    'Milk': np.random.randint(100, 80000, n_samples),
    'Grocery': np.random.randint(100, 100000, n_samples),
    'Frozen': np.random.randint(100, 50000, n_samples),
    'Detergents_Paper': np.random.randint(100, 60000, n_samples),
    'Delicassen': np.random.randint(100, 30000, n_samples)
}

df = pd.DataFrame(data)
os.makedirs('data', exist_ok=True)  # ← Sekarang os sudah terdefinisi
df.to_csv('data/wholesale_customers.csv', index=False)
print("Dataset berhasil dibuat!")