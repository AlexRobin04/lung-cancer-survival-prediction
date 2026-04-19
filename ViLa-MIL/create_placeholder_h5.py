import h5py
import numpy as np
import os
import pandas as pd

# Create directories if they don't exist
os.makedirs('features/20', exist_ok=True)
os.makedirs('features/10', exist_ok=True)

# Read slide IDs from splits_0.csv
csv_path = 'splits/TCGA_LUSC/splits_0.csv'
df = pd.read_csv(csv_path)

# Collect all slide IDs from train, val, and test columns
slide_ids = []
for col in ['train', 'val', 'test']:
    slide_ids.extend(df[col].dropna().tolist())

# Remove duplicates
slide_ids = list(set(slide_ids))

print(f'Found {len(slide_ids)} unique slide IDs')

# Create placeholder h5 files for each slide ID
for i, slide_id in enumerate(slide_ids):
    if isinstance(slide_id, str):
        # Remove .svs extension
        base_name = slide_id[:-4]
        
        # Create 20x features file
        h5_path_20 = os.path.join('features/20', f'{base_name}.h5')
        with h5py.File(h5_path_20, 'w') as f:
            f.create_dataset('features', data=np.random.rand(10, 512))
            f.create_dataset('coords', data=np.random.rand(10, 2))
        
        # Create 10x features file
        h5_path_10 = os.path.join('features/10', f'{base_name}.h5')
        with h5py.File(h5_path_10, 'w') as f:
            f.create_dataset('features', data=np.random.rand(10, 512))
            f.create_dataset('coords', data=np.random.rand(10, 2))
        
        if (i + 1) % 10 == 0:
            print(f'Created placeholder h5 files for {i + 1}/{len(slide_ids)} slide IDs')

print('All placeholder h5 files created successfully!')
