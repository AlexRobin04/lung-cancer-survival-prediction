import pandas as pd

path = "/home/yuehailin/dazewen/ViLa-MIL/datasets_csv/ESCA.csv"
df = pd.read_csv(path)
print("Loaded CSV:", path)
print("label unique values:", df['label'].unique())
