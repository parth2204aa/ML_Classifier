import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def clean_path(df):
    return df.dropna()    

def split_data(df):
    x = df.drop('species', axis = 1)    
    y = df['species']
    return train_test_split(x, y, test_size = 0.2, random_state = 42)