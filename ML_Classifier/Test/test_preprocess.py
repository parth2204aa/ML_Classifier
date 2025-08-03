import pytest  
from preprocess import load_data, clean_data

def test_load_data_shape():
    data = load_data('data/iris.csv')
    assert data.shape[1] == 5 # 4 features + 1 label

def test_clean_data_removes_nulls():
    data = load_data('data/iris.csv')
    cleaned = clean_data(data)
    assert cleaned.isnull().sum().sum() == 0    
