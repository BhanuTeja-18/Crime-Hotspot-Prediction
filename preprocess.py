import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    df['hour'] = df['time']
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 20 else 0)

    X = df[['latitude', 'longitude', 'hour', 'is_night']]
    return X