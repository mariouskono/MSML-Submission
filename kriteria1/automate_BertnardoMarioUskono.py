import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def load_dataset(path):
    return pd.read_csv(path)


def preprocess(df):
    df['label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    df.drop('quality', axis=1, inplace=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop('label', axis=1))

    df_scaled = pd.DataFrame(X_scaled, columns=df.drop('label', axis=1).columns)
    df_scaled['label'] = df['label']

    return df_scaled


def save_dataset(df, output_path):
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = 'winequality-red.csv'
    output_path = 'winequality_preprocessed.csv'

    df_raw = load_dataset(input_path)
    df_processed = preprocess(df_raw)
    save_dataset(df_processed, output_path)
    print("Preprocessing selesai dan data disimpan.")
