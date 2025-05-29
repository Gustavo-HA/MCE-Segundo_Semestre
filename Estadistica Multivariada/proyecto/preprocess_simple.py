import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder, # Use OrdinalEncoder for features instead of LabelEncoder
    OneHotEncoder,
    FunctionTransformer,
    RobustScaler
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import click
import os
import pickle
from pathlib import Path


CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "poutcome",
    "contact",
    "default",
    "housing",
    "loan",
    "month"
]

dict_binary = {
    "yes": 1,
    "no": 0
}

NUMERIC_COLS = ['age', 'balance', 'day',"duration", 'campaign', 'pdays', 'previous']


@click.command()
@click.argument('file_path', default="./data/bank-full.csv", type=click.Path(exists=True))
@click.argument('output_path', default="./data", type=click.Path())
def preprocess(file_path : str, output_path : str):
    file_path = Path(file_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure output path exists

    df = pd.read_csv(file_path, sep=";")

    X = df.drop(columns=["y"])
    y = df["y"].map(dict_binary) # Preprocess target variable

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Transformer for one-hot encoded columns
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore',
                                    drop="first")

    
    feature_encoder_transformer = ColumnTransformer(
        transformers=[
            ('onehot', one_hot_encoder, CATEGORICAL_COLS),
            ('numeric_passthrough', 'passthrough', NUMERIC_COLS) # Solo las pasa sin modificar
        ],
        remainder='drop'
    )
    
    try:
        feature_encoder_transformer.set_output(transform="pandas")
        print("Preprocessor configured to output pandas DataFrames.")
    except AttributeError:
        print("Warning: `preprocessor.set_output(transform='pandas')` is not available. "
            "The StandardScaler warning might persist if input types vary.")

    # El pipeline completo incluiría el escalador DESPUÉS del ColumnTransformer
    full_pipeline = Pipeline(steps=[
        ('feature_processing', feature_encoder_transformer),
        ('scaler', StandardScaler()) # Ahora el scaler se aplica a la salida de feature_processing
    ])

    # Fit the pipeline on the training data
    full_pipeline.fit(X_train)

    # Save the entire pipeline object
    with open(output_path / "preprocessing_pipeline.pkl", "wb") as f:
        pickle.dump(full_pipeline, f)

    # Transform training and testing data
    # Using set_output for pandas DataFrame output (requires scikit-learn >= 1.2)
    full_pipeline.set_output(transform="pandas")
    X_train_processed = full_pipeline.transform(X_train)
    X_test_processed = full_pipeline.transform(X_test)


    # Prepare y DataFrames (ensuring index compatibility for concat)
    y_train_df = pd.DataFrame(y_train, index=X_train_processed.index)
    y_test_df = pd.DataFrame(y_test, index=X_test_processed.index)
    if y_train_df.columns[0] != 'y': # If y_train was a Series, .name might be None
        y_train_df.columns = ['y']
        y_test_df.columns = ['y']


    # Concatenate features and target
    train_data = pd.concat([X_train_processed, y_train_df], axis=1)
    test_data = pd.concat([X_test_processed, y_test_df], axis=1)

    train_data.to_csv(output_path / "train.csv", index=False)
    test_data.to_csv(output_path / "test.csv", index=False)

    print("Preprocessing complete. Processed files saved to:", output_path)

if __name__ == "__main__":
    preprocess()