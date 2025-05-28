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

# Define column groups (consider making these more dynamic or config-driven if needed)
LABEL_ENCODER_COLS = [ # Renamed to reflect OrdinalEncoder usage for features
    "month"
]

BINARY_COLS = [
    "default",
    "housing",
    "loan"
]

CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "poutcome",
    "contact"
]

NUMERIC_COLS = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


dict_binary = {
    "yes": 1,
    "no": 0
}

# Helper function for FunctionTransformer if mapping is preferred for binary
def map_binary_columns(df_slice: pd.DataFrame) -> pd.DataFrame:
    return df_slice.replace(dict_binary).infer_objects(copy=False)


# Cyclical encoding for 'month' column
month_to_int_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
MAX_MONTH_VALUE = 12
class CyclicalFeatureEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col_map, max_val):
        self.col_map = col_map
        self.max_val = max_val
        self.feature_names_ = [] # To store output feature names based on input

    def fit(self, X, y=None):
        # X is expected to be a DataFrame with one column (the feature to encode)
        # Store the input feature name to generate output feature names
        self.input_feature_name_ = X.columns[0]
        self.feature_names_ = [f'{self.input_feature_name_}_sin', f'{self.input_feature_name_}_cos']
        return self

    def transform(self, X, y=None):
        # X is expected to be a DataFrame with one column
        X_copy = X.copy()
        col_to_transform = X_copy.columns[0]

        # Step 1: Map string month to integer (handle potential case issues)
        try:
            month_numeric = X_copy[col_to_transform].str.lower().map(self.col_map)
        except AttributeError: # If data is already numeric (e.g., during cross-val with already transformed numbers)
            month_numeric = X_copy[col_to_transform]

        # Handle potential NaNs if a month string isn't in the map or input was NaN
        # Fill with a value that results in (0,0) or another neutral sin/cos pair if appropriate,
        # or ensure your month_map is exhaustive and data is clean.
        # For simplicity, if month_numeric contains NaN, sin/cos will also be NaN,
        # which might need imputation later if not desired. Or fillna here:
        month_numeric = month_numeric.fillna(self.max_val / 4) # e.g., yields sin=1, cos=0 for month 3 if max_val is 12

        # Step 2: Apply sin/cos transformation
        sin_vals = np.sin(2 * np.pi * month_numeric / self.max_val)
        cos_vals = np.cos(2 * np.pi * month_numeric / self.max_val)

        out_df = pd.DataFrame({
            self.feature_names_[0]: sin_vals,
            self.feature_names_[1]: cos_vals
        }, index=X_copy.index) # Preserve index

        return out_df

    def get_feature_names_out(self, input_features=None):
        # If called after fit, self.feature_names_ should be set.
        # input_features is a list of input column names (should be one for this transformer)
        if hasattr(self, 'feature_names_') and self.feature_names_:
            return self.feature_names_
        elif input_features: # Fallback if fit wasn't called but names are needed
             return [f'{input_features[0]}_sin', f'{input_features[0]}_cos']
        else:
            return [] # Should not happen in a normal pipeline flow


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

    # Transformer for binary columns (map to 0/1)
    binary_transformer = FunctionTransformer(map_binary_columns)

    # Transformer for label-encoded columns (using OrdinalEncoder for features)
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # Transformer for one-hot encoded columns
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    
    month_cyclical_transformer = CyclicalFeatureEncoder(col_map=month_to_int_map, max_val=MAX_MONTH_VALUE)

    # Preprocessing for features
    # Note: StandardScaler will be applied to the output of all transformers below.
    # If you want to scale only specific columns (e.g., only original numerics and ordinals),
    # you would include StandardScaler within those specific transformer pipelines.
    # The current script scales *everything* after all encodings.
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', binary_transformer, BINARY_COLS),
            ('ordinal', month_cyclical_transformer, LABEL_ENCODER_COLS),
            ('onehot', one_hot_encoder, CATEGORICAL_COLS),
            ('numeric', RobustScaler(), NUMERIC_COLS) # Numeric columns are passed through initially
        ],
        remainder='drop'  # Or 'passthrough' if you have other columns to keep and scale
    )

    try:
        preprocessor.set_output(transform="pandas")
        print("Preprocessor configured to output pandas DataFrames.")
    except AttributeError:
        print("Warning: `preprocessor.set_output(transform='pandas')` is not available. "
            "The StandardScaler warning might persist if input types vary.")
    
    # Create the full pipeline: Preprocessing + Scaling of all processed features
    # The output of 'preprocessor' will be entirely numeric.
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
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