import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def preprocess_data():
    """Load raw data, preprocess, and save train/test sets."""
    df = pd.read_csv('sustainability_data.csv')
    
    # Feature engineering
    df['materials_str'] = df['materials'].apply(lambda x: ','.join(eval(x)))  # Convert string to list first
    
    # Split features/target
    X = df.drop(['sustainability_score', 'materials'], axis=1)
    y = df['sustainability_score']
    
    # Define column types
    categorical_cols = ['root_category', 'main_category', 'materials_str', 'condition']
    boolean_cols = ['recyclability', 'ethical_sourcing', 'cruelity_free', 'plastic_free', 'natural', 'destructable', 'hazardous']
    numeric_cols = ['recycled_content', 'carbon_footprint', 'energy_efficiency', 'durability', 'repairability_score']
    
    # Preprocessing pipelines
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    bool_pipeline = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(dtype=int)),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipeline, categorical_cols),
            ('num', num_pipeline, numeric_cols),
            ('bool', bool_pipeline, boolean_cols)
        ]
    )

    # Save preprocessing pipeline
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    # Split and save data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pd.concat([X_train, y_train], axis=1).to_csv('train_data.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('test_data.csv', index=False)
    print("Preprocessing complete. Saved train_data.csv, test_data.csv, and preprocessor.pkl.")

if __name__ == "__main__":
    preprocess_data()
