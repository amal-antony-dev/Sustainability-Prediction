import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns

# Step 1: Generate synthetic dataset
def generate_dataset(n_samples=1000):
    """
    Generate a synthetic dataset based on the Product model attributes.
    
    Args:
        n_samples: Number of product samples to generate
    
    Returns:
        DataFrame with product features
    """
    # Define the choices for categorical features
    root_categories = ["Household", "Electronics", "Fashion", "Food", "Furniture"]
    main_categories = ["Kitchen", "Bathroom", "Office", "Bedroom", "Garden", "Mobiles", "Appliances"]
    materials = ["wood", "metal", "plastic", "glass", "paper", "fabric", "ceramic", 
                "rubber", "leather", "stone", "bamboo", "cotton", "silk", "wool", 
                "linen", "hemp", "jute", "coir", "sisal", "cork"]
    conditions = ["New", "Used - Like New", "Used - Good", "Used - Fair", "For Parts or Not Working"]
    
    # Generate random data
    data = {
        'root_category': [random.choice(root_categories) for _ in range(n_samples)],
        'main_category': [random.choice(main_categories) for _ in range(n_samples)],
        # Generate 1-3 materials per product
        'materials': [random.sample(materials, random.randint(1, 3)) for _ in range(n_samples)],
        'condition': [random.choice(conditions) for _ in range(n_samples)],
        'recycled_content': [random.uniform(0, 100) if random.random() > 0.2 else None for _ in range(n_samples)],
        'recyclability': [random.choice([True, False]) for _ in range(n_samples)],
        'carbon_footprint': [random.uniform(0.5, 50) if random.random() > 0.3 else None for _ in range(n_samples)],
        'energy_efficiency': [random.uniform(10, 500) if random.random() > 0.4 else None for _ in range(n_samples)],
        'durability': [random.randint(1, 20) if random.random() > 0.2 else None for _ in range(n_samples)],
        'repairability_score': [random.uniform(0, 10) if random.random() > 0.3 else None for _ in range(n_samples)],
        'ethical_sourcing': [random.choice([True, False]) for _ in range(n_samples)],
        'cruelity_free': [random.choice([True, False]) for _ in range(n_samples)],
        'plastic_free': [random.choice([True, False]) for _ in range(n_samples)],
        'natural': [random.choice([True, False]) for _ in range(n_samples)],
        'destructable': [random.choice([True, False]) for _ in range(n_samples)],
        'hazardous': [random.choice([True, False]) for _ in range(n_samples)],
    }
    
    df = pd.DataFrame(data)
    
    # Calculate sustainability score using rules
    df['sustainability_score'] = df.apply(calculate_sustainability_score, axis=1)
    
    return df

# Step 2: Define the sustainability score calculation
def calculate_sustainability_score(row):
    """
    Calculate a sustainability score (0-100) based on product attributes
    using a rule-based approach.
    
    Args:
        row: DataFrame row containing product attributes
    
    Returns:
        Float: Sustainability score (0-100)
    """
    score = 50  # Start with a neutral score
    
    # Materials impact (analyze the primary material)
    sustainable_materials = ["wood", "bamboo", "cotton", "hemp", "cork"]
    unsustainable_materials = ["plastic", "rubber"]
    
    # Check if any material in the product is sustainable or unsustainable
    has_sustainable = any(mat in sustainable_materials for mat in row['materials'])
    has_unsustainable = any(mat in unsustainable_materials for mat in row['materials'])
    
    if has_sustainable:
        score += 10
    if has_unsustainable:
        score -= 10
    
    # Recycled content
    if pd.notna(row['recycled_content']):
        score += row['recycled_content'] * 0.2  # Up to 20 points for 100% recycled
    
    # Recyclability
    if row['recyclability']:
        score += 10
    else:
        score -= 5
    
    # Carbon footprint (lower is better)
    if pd.notna(row['carbon_footprint']):
        if row['carbon_footprint'] < 5:
            score += 10
        elif row['carbon_footprint'] < 15:
            score += 5
        elif row['carbon_footprint'] > 30:
            score -= 10
    
    # Energy efficiency (lower is better)
    if pd.notna(row['energy_efficiency']):
        if row['energy_efficiency'] < 50:
            score += 10
        elif row['energy_efficiency'] > 300:
            score -= 10
    
    # Durability (higher is better)
    if pd.notna(row['durability']):
        score += min(row['durability'], 10)  # Up to 10 points for durability
    
    # Repairability (higher is better)
    if pd.notna(row['repairability_score']):
        score += row['repairability_score']  # Up to 10 points
    
    # Ethical attributes
    if row['ethical_sourcing']:
        score += 5
    if row['cruelity_free']:
        score += 5
    if row['plastic_free']:
        score += 5
    if row['natural']:
        score += 5
    
    # Negative attributes
    if row['hazardous']:
        score -= 20
    
    # Condition impact
    if row['condition'] == 'New':
        score -= 5  # Slight penalty for new vs. reused products
    elif 'Used' in row['condition']:
        score += 5  # Bonus for used products
    elif row['condition'] == 'Refurbished':
        score += 10  # Higher bonus for refurbished products
    
    # Clamp to 0-100 range
    return max(0, min(100, score))

# Step 3: Preprocess the dataset
def preprocess_data(df):
    """
    Preprocess the dataset for machine learning.
    
    Args:
        df: DataFrame with product features
    
    Returns:
        X: Features DataFrame
        y: Target Series
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
    """
    # Explode materials from list to string representation for preprocessing
    df['materials_str'] = df['materials'].apply(lambda x: ','.join(x))
    
    # Define features and target
    X = df.drop(['sustainability_score', 'materials'], axis=1)
    y = df['sustainability_score']
    
    # Get final column lists after preprocessing
    categorical_cols = ['root_category', 'main_category', 'materials_str', 'condition']
    boolean_cols = ['recyclability', 'ethical_sourcing', 'cruelity_free', 
                    'plastic_free', 'natural', 'destructable', 'hazardous']
    numeric_cols = ['recycled_content', 'carbon_footprint', 'energy_efficiency', 
                    'durability', 'repairability_score']
    
    return X, y, categorical_cols, numeric_cols, boolean_cols


# Define at the top level of your module/script
def bool_to_int(X):
    """Convert boolean values to integers"""
    return X.astype(int)

# Step 4: Create and train the model (fixed version)
def train_model(X, y, categorical_cols, numeric_cols, boolean_cols):
    """
    Create and train a model to predict sustainability score.
    
    Args:
        X: Features DataFrame
        y: Target Series
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        boolean_cols: List of boolean column names
    
    Returns:
        Trained model
        Preprocessing pipeline
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipelines
    # For categorical features: OneHotEncoding with imputation
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # For numeric features: Imputation and StandardScaler
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # For boolean features: Convert to int first, then impute
    # Inside your train_model function, change this line:
    boolean_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(dtype=int)),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
            
    # Combine transformers in a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numeric_transformer, numeric_cols),
            ('bool', boolean_transformer, boolean_cols)
        ])
    
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, random_state=46))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual Sustainability Scores')
    plt.tight_layout()
    plt.show()
    
    # Feature importance
    if hasattr(model['regressor'], 'feature_importances_'):
        # Get feature names after preprocessing
        feature_names = []
        for name, transformer, cols in model['preprocessor'].transformers_:
            if name == 'cat':
                # Get all the one-hot encoded feature names
                feature_names.extend(model['preprocessor'].named_transformers_['cat']['onehot']
                                    .get_feature_names_out(cols).tolist())
            else:
                # Add numeric and boolean feature names directly
                feature_names.extend(cols)
        
        # Get feature importances
        importances = model['regressor'].feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features
        top_n = min(20, len(feature_names))
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(top_n), importances[indices][:top_n], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
        plt.tight_layout()
        plt.show()
    
    return model, X_test, y_test

# Step 5: Predict sustainability score for new products
def predict_sustainability_score(model, new_product_data):
    """
    Predict sustainability score for new product data.
    
    Args:
        model: Trained prediction model
        new_product_data: DataFrame with new product features
    
    Returns:
        Predicted sustainability scores
    """
    # Process materials for prediction
    if 'materials' in new_product_data.columns:
        new_product_data['materials_str'] = new_product_data['materials'].apply(lambda x: ','.join(x))
        new_product_data = new_product_data.drop('materials', axis=1)
    
    # Make predictions
    predictions = model.predict(new_product_data)
    
    # Return predictions with original data
    result = new_product_data.copy()
    result['predicted_sustainability_score'] = predictions
    
    return result

# Step 7: Main function to run the entire process
def main():
    # Import the required transformer at the top
    from sklearn.preprocessing import FunctionTransformer
    
    # Generate dataset
    print("Generating synthetic dataset...")
    df = generate_dataset(n_samples=2000)
    print(f"Dataset generated with {len(df)} samples.")
    
    # Display dataset info
    print("\nDataset overview:")
    print(df.info())
    print("\nSample data:")
    print(df.head())
    
    # Summary statistics
    print("\nSustainability score distribution:")
    print(df['sustainability_score'].describe())
    
    # Visualize sustainability score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sustainability_score'], kde=True)
    plt.title('Distribution of Sustainability Scores')
    plt.xlabel('Sustainability Score')
    plt.ylabel('Frequency')
    plt.axvline(df['sustainability_score'].mean(), color='red', linestyle='--', label=f'Mean: {df["sustainability_score"].mean():.2f}')
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig('sustainability_distribution.png')
        print("Saved distribution plot to sustainability_distribution.png")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y, categorical_cols, numeric_cols, boolean_cols = preprocess_data(df)
    
    # Train model
    print("\nTraining model...")
    model, X_test, y_test = train_model(X, y, categorical_cols, numeric_cols, boolean_cols)
    
    # Example of predicting for new products
    print("\nPredicting sustainability scores for new products...")
    
    # Create a small test set of 5 new products
    sample_size = 5
    new_products = generate_dataset(n_samples=sample_size)
    
    # Drop the target variable for prediction
    new_products_features = new_products.drop('sustainability_score', axis=1)
    
    # Get predictions
    predictions = predict_sustainability_score(model, new_products_features)
    
    # Compare actual vs predicted scores for these samples
    print("\nNew product predictions:")
    comparison = pd.DataFrame({
        'Actual': new_products['sustainability_score'],
        'Predicted': predictions['predicted_sustainability_score']
    })
    print(comparison)
    
    # Calculate accuracy metrics for this mini-test
    prediction_mse = mean_squared_error(comparison['Actual'], comparison['Predicted'])
    prediction_r2 = r2_score(comparison['Actual'], comparison['Predicted'])
    
    print(f"\nPrediction accuracy on new products:")
    print(f"Mean Squared Error: {prediction_mse:.2f}")
    print(f"R² Score: {prediction_r2:.2f}")
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    x = np.arange(len(comparison))
    width = 0.35
    
    plt.bar(x - width/2, comparison['Actual'], width, label='Actual')
    plt.bar(x + width/2, comparison['Predicted'], width, label='Predicted')
    
    plt.xlabel('Product')
    plt.ylabel('Sustainability Score')
    plt.title('Actual vs Predicted Sustainability Scores')
    plt.xticks(x, [f'Product {i+1}' for i in range(len(comparison))])
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig('prediction_comparison.png')
        print("Saved prediction comparison plot to prediction_comparison.png")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    
    # Feature correlation analysis
# Feature correlation analysis
    print("\nAnalyzing feature correlations...")

    # Create a new DataFrame for correlation analysis
    corr_df = df.copy()

    # Remove non-numeric columns (we can't directly correlate lists or strings)
    corr_df = corr_df.drop(['materials', 'materials_str', 'root_category', 'main_category', 'condition'], axis=1)

    # Convert boolean columns to int for correlation
    for col in boolean_cols:
        corr_df[col] = corr_df[col].astype(int)

    # Calculate correlation matrix
    correlation = corr_df.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
        # Plot correlation heatmap
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    try:
        plt.savefig('correlation_heatmap.png')
        print("Saved correlation heatmap to correlation_heatmap.png")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    
    # Save the model for later use
    try:
        import joblib
        joblib.dump(model, 'sustainability_model.pkl')
        print("\nModel saved to sustainability_model.pkl")
    except Exception as e:
        print(f"Failed to save model: {e}")
    
    print("\nProcess completed successfully!")

# Execute main function if script is run directly
if __name__ == "__main__":
    # Import the required transformer here too for direct script execution
    from sklearn.preprocessing import FunctionTransformer
    main()