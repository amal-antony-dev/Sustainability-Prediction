import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, param_grid=None):
    print(f"Training {model_name}...")

    numeric_features = ['recycled_content', 'carbon_footprint', 'energy_efficiency', 'durability', 'repairability_score']
    categorical_features = ['root_category', 'main_category', 'condition', 'recyclability', 'ethical_sourcing', 
                            'cruelty_free', 'plastic_free', 'natural', 'destructable', 'hazardous']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # For sklearn >= 1.2
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Optional hyperparameter tuning
    if param_grid:
        print(f"Tuning hyperparameters for {model_name}...")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=20,
            scoring='r2',
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        print("Best Parameters:", search.best_params_)
    else:
        pipeline.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipeline, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    print(f"{model_name} model trained and saved.")

    # Evaluate the model on the test set
    y_pred = pipeline.predict(X_test)
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Test R²: {r2_score(y_test, y_pred):.2f}")

    # Plot predictions
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} Predictions')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_predictions.png')
    plt.close()


if __name__ == "__main__":
    # Load preprocessed data
    train_data = pd.read_csv('train_data.csv')
    X_train = train_data.drop('sustainability_score', axis=1)
    y_train = train_data['sustainability_score']

    test_data = pd.read_csv('test_data.csv')
    X_test = test_data.drop('sustainability_score', axis=1)
    y_test = test_data['sustainability_score']

    # Define hyperparameter grid for tuning Random Forest
    rf_param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 5, 10],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

    # Define hyperparameter grid for tuning Gradient Boosting
    gb_param_grid = {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 4, 5],
        'regressor__subsample': [0.6, 0.8, 1.0],
        'regressor__min_samples_split': [2, 5, 10]
    }

    # Define hyperparameter grid for tuning Hist Gradient Boosting
    hgb_param_grid = {
        'regressor__max_iter': [100, 150, 200],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 4, 5],
        'regressor__min_samples_leaf': [20, 30, 40]
    }

    # List of models with their respective hyperparameter grids
    models = [
        ("Random Forest", RandomForestRegressor(random_state=40), rf_param_grid),
        ("Gradient Boosting", GradientBoostingRegressor(random_state=40), gb_param_grid),
        ("Hist Gradient Boosting", HistGradientBoostingRegressor(random_state=43), hgb_param_grid),
    ]

    # Train and evaluate each model with hyperparameter tuning if defined
    for model_name, model, param_grid in models:
        train_and_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, param_grid)
