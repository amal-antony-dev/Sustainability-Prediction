import pandas as pd
import numpy as np
import random
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---- STEP 1: Generate synthetic sample data ----

def generate_sample_data(n_samples=50):
    
    root_categories = [
        "Household", "Electronics", "Education", "Furniture",
        "Fashion", "Sports", "Automotive", "Toys", "Health", "Beauty", "Tools", "Pet Supplies"
    ]

    main_category_mapping = {
        "Household": ["Kitchen", "Bathroom", "Bedroom", "Garden"],
        "Electronics": ["Mobiles", "Appliances", "Laptops", "Gaming"],
        "Education": ["Books", "Stationery", "Laptops"],
        "Furniture": ["Bedroom", "Office", "Kitchen"],
        "Fashion": ["Clothing", "Shoes", "Accessories"],
        "Sports": ["Fitness", "Outdoor"],
        "Automotive": ["Car Accessories", "Tools"],
        "Toys": ["Pet Toys", "Gaming"],
        "Health": ["Fitness", "Skin Care", "Hair Care"],
        "Beauty": ["Hair Care", "Skin Care"],
        "Tools": ["Power Tools", "Hand Tools"],
        "Pet Supplies": ["Pet Food", "Pet Toys"]
    }

    materials = ["wood", "metal", "plastic", "glass", "paper", "fabric", "ceramic", "rubber", "leather", "stone",
                "bamboo", "cotton", "silk", "wool", "linen", "hemp", "jute", "coir", "sisal", "cork"]

    conditions = ["New", "Used - Like New", "Used - Good", "Used - Fair", "For Parts or Not Working"]

    # First choose root categories
    root_list = [random.choice(root_categories) for _ in range(n_samples)]

    # Then choose main categories based on selected root
    main_list = [random.choice(main_category_mapping[root]) for root in root_list]

    # Generate data dictionary
    data = {
        'root_category': root_list,
        'main_category': main_list,
        'materials': [random.sample(materials, random.randint(1, random.randint(2, 5))) for _ in range(n_samples)],
        'condition': [random.choice(conditions) for _ in range(n_samples)],
        'recycled_content': [random.uniform(0, 100) if random.random() > 0.2 else None for _ in range(n_samples)],
        'recyclability': [random.choice([True, False]) for _ in range(n_samples)],
        'carbon_footprint': [random.uniform(0.5, 50) if random.random() > 0.3 else None for _ in range(n_samples)],
        'energy_efficiency': [random.uniform(10, 100) if random.random() > 0.4 else None for _ in range(n_samples)],
        'durability': [random.randint(1, 20) if random.random() > 0.2 else None for _ in range(n_samples)],
        'repairability_score': [random.uniform(0, 10) if random.random() > 0.3 else None for _ in range(n_samples)],
        'ethical_sourcing': [random.choice([True, False]) for _ in range(n_samples)],
        'cruelty_free': [random.choice([True, False]) for _ in range(n_samples)],
        'plastic_free': [random.choice([True, False]) for _ in range(n_samples)],
        'natural': [random.choice([True, False]) for _ in range(n_samples)],
        'destructable': [random.choice([True, False]) for _ in range(n_samples)],
        'hazardous': [random.choice([True, False]) for _ in range(n_samples)],
    }

    df = pd.DataFrame(data)
    df['materials_str'] = df['materials'].apply(lambda x: ', '.join(x))
    df['sustainability_score'] = df.apply(calculate_sustainability_score, axis=1)
    return df

def calculate_sustainability_score(row):
    score = 0
    
    sustainable_materials = {"wood", "bamboo", "cotton", "hemp", "cork"}
    unsustainable_materials = {"plastic", "rubber"}

    materials = row['materials']
    has_sustainable = any(mat in sustainable_materials for mat in materials)
    has_unsustainable = any(mat in unsustainable_materials for mat in materials)

    if has_sustainable:
        score += 10
    if has_unsustainable:
        score = max(0, score - 10)

    if pd.notna(row['recycled_content']):
        score += min(20, row['recycled_content'] * 0.1)

    score += 10 if row['recyclability'] else -9

    if pd.notna(row['carbon_footprint']):
        cf = row['carbon_footprint']
        if cf < 5:
            score += 10
        elif cf < 15:
            score += 5
        elif cf < 30:
            score = max(0, score - 7)
        else:
            score = max(0, score - 10)

    if pd.notna(row['energy_efficiency']):
        ee = row['energy_efficiency']
        if ee < 25:
            score += 10
        elif ee < 50:
            score += 5
        else:
            score = max(0, score - 7)

    if pd.notna(row['durability']):
        score += min(row['durability'], 10)

    if pd.notna(row['repairability_score']):
        score += min(row['repairability_score'], 10)

    for key in ['ethical_sourcing', 'cruelty_free', 'plastic_free', 'natural']:
        if row[key]:
            score += 5

    if row['hazardous']:
        score = max(0, score - 10)

    if row['condition'] == 'New':
        score = max(0, score - 7)
    elif row['condition'] == 'Used - Like New':
        score = max(0, score - 5)
    elif row['condition'] in ['Used - Good', 'Used - Fair']:
        score += 5
    elif row['condition'] == 'For Parts or Not Working':
        score += 10

    return round(max(0, min(100, score)), 2)

# # ---- STEP 2: Load the trained model ----

def load_model(model_file='gradient_boosting_model.pkl'):
    try:
        model = joblib.load(model_file)
        print(f"Model '{model_file}' loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Model file '{model_file}' not found. Make sure to train and save the model first.")
        exit(1)

# ---- STEP 3: Predict using the model ----

def predict_scores(model, input_data):
    return model.predict(input_data)

# ---- STEP 4: Main Execution ----

if __name__ == "__main__":
    # Generate synthetic test input
    # new_data = generate_sample_data(n_samples=20)

    # Save original data for reference
    new_data = pd.read_csv("synthetic_prediction_input.csv")

    # Load the saved model (change filename to test other models)
    model = load_model('hist_gradient_boosting_model.pkl')

    # Drop target and excluded columns
    features_for_prediction = new_data.drop(columns=['materials', 'sustainability_score'])

    # Predict scores
    predicted_scores = predict_scores(model, features_for_prediction)

    # Evaluate prediction performance
    actual_scores = new_data['sustainability_score']
    mae = mean_absolute_error(actual_scores, predicted_scores)
    mse = mean_squared_error(actual_scores, predicted_scores)
    r2 = r2_score(actual_scores, predicted_scores)

    # Display predictions
    results = new_data.copy()
    results['predicted_sustainability_score'] = predicted_scores

    print("\n--- Sample Predictions ---")
    print(results[['root_category', 'main_category', 'condition', 'sustainability_score', 'predicted_sustainability_score']].head(10))

    print("\n--- Model Evaluation on Realistic Synthetic Data ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Optionally save results
    results.to_csv("synthetic_predictions.csv", index=False)
