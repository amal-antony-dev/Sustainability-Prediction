import pandas as pd
import numpy as np
import random

def generate_dataset(n_samples=1000):
    """Generate synthetic sustainability data and save to CSV."""
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
    df['sustainability_score'] = df.apply(calculate_sustainability_score, axis=1)
    df.to_csv('sustainability_data.csv', index=False)
    print(f"Dataset saved to sustainability_data.csv with {n_samples} samples.")

def calculate_sustainability_score(row):
    """
    Calculate a sustainability score (0-100) based on product attributes.
    """
    score = 0
    
    # Materials impact
    sustainable_materials = {"wood", "bamboo", "cotton", "hemp", "cork"}
    unsustainable_materials = {"plastic", "rubber"}

    materials = row['materials']
    has_sustainable = any(mat in sustainable_materials for mat in materials)
    has_unsustainable = any(mat in unsustainable_materials for mat in materials)

    if has_sustainable:
        score += 10
    if has_unsustainable:
        score = max(0, score - 10)

    # Recycled content (up to 20 pts)
    if pd.notna(row['recycled_content']):
        score += min(20, row['recycled_content'] * 0.1)

    # Recyclability
    score += 10 if row['recyclability'] else -9

    # Carbon footprint (lower is better)
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

    # Energy efficiency (lower is better)
    if pd.notna(row['energy_efficiency']):
        ee = row['energy_efficiency']
        if ee < 25:
            score += 10
        elif ee < 50:
            score += 5
        else:
            score = max(0, score - 7)

    # Durability (up to 10 pts)
    if pd.notna(row['durability']):
        score += min(row['durability'], 10)

    # Repairability (up to 10 pts)
    if pd.notna(row['repairability_score']):
        score += min(row['repairability_score'], 10)

    # Ethical traits
    for key in ['ethical_sourcing', 'cruelty_free', 'plastic_free', 'natural']:
        if row[key]:
            score += 5

    # Negative traits
    if row['hazardous']:
        score = max(0, score - 10)

    # Condition
    if row['condition'] == 'New':
        score = max(0, score - 7)
    elif row['condition'] == 'Used - Like New':
        score = max(0, score - 5)
    elif row['condition'] in ['Used - Good', 'Used - Fair']:
        score += 5
    elif row['condition'] == 'For Parts or Not Working':
        score += 10

    # Normalize final score to 0-100
    return round(max(0, min(100, score)), 2)

if __name__ == "__main__":
    generate_dataset(n_samples=2000)