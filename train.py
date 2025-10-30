import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def train_models():
    """
    Trains and saves both models for the coffee rating prediction task.
    
    Returns:
        tuple: (linear_model, tree_model) The trained model objects
    """
    # Load the coffee analysis data
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    df = pd.read_csv(url)
    
    # Model 1: Creation
    # Features and target for model 1
    feature = df[['100g_USD']]
    target = df['rating']
    
    # Train model 1
    model_1 = LinearRegression()
    model_1.fit(feature, target)
    
    # Save model 1
    saveModel = open('model_1.pickle', 'wb')
    pickle.dump(model_1, saveModel)
    saveModel.close()
    
    # Model 2: Creation
    # Roast mapping to dictionary
    roast_mapping = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    
    # Create numerical roast column
    df['roast_num'] = df['roast'].map(roast_mapping)
    
    # Features for model 2
    features = df[['100g_USD', 'roast_num']]
    
    # Train model 2
    # Use same target as model 1
    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(features, target)
    
    # Save model 2
    saveModel = open('model_2.pickle', 'wb')
    pickle.dump(model_2, saveModel)
    saveModel.close()
    
    return model_1, model_2

if __name__ == "__main__":
    # Train and save both models
    model_1, model_2 = train_models()
    
    print("Model 1 (Linear Regression) trained and saved as 'model_1.pickle'")
    
    print("\nModel 2 (Decision Tree Regressor) trained and saved as 'model_2.pickle'")
