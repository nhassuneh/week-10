# your code here
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle
import numpy as np

# Load data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

def train_model_1():
    """
    Trains a linear regression model to predict coffee ratings based on price per 100g.
    Saves the trained model as 'model_1.pickle'.
    
    Returns:
        LinearRegression: The trained model
    """
    # Prepare features and target
    feature = df.loc[:, ['100g_USD']] 
    target = df.loc[:, 'rating']
    
    # Model training
    model_1 = LinearRegression()
    model_1.fit(feature, target)
    
    # Save trained model
    saveModel = open('model_1.pickle', 'wb')
    pickle.dump(model_1, saveModel)
    saveModel.close()
    
    return model_1


def train_model_2():
    """
    Trains a decision tree regressor to predict coffee ratings based on price per 100g and roast type.
    Saves the trained model as 'model_2.pickle'.
    
    Returns:
        DecisionTreeRegressor: The trained model
    """
    
    
    # Roast mapping to dictionary
    roast_mapping = {
        'Light': 0,
        'Medium-Light': 1,
        'Medium': 2,
        'Medium-Dark': 3,
        'Dark': 4
    }
    
    # Create numerical roast column
    df['roast'] = df['roast'].map(roast_mapping)
    
    # Prepare features and target
    features = df[['100g_USD', 'roast']]
    target = df['rating']
    
    # Train the decision tree regressor
    model_2 = DecisionTreeRegressor(random_state=42)
    model_2.fit(features, target)
    
    # Save trained model
    saveModel = open('model_2.pickle', 'wb')
    pickle.dump(model_2, saveModel)
    saveModel.close()
    
    return model_2

if __name__ == "__main__":
    # Train and save both models
    model_1 = train_model_1()
    model_2 = train_model_2()
    
    print("\nModel 1 (Linear Regression):")
    print("Trained and saved as 'model_1.pickle'")
    
    print("\nModel 2 (Decision Tree Regressor):")
    print("Trained and saved as 'model_2.pickle'")