# Import functions from apputil
from apputil import train_model_1, train_model_2

def print_models():
    "Running the models from apputil.py"
    
    # Test train_model_1
    print("\nTraining Model 1 (Linear Regression)...")
    model_1 = train_model_1()

    # Test train_model_2
    print("\nTraining Model 2 (Decision Tree Regressor)...")
    model_2 = train_model_2()
    
    return model_1, model_2

if __name__ == "__main__":
    # Test the functions from apputil.py
    model_1, model_2 = print_models()
    
    print("\nBoth models trained and tested successfully!")
    print("Model 1 saved as 'model_1.pickle'")
    print("Model 2 saved as 'model_2.pickle'")
