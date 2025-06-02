import joblib
import numpy as np
from sklearn.metrics import classification_report

# Function to load the saved model and test dataset
def load_model_and_data():
    # Loading
    model_info = joblib.load('models/best_heart_disease_model.pkl')
    test_data = joblib.load('models/test_data.pkl')

    
    # Return all the loaded objects
    return {
        'model_info': model_info,
        'test_data': test_data,
    }

# Print classification report using saved test data
def print_classification_report(loaded_objects):

    # Extract test features and labels
    X_test = loaded_objects['test_data']['X_test']
    y_test = loaded_objects['test_data']['y_test']
    
    # Initialize an empty array to hold predictions from 3 decision trees
    predictions = np.zeros((len(X_test), 3))

    # Loop through each decision tree stored in the model and make predictions
    for i, tree in enumerate(loaded_objects['model_info']['trees']):
        predictions[:,i] = tree.predict(X_test)
    
    # Final prediction using majority voting
    y_pred = (np.mean(predictions, axis=1) > 0.5).astype(int)
    

    print("\nClassification Report on Test Data:")
    print(classification_report(y_test, y_pred))
    print(f"Model Accuracy: {loaded_objects['model_info']['accuracy']:.4f}")

if __name__ == "__main__":

    # Load all necessary objects
    loaded_objects = load_model_and_data()
    
    # Print classification report
    print_classification_report(loaded_objects)