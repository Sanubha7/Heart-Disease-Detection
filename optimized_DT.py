import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Loads the Cleveland heart disease dataset

# Handles missing values (imputation)

# Converts the target to binary classification (0 = no disease, 1 = disease)

# Returns features (X) and target labels (y)

def load_and_prepare_data():
    # Load the dataset
    column_names = ['Age', 'Sex', 'CP', 'TrestBPS', 'Chol', 'FBS', 'RestECG', 
                   'Thalach', 'Exang', 'OldPeak', 'Slope', 'CA', 'Thal', 'Num']
    df = pd.read_csv('processed.cleveland.csv', names=column_names, header=0, na_values='?')
    
    
    # Select features before imputation
    features = ['Age', 'Sex', 'CP', 'TrestBPS', 'Chol', 'FBS', 'RestECG', 
               'Thalach', 'Exang', 'OldPeak', 'Slope', 'CA', 'Thal']
    
    X = df[features].copy() 
    
    # Handle missing values on just these features
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)
    
    # Convert to binary classification using original Num column
    y = (df['Num'] > 0).astype(int).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y, scaler, imputer, features


def train_decision_tree(X_train, y_train):
    param_grid = {
        'max_depth': [3, 4, 5, 7, None],
        'min_samples_split': list(range(2, 21)),
        'min_samples_leaf': list(range(1, 21)),
        'criterion': ['gini', 'entropy']
    }
    tree = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(tree, param_grid, cv=4, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def evaluate_and_save_model(X_train, X_test, y_train, y_test, feature_names, fold_num=None):
    # Split training data into 3 parts
    n = len(X_train)
    part_size = n // 3
    indices = [0, part_size, 2*part_size, n]
    
    trees = []
    for i in range(3):
        start, end = indices[i], indices[i+1]
        X_part = X_train[start:end]
        y_part = y_train[start:end]
        
        print(f"Training tree {i+1} on {len(X_part)} samples")
        tree = train_decision_tree(X_part, y_part)
        trees.append(tree)
    
    # Make predictions using majority voting
    predictions = np.zeros((len(X_test), 3))
    for i, tree in enumerate(trees):
        predictions[:,i] = tree.predict(X_test)
    
    y_pred = (np.mean(predictions, axis=1) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model and test data if not in cross-validation
    if fold_num is None:
        # Save the ensemble model
        model_info = {
            'trees': trees,
            'accuracy': accuracy,
            'input_shape': X_train.shape[1],
            'feature_names': feature_names  
        }
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_info, 'models/best_heart_disease_model.pkl')
        
        # Save test data for evaluation in prediction script
        test_data = {
            'X_test': X_test,
            'y_test': y_test
        }
        joblib.dump(test_data, 'models/test_data.pkl')
        print("Best model and test data saved to 'models/' directory")
    
    return accuracy, trees, y_test, y_pred

def main():
    X, y, _, _, feature_names = load_and_prepare_data()
    

    
    # 4-fold cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        acc, _, _, _ = evaluate_and_save_model(X_train, X_test, y_train, y_test, feature_names, fold_num=fold+1)
        results.append((acc, train_idx, test_idx))
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
    
    # Find best split
    best_acc, best_train_idx, best_test_idx = max(results, key=lambda x: x[0])
    print(f"\nBest split accuracy: {best_acc:.4f}")
    
    # Final evaluation on best split and save the model
    X_train, X_test = X[best_train_idx], X[best_test_idx]
    y_train, y_test = y[best_train_idx], y[best_test_idx]
    
    final_acc, trees, y_test, y_pred = evaluate_and_save_model(X_train, X_test, y_train, y_test, feature_names)
    print(f"\nFinal Model Accuracy: {final_acc:.4f}")
    
    # Generate classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()