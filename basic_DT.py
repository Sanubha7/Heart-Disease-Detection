import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load and prepare data
def load_data():
    column_names = ['Age', 'Sex', 'CP', 'TrestBPS', 'Chol', 'FBS', 'RestECG', 
                   'Thalach', 'Exang', 'OldPeak', 'Slope', 'CA', 'Thal', 'Num']
    df = pd.read_csv('processed.cleveland.csv', names=column_names, header=0, na_values='?')
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Create binary target
    df_imputed['Target'] = (df_imputed['Num'] > 0).astype(int)
    
    features = ['Age', 'Sex', 'CP', 'TrestBPS', 'Chol', 'FBS', 'RestECG', 
               'Thalach', 'Exang', 'OldPeak', 'Slope', 'CA', 'Thal']
    X = df_imputed[features].values
    y = df_imputed['Target'].values
    
    return X, y

# Main execution
if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Train basic decision tree
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tree.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Print results 
    print(f"Final Model Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2))