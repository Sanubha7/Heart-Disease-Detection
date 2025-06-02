## About the Files

1) processed.cleveland.csv is the dataset used.
   - Dataset Source (UCI Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease)

3) The data_analysis.py file is the initial attempt to explore the dataset.  
   - Checked for any missing data.  
   - Examined whether there was a significant class imbalance.
   - Running this file outputs some statistics about the dataset

4) In basic_DT.py, a basic decision tree was built using Scikit-learn's DecisionTreeClassifier.  
   - The model achieved an accuracy of 65.8%.
   - Running this trains a basic decision tree and prints the classificiation report

5) The optimized_DT.py file contains an optimized decision tree with hyperparameter tuning and a Random Forest implementation.
   - This model achieved an accuracy of 85.8%
   - Running this trains the optimized model and saves it

6) The models are saved data from optimized_DT.py

7) Running predict.py prints the classification report for the optimized decision tree 

## How to run:
 Run "predict.py" to print out the classification report for the optimized decision tree.# Heart-Disease-Detection
