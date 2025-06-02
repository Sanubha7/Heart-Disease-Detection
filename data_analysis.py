import pandas as pd

heart_data = pd.read_csv("processed.cleveland.csv" , na_values='?')

# checking for missing data

missing_data_summary = heart_data.isna().sum()
missing_data_summary = missing_data_summary[missing_data_summary > 0]


print("Missing data summary:")
print(missing_data_summary)

# class imbalance
num_zero = (heart_data["Num"] == 0).sum()
num_non_zero = (heart_data["Num"] != 0).sum()

print(f"Number of 0s in target column: {num_zero}")
print(f"Number of 1s in target column: {num_non_zero}")
