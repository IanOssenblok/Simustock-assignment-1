import pandas as pd

df = pd.read_excel("data/data_assignment.xlsx")
# print(df.head())
# print(df.columns) # Index(['Date', 'Category', 'Income/Expense', 'Amount'], dtype='object')
print(df["Amount"].describe())
# print(df.isna().sum()) # no missing values

