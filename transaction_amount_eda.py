import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data pre-processing
df = pd.read_excel("data/data_assignment.xlsx", parse_dates=["Date"])
df['Income/Expense'] = df['Income/Expense'].str.strip().str.lower() # removing any leading/trailing whitespace and make everything lowercase
df['is_income'] = df['Income/Expense'].isin(['income']) # looks at all the 'Income/Expense' column and keeps track of 'income'
df['is_expense'] = df['Income/Expense'].isin(['expense']) # looks at all the 'Income/Expense' column and keeps track of 'expense'
df['signed_amount'] = np.where(df['is_income'], df['Amount'], -np.abs(df['Amount'])) # creating a column where value is positive for income and negative for expenses
df = df.sort_values('Date').reset_index(drop=True) # sorting dataframe by 'date' chronologically and resetting index for sequential ordering

# table containing the summary statistics for overall income and expenses
summary = df.groupby('Income/Expense')['Amount'].agg(['count', 'sum', 'mean', 'std', 'min', 'max']).T
print(summary)

# table containing the summary statistics for each category of either income or expense
cat_summary = df.groupby('Category')['Amount'].agg(['count', 'sum', 'mean', 'std']).sort_values('sum', ascending=False)
cat_summary['Type'] = df.groupby('Category')['Income/Expense'].first()
print(cat_summary)

# figure for transaction according to the day of the week
df['Weekday'] = df['Date'].dt.day_name()
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Weekday', hue='Income/Expense', order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title("Number of Transactions by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
