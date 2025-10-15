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


daily = df.groupby('Date')['signed_amount'].sum().reset_index()
daily['balance'] = daily['signed_amount'].cumsum()

# figure for running balance over time
plt.figure(figsize=(12, 4))
plt.plot(daily['Date'], daily['balance'], label='Running Balance', color='purple')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Running Balance Over Time")
plt.xlabel("Date")
plt.ylabel("Balance (€)")
plt.tight_layout()
plt.show()

# function to compute interarrival times (in days) between unique dates
def interarrival(dates):
    sorted_dates = np.sort(pd.to_datetime(dates).unique())
    return np.diff(sorted_dates).astype('timedelta64[D]').astype(int)

inc_ia = interarrival(df.loc[df['is_income'], 'Date']) # computing interarrival times for income transactions
exp_ia = interarrival(df.loc[df['is_expense'], 'Date']) # computing interarrival times for expense transactions


# figure for of income interarrival times
plt.figure(figsize=(10, 4))
sns.histplot(inc_ia, bins=15, kde=False, color='green')
plt.title("Interarrival Times – Income (days)")
plt.xlabel("Days Between Income Events")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# figure for expense interarrival times
plt.figure(figsize=(10, 4))
sns.histplot(exp_ia, bins=15, kde=False, color='red')
plt.title("Interarrival Times – Expense (days)")
plt.xlabel("Days Between Expense Events")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
