import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

df = pd.read_excel("data/data_assignment.xlsx")
df["Date"] = pd.to_datetime(df["Date"]) # makes sure dates are in datetime format
print(df.head())
print("")
print(df.describe())
print("")

cat_mean_std_df = df.groupby("Category")["Date"].agg(["mean", "std"])
print(cat_mean_std_df)

fig = plt.figure(figsize=(10,6))
plt.hist(df["Date"], bins=30)
plt.xlabel("Date")
plt.ylabel("Frequency")
plt.title("Distribution of Dates")
plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(10, 6))
plt.boxplot([mdates.date2num(df['Date'])])

plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())

plt.title('Boxplot of Dates')
plt.ylabel('Date')
plt.xticks([1], ['Dates'])  # Label for the single box
plt.tight_layout()
plt.show()

fig3 = plt.figure(figsize=(14, 8))

categories = df['Category'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

for i, category in enumerate(categories):
    category_dates = df[df['Category'] == category]['Date'].sort_values()

    if len(category_dates) > 1:
        # lines
        plt.plot(category_dates, [i] * len(category_dates),
                 color=colors[i], linewidth=2, alpha=0.7)

    # points
    plt.scatter(category_dates, [i] * len(category_dates),
                color=colors[i], s=100, alpha=0.8, edgecolor='black')

# Format x-axis as dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

plt.yticks(range(len(categories)), categories)
plt.title('Date Distribution by Category', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Category', fontweight='bold')
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()