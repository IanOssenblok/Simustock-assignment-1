# ================================================================
# Combined Visualization and Simulation Script
# Saves all figures in organized folders for Assignment 1
# ================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats
from scipy.stats import bernoulli
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.distributions.empirical_distribution import ECDF

# ------------------------------------------------
# Create required folders
# ------------------------------------------------
folders = ["time_eda", "amount_eda", "distribution_fitting", "simulation"]
for f in folders:
    os.makedirs(f, exist_ok=True)

# ================================================================
# 1. TIME-BASED EDA
# ================================================================

df = pd.read_excel("data/data_assignment.xlsx")
df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime format

cat_mean_std_df = df.groupby("Category")["Date"].agg(["mean", "std"])
print(cat_mean_std_df)

# Figure 1
plt.figure(figsize=(10, 6))
plt.hist(df["Date"], bins=30)
plt.xlabel("Date")
plt.ylabel("Frequency")
plt.title("Distribution of Dates")
plt.tight_layout()
plt.savefig("time_eda/01_Distribution_of_Dates.png", dpi=300)
plt.show()

# Figure 2
plt.figure(figsize=(10, 6))
plt.boxplot([mdates.date2num(df['Date'])])
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().yaxis.set_major_locator(mdates.AutoDateLocator())
plt.title('Boxplot of Dates')
plt.ylabel('Date')
plt.xticks([1], ['Dates'])
plt.tight_layout()
plt.savefig("time_eda/02_Boxplot_of_Dates.png", dpi=300)
plt.show()

# Figure 3
plt.figure(figsize=(14, 8))
categories = df['Category'].unique()
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

for i, category in enumerate(categories):
    category_dates = df[df['Category'] == category]['Date'].sort_values()
    if len(category_dates) > 1:
        plt.plot(category_dates, [i] * len(category_dates),
                 color=colors[i], linewidth=2, alpha=0.7)
    plt.scatter(category_dates, [i] * len(category_dates),
                color=colors[i], s=100, alpha=0.8, edgecolor='black')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.yticks(range(len(categories)), categories)
plt.title('Date Distribution by Category', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontweight='bold')
plt.ylabel('Category', fontweight='bold')
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("time_eda/03_Date_Distribution_by_Category.png", dpi=300)
plt.show()


# ================================================================
# 2. AMOUNT-BASED EDA
# ================================================================

df = pd.read_excel("data/data_assignment.xlsx", parse_dates=["Date"])
df['Income/Expense'] = df['Income/Expense'].str.strip().str.lower()
df['is_income'] = df['Income/Expense'].isin(['income', 'in', 'credit', '+'])
df['is_expense'] = df['Income/Expense'].isin(['expense', 'out', 'debit', '-'])
if not df['is_income'].any() and not df['is_expense'].any():
    df['is_income'] = df['Amount'] > 0
    df['is_expense'] = df['Amount'] < 0
df['signed_amount'] = np.where(df['is_income'], df['Amount'], -np.abs(df['Amount']))
df = df.sort_values('Date').reset_index(drop=True)

cat_summary = df.groupby('Category')['Amount'].agg(['count', 'sum', 'mean', 'std']).sort_values('sum', ascending=False)
cat_summary['Type'] = df.groupby('Category')['Income/Expense'].first()
daily = df.groupby('Date')['signed_amount'].sum().reset_index()
daily['balance'] = daily['signed_amount'].cumsum()

# Plot 1: Daily Net Flow
plt.figure(figsize=(12, 4))
plt.plot(daily['Date'], daily['signed_amount'])
plt.axhline(0, color='gray', linestyle='--')
plt.title("Daily Net Inflow/Outflow")
plt.xlabel("Date")
plt.ylabel("Amount (€)")
plt.tight_layout()
plt.savefig("amount_eda/01_Daily_Net_Flow.png", dpi=300)
plt.show()

# Plot 2: Running Balance
plt.figure(figsize=(12, 4))
plt.plot(daily['Date'], daily['balance'], color='purple')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Running Balance Over Time")
plt.xlabel("Date")
plt.ylabel("Balance (€)")
plt.tight_layout()
plt.savefig("amount_eda/02_Running_Balance.png", dpi=300)
plt.show()

# Plot 3: Income Amounts Distribution
plt.figure(figsize=(10, 4))
plt.hist(df[df['is_income']]['Amount'], bins=30, color='green', alpha=0.7)
plt.title("Income Amounts Distribution")
plt.xlabel("Amount (€)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("amount_eda/03_Income_Amounts_Distribution.png", dpi=300)
plt.show()

# Plot 4: Expense Amounts Distribution
plt.figure(figsize=(10, 4))
plt.hist(df[df['is_expense']]['Amount'].abs(), bins=30, color='red', alpha=0.7)
plt.title("Expense Amounts Distribution")
plt.xlabel("Amount (€)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("amount_eda/04_Expense_Amounts_Distribution.png", dpi=300)
plt.show()

def interarrival(dates):
    sorted_dates = np.sort(pd.to_datetime(dates).unique())
    return np.diff(sorted_dates).astype('timedelta64[D]').astype(int)

inc_ia = interarrival(df.loc[df['is_income'], 'Date'])
exp_ia = interarrival(df.loc[df['is_expense'], 'Date'])

# Plot 5: Interarrival Income
plt.figure(figsize=(10, 4))
plt.hist(inc_ia, bins=15, color='green', alpha=0.7)
plt.title("Interarrival Times – Income (days)")
plt.xlabel("Days Between Income Events")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("amount_eda/05_Interarrival_Income.png", dpi=300)
plt.show()

# Plot 6: Interarrival Expense
plt.figure(figsize=(10, 4))
plt.hist(exp_ia, bins=15, color='red', alpha=0.7)
plt.title("Interarrival Times – Expense (days)")
plt.xlabel("Days Between Expense Events")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("amount_eda/06_Interarrival_Expense.png", dpi=300)
plt.show()

# Plot 7: Transactions by Weekday
df['Weekday'] = df['Date'].dt.day_name()
weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
counts = df.groupby(['Weekday','Income/Expense']).size().unstack(fill_value=0).reindex(weekdays)
counts.plot(kind='bar', figsize=(10,5))
plt.title("Number of Transactions by Day of Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("amount_eda/07_Transactions_by_Day_of_Week.png", dpi=300)
plt.show()

cat_summary.to_csv("amount_eda/category_summary.csv")
daily.to_csv("amount_eda/daily_balance.csv")


# ================================================================
# 3. DISTRIBUTION FITTING
# ================================================================

df = pd.read_excel("data/data_assignment.xlsx")
df_expenses = df[df["Income/Expense"] == "Expense"]["Amount"].copy()
df_incomes = df[df["Income/Expense"] == "Income"]["Amount"].copy()
df["Date"] = pd.to_datetime(df["Date"])

def create_ecdf(data, label):
    M1 = np.mean(data)
    M2 = np.mean(data ** 2)
    mu = M1
    sigma2 = M2 - M1 ** 2
    xs = np.arange(np.min(data), np.max(data), 0.1)
    ecdf = ECDF(data)
    plt.figure(figsize=(8,6))
    plt.step(ecdf.x, ecdf.y, color='black', where='post', label="Empirical CDF")

    results = []
    fitNorm = stats.norm(mu, np.sqrt(sigma2))
    plt.plot(xs, fitNorm.cdf(xs), 'b', label='Normal')
    results.append(("Normal", *stats.kstest(data, fitNorm.cdf)))

    alpha = M1**2 / (M2 - M1**2)
    beta = M1 / (M2 - M1**2)
    fitGamma = stats.gamma(alpha, scale=1/beta)
    plt.plot(xs, fitGamma.cdf(xs), 'r', label='Gamma')
    results.append(("Gamma", *stats.kstest(data, fitGamma.cdf)))

    lam = 1 / M1
    fitExp = stats.expon(scale=1/lam)
    plt.plot(xs, fitExp.cdf(xs), 'g', label='Exponential')
    results.append(("Exponential", *stats.kstest(data, fitExp.cdf)))

    a, b = np.min(data), np.max(data)
    fitUnif = stats.uniform(loc=a, scale=b - a)
    plt.plot(xs, fitUnif.cdf(xs), color='orange', label='Uniform')
    results.append(("Uniform", *stats.kstest(data, fitUnif.cdf)))

    shape, _, scale = stats.lognorm.fit(data)
    fitLogNorm = stats.lognorm(shape, scale=scale)
    plt.plot(xs, fitLogNorm.cdf(xs), 'purple', label='Lognormal')
    results.append(("Lognormal", *stats.kstest(data, fitLogNorm.cdf)))

    c, _, scale = stats.fisk.fit(data)
    fitLogLogis = stats.fisk(c, scale=scale)
    plt.plot(xs, fitLogLogis.cdf(xs), 'brown', label='Log-logistic')
    results.append(("Log-logistic", *stats.kstest(data, fitLogLogis.cdf)))

    plt.legend()
    plt.title(f"ECDF with Distribution Fits ({label})")
    plt.xlabel("x"); plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(f"distribution_fitting/{label}_ECDF.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10,10))
    plt.hist(data, bins=15, density=True, alpha=1, color='gray', label='Histogram')
    plt.plot(xs, fitNorm.pdf(xs), 'b', label='Normal')
    plt.plot(xs, fitGamma.pdf(xs), 'r', label='Gamma')
    plt.plot(xs, fitExp.pdf(xs), 'g', label='Exponential')
    plt.plot(xs, fitUnif.pdf(xs), color='orange', label='Uniform')
    plt.plot(xs, fitLogNorm.pdf(xs), 'purple', label='Lognormal')
    plt.plot(xs, fitLogLogis.pdf(xs), 'brown', label='Log-logistic')
    plt.legend()
    plt.title(f"Histogram with Fitted Distributions ({label})")
    plt.xlabel("x"); plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"distribution_fitting/{label}_Histogram.png", dpi=300)
    plt.show()

create_ecdf(df_incomes, "Income")
create_ecdf(df_expenses, "Expense")


# ================================================================
# 4. SIMULATION
# ================================================================

df = pd.read_excel("data/data_assignment.xlsx")
df_expenses = df[df["Income/Expense"] == "Expense"]["Amount"].copy()
df_incomes = df[df["Income/Expense"] == "Income"]["Amount"].copy()
df["Date"] = pd.to_datetime(df["Date"])
total_days = (df['Date'].max().normalize() - df['Date'].min().normalize()).days + 1

lam_combined = len(df) / total_days
lam_incomes = len(df_incomes) / total_days
lam_expenses = len(df_expenses) / total_days
T = total_days
p_income = len(df_incomes) / len(df)

def single_simulation(data, lam, T, R, label):
    expDist = stats.expon(scale=1/lam)
    M1 = np.mean(data)
    M2 = np.mean(data ** 2)
    alpha = M1**2 / (M2 - M1**2)
    beta = M1 / (M2 - M1**2)
    fitGamma = stats.gamma(alpha, scale=1/beta)
    lamExp = 1/M1
    fitExp = stats.expon(scale=1/lamExp)
    results = []

    for r in range(R):
        t = expDist.rvs()
        arrTimes, vals = [], []
        while t < T:
            arrTimes.append(t)
            if data is df_incomes:
                vals.append(fitGamma.rvs())
            if data is df_expenses:
                vals.append(fitExp.rvs())
            t += expDist.rvs()
        vals = np.array(vals)
        results.append(np.sum(vals))
        if r == 0:
            xs = np.concatenate(([0], arrTimes, [T]))
            ys = np.concatenate(([0], np.cumsum(vals), [np.cumsum(vals)[-1]]))
            plt.figure(figsize=(8,4))
            plt.step(xs, ys, where='post')
            plt.xlabel("Time (days)")
            plt.ylabel("Cumulative Income/Expense")
            plt.title(f"{label} Simulation Example (R=1)")
            plt.tight_layout()
            plt.savefig(f"simulation/{label}_Example_Run.png", dpi=300)
            plt.show()
    avg = np.mean(results)
    ci = DescrStatsW(results).tconfint_mean(alpha=0.05)
    return avg, ci

def combined_simulation(incomes, expenses, lam, T, R, p_income, label):
    expDist = stats.expon(scale=1/lam)
    M1i, M2i = np.mean(incomes), np.mean(incomes ** 2)
    alpha = M1i**2 / (M2i - M1i**2)
    beta = M1i / (M2i - M1i**2)
    fitGamma = stats.gamma(alpha, scale=1/beta)
    M1e = np.mean(expenses)
    lamExp = 1 / M1e
    fitExp = stats.expon(scale=1 / lamExp)
    results = []

    for r in range(R):
        t = expDist.rvs()
        arrTimes, vals = [], []
        while t < T:
            arrTimes.append(t)
            vals.append(fitGamma.rvs() if bernoulli.rvs(p_income) else -fitExp.rvs())
            t += expDist.rvs()
        vals = np.array(vals)
        results.append(np.sum(vals))
        if r == 0:
            xs = np.concatenate(([0], arrTimes, [T]))
            ys = np.concatenate(([0], np.cumsum(vals), [np.cumsum(vals)[-1]]))
            plt.figure(figsize=(8,4))
            plt.step(xs, ys, where='post')
            plt.xlabel("Time (days)")
            plt.ylabel("Balance")
            plt.title(f"{label} Combined Simulation Example (R=1)")
            plt.tight_layout()
            plt.savefig(f"simulation/{label}_Combined_Example.png", dpi=300)
            plt.show()
    avg = np.mean(results)
    ci = DescrStatsW(results).tconfint_mean(alpha=0.05)
    return avg, ci

# Example usage
print("Income:", single_simulation(df_incomes, lam_incomes, T, 1000, "Income"))
print("Expense:", single_simulation(df_expenses, lam_expenses, T, 1000, "Expense"))
print("Combined:", combined_simulation(df_incomes, df_expenses, lam_combined, T, 1000, p_income, "Baseline"))

# Exponential fit check
def check_exponential_fit(interarrivals, alpha=0.05, label="Exponential_Fit"):
    interarrivals = np.array(interarrivals)
    interarrivals = interarrivals[interarrivals > 0]
    lam_hat = 1 / np.mean(interarrivals)
    ks_stat, p_value = stats.kstest(interarrivals, 'expon', args=(0, 1/lam_hat))
    x = np.linspace(0, np.max(interarrivals), 100)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(interarrivals, bins=30, density=True, alpha=0.6, label='Data')
    plt.plot(x, lam_hat*np.exp(-lam_hat*x), 'r-', label=f'Exp(lambda={lam_hat:.3f})')
    plt.legend(); plt.title("Histogram + Fitted Exp PDF")
    plt.subplot(1,2,2)
    stats.probplot(interarrivals, dist=stats.expon(scale=1/lam_hat), plot=plt)
    plt.title("Exponential Q-Q Plot")
    plt.tight_layout()
    plt.savefig(f"simulation/{label}.png", dpi=300)
    plt.show()
    return {"lambda_hat": lam_hat, "KS_statistic": ks_stat, "p_value": p_value}

expDist = stats.expon(scale=1/lam_combined)
times = []
t = expDist.rvs()
while t < T:
    times.append(t)
    t += expDist.rvs()
interarrivals = np.diff([0] + times)
print(check_exponential_fit(interarrivals))
