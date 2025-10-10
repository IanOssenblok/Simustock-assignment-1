import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
import pandas as pd
from numpy import append

df = pd.read_excel("data/data_assignment.xlsx")
df_expenses = df[df["Income/Expense"] == "Expense"].copy()
df_expenses = df_expenses["Amount"].copy()
df_incomes = df[df["Income/Expense"] == "Income"].copy()
df_incomes = df_incomes["Amount"].copy()
df["Date"] = pd.to_datetime(df["Date"]) # makes sure dates are in datetime format

total_days = (df['Date'].max().normalize() - df['Date'].min().normalize()).days + 1

lam_combined = len(df) / total_days
lam_incomes = len(df_incomes) / total_days
lam_expenses = len(df_expenses) / total_days
T = total_days
p_income = len(df_incomes) / len(df)


def single_simulation(data, lam, T, R):
    expDist = stats.expon(scale=1/lam)

    # fit distribution for income data
    M1 = np.mean(data)
    M2 = np.mean(data ** 2)
    alpha = M1 ** 2 / (M2 - M1 ** 2)
    beta = M1 / (M2 - M1 ** 2)
    fitGammaDist = stats.gamma(alpha, scale=1 / beta)

    # fit distribution for expense data
    c, _, scale = stats.fisk.fit(data)
    fitLogLogis = stats.fisk(c, scale=scale)

    #
    shape, _, scale = stats.lognorm.fit(data)
    fitLogNorm = stats.lognorm(shape, scale=scale)

    simulation_results = []

    for r in range(R):
        t = expDist.rvs()
        arrTimes = []
        incomes = []

        while t < T:
            arrTimes.append(t)
            if data is df_incomes:
                incomes.append(fitGammaDist.rvs())
            if data is df_expenses:
                incomes.append(fitLogLogis.rvs())
            t += expDist.rvs()

        arrTimes = np.array(arrTimes)
        cumulative_income = np.cumsum(incomes)
        simulation_results.append(cumulative_income[-1])

        # Plot only the last simulation
        if r == 0:
            xs = np.concatenate(([0], arrTimes, [T]))
            ys = np.concatenate(([0], cumulative_income, [cumulative_income[-1]]))

            plt.figure(figsize=(8, 4))
            plt.step(xs, ys, where='post', label=f"Simulated cumulative income/expense (Run 1)")
            plt.xlabel("Time (days)")
            plt.ylabel("Cumulative Income/Expense")
            plt.title(f"Simulated Income/Expense Over Time (R=1)")
            plt.legend()
            plt.grid(True)
            plt.show()

    # Return the average of the final cumulative income values
    avg_final_income = np.mean(simulation_results)
    return avg_final_income


# print(single_simulation(df_incomes, lam_incomes, T, 1000))  # log-log: 35k avg, log-normal: 3k avg, gamma: 2,5k avg
# print(single_simulation(df_expenses, lam_expenses, T, 1000)) #log-log: 3k avg, log-normal: 2.2k avg



def combined_simulation(incomes, expenses, lam, T, R, p_income):
    expDist = stats.expon(scale=1 / lam)

    # fit distribution for income data
    M1 = np.mean(incomes)
    M2 = np.mean(incomes ** 2)
    alpha = M1 ** 2 / (M2 - M1 ** 2)
    beta = M1 / (M2 - M1 ** 2)
    fitGammaDist = stats.gamma(alpha, scale=1 / beta)

    # fit distribution for expense data
    c, _, scale = stats.fisk.fit(expenses)
    fitLogLogis = stats.fisk(c, scale=scale)

    #
    shape, _, scale = stats.lognorm.fit(expenses)
    fitLogNorm = stats.lognorm(shape, scale=scale)

    simulation_results = []

    for r in range(R):
        t = expDist.rvs()
        arrTimes = []
        balance = []

        while t < T:
            arrTimes.append(t)
            if np.random.uniform(0, 1) < p_income: # ask the tutor if this is allowed
                balance.append(fitGammaDist.rvs()) # incomes
            else:
                balance.append(-fitLogNorm.rvs()) # expenses
            t += expDist.rvs()

        arrTimes = np.array(arrTimes)
        cumulative_income = np.cumsum(balance)
        simulation_results.append(cumulative_income[-1])

        # Plot only the last simulation
        if r == 0:
            xs = np.concatenate(([0], arrTimes, [T]))
            ys = np.concatenate(([0], cumulative_income, [cumulative_income[-1]]))

            plt.figure(figsize=(8, 4))
            plt.step(xs, ys, where='post', label=f"Simulated balance (Run 1)")
            plt.xlabel("Time (days)")
            plt.ylabel("Balance")
            plt.title(f"Simulated Balance Over Time (R=1)")
            plt.legend()
            plt.grid(True)
            plt.show()

    # Return the average of the final cumulative income values
    avg_final_income = np.mean(simulation_results)
    return avg_final_income

# print(combined_simulation(df_incomes, df_expenses, lam_combined, T, 1000, p_income))

# Example 1 where food prices are increased by 20%
df_example_food20 = df[df["Income/Expense"] == "Expense"].copy()
df_example_food20.loc[df_example_food20["Category"] == "Food", "Amount"] *= 1.2
df_example_food20 = df_example_food20['Amount'].copy()
# print(single_simulation(df_example_food20, lam_expenses, T, 1000))
# print(combined_simulation(df_incomes, df_example_food20, lam_combined, T, 1000))

# Example 2 where she does not have a job
df_example_salary = df[df["Income/Expense"] == "Income"].copy()
df_example_salary = df_example_salary[df_example_salary["Category"] != "Salary"]
df_example_salary = df_example_salary['Amount'].copy()
lam_combined_ex2 = (len(df_example_salary) + len(df_expenses)) / total_days
# print(combined_simulation(df_example_salary, df_expenses, lam_combined_ex2, T, 1000))



