import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import bernoulli
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


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
    # fit gamma distribution for income data (use this for simulations)
    M1 = np.mean(data)
    M2 = np.mean(data ** 2)
    alpha = M1 ** 2 / (M2 - M1 ** 2)
    beta = M1 / (M2 - M1 ** 2)
    fitGammaDist = stats.gamma(alpha, scale=1 / beta)

    # fit log logistic distribution for expense/income data (only used for sensitivity analysis)
    c, _, scale = stats.fisk.fit(data)
    fitLogLogis = stats.fisk(c, scale=scale)

    # fit log normal distribution for expense/income data (only used for sensitivity analysis)
    shape, _, scale = stats.lognorm.fit(data)
    fitLogNorm = stats.lognorm(shape, scale=scale)

    # fit exponential distribution for expense data (use this for simulations)
    lamExp = 1 / M1
    fitExpDist = stats.expon(scale=1 / lamExp)

    # save each simulation result
    simulation_results = []

    for r in range(R):
        t = expDist.rvs()
        arrTimes = []
        incomes = []

        while t < T:
            arrTimes.append(t)
            if data is df_incomes:
                incomes.append(fitGammaDist.rvs()) # change income distribution here
            if data is df_expenses:
                incomes.append(fitExpDist.rvs()) # change expense distribution here
            t += expDist.rvs()

        arrTimes = np.array(arrTimes)
        cumulative_income = np.cumsum(incomes)
        simulation_results.append(cumulative_income[-1])

        # Plot only the first simulation
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
    ci = DescrStatsW(simulation_results).tconfint_mean(alpha=0.05)
    return avg_final_income, ci


def combined_simulation(incomes, expenses, lam, T, R, p_income):
    expDist = stats.expon(scale=1 / lam)

    # fit gamma distribution for income data
    M1 = np.mean(incomes)
    M2 = np.mean(incomes ** 2)
    alpha = M1 ** 2 / (M2 - M1 ** 2)
    beta = M1 / (M2 - M1 ** 2)
    fitGammaDist = stats.gamma(alpha, scale=1 / beta)

    # fit exponential distribution for expense data
    M1Exp = np.mean(expenses)
    lamExp = 1 / M1Exp
    fitExpDist = stats.expon(scale=1 / lamExp)

    simulation_results = []

    for r in range(R):
        t = expDist.rvs()
        arrTimes = []
        balance = []

        while t < T:
            arrTimes.append(t)
            if bernoulli.rvs(p_income):  # Bernoulli trial to decide if the event is an income or expense
                balance.append(fitGammaDist.rvs())  # incomes
            else:
                balance.append(-fitExpDist.rvs())  # expenses
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
    ci = DescrStatsW(simulation_results).tconfint_mean(alpha=0.05)
    return avg_final_income, ci


# Example 1 where food prices are increased by 20%
df_example_food20 = df[df["Income/Expense"] == "Expense"].copy()
df_example_food20.loc[df_example_food20["Category"] == "Food", "Amount"] *= 1.2
df_example_food20 = df_example_food20['Amount'].copy()


# Example 2 where she does not have a job
df_example_salary = df[df["Income/Expense"] == "Income"].copy()
df_example_salary = df_example_salary[df_example_salary["Category"] != "Salary"]
df_example_salary = df_example_salary['Amount'].copy()
lam_combined_ex2 = (len(df_example_salary) + len(df_expenses)) / total_days # removing salary alters the amount of events
p_income_ex2 = len(df_example_salary) / (len(df_example_salary) + len(df_expenses))


# Example 3 where her regular public transport journeys are twice as long (double the price)
df_example_trains = df[df["Income/Expense"] == "Expense"].copy()
df_example_trains.loc[df_example_trains["Category"] == "Transportation", "Amount"] *= 2
df_example_trains = df_example_trains['Amount'].copy()


# Example 4 where she goes out twice as often (Social life)
df_example_social_life = df[df["Income/Expense"] == "Expense"].copy()
social_life_rows = df_example_social_life[df_example_social_life["Category"] == "Social Life"]
df_example_social_life = pd.concat([df_example_social_life, social_life_rows], ignore_index=True) # duplicate all entries for Social Life
df_example_social_life = df_example_social_life['Amount'].copy()
lam_combined_ex4 = (len(df_incomes) + len(df_example_social_life)) / total_days # adding new social life expenses alters the amount of events
p_income_ex4 = len(df_incomes) / (len(df_incomes) + len(df_example_social_life))


print(f"income simulation w/ gamma, result: {single_simulation(df_incomes, lam_incomes, T, 10000)}")  # log-log: 265k avg, log-normal: 3k avg, gamma: 2,5k avg
print(f"expense simulation w/ exponential, result: {single_simulation(df_expenses, lam_expenses, T, 10000)}") #log-log: 4.9k avg, log-normal: 2.2k avg, expontential: 2.3k avg
print(f"combined simulation, result: {combined_simulation(df_incomes, df_expenses, lam_combined, T, 10000, p_income)}")
print(f"example 1: 20% increased food prices, result: {combined_simulation(df_incomes, df_example_food20, lam_combined, T, 10000, p_income)}")
print(f"example 2: no job income, result: {combined_simulation(df_example_salary, df_expenses, lam_combined_ex2, T, 10000, p_income_ex2)}")
print(f"example 3: double transportation costs, result: {combined_simulation(df_incomes, df_example_trains, lam_combined, T, 10000, p_income)}")
print(f"example 4: twice as many social life events, result: {combined_simulation(df_incomes, df_example_social_life, lam_combined_ex4, T, 10000, p_income_ex4)}")


def check_exponential_fit(interarrivals, alpha=0.05, plot=True): # 95% significance
    interarrivals = np.array(interarrivals)
    interarrivals = interarrivals[interarrivals > 0]

    # Estimate rate parameter lamnda = 1/mean
    lam_hat = 1 / np.mean(interarrivals)

    # Perform a K-test for exponentiality
    ks_stat, p_value = stats.kstest(interarrivals, 'expon', args=(0, 1 / lam_hat))

    # Conclusion
    conclusion = (
        "Fail to reject H_0: data appear exponential"
        if p_value > alpha else
        "Reject H_0: data do NOT appear exponential"
    )

    if plot:
        x = np.linspace(0, np.max(interarrivals), 100)
        plt.figure(figsize=(10, 4))

        # Histogram and fitted exponential PDF
        plt.subplot(1, 2, 1)
        plt.hist(interarrivals, bins=30, density=True, alpha=0.6, label='Data')
        plt.plot(x, lam_hat * np.exp(-lam_hat * x), 'r-', label=f'Exp(lambda={lam_hat:.3f})')
        plt.xlabel("Interarrival Time")
        plt.ylabel("Density")
        plt.title("Histogram + Fitted Exponential PDF")
        plt.legend()

        # Q-Q plot
        plt.subplot(1, 2, 2)
        stats.probplot(interarrivals, dist=stats.expon(scale=1 / lam_hat), plot=plt)
        plt.title("Exponential Q-Q Plot")

        plt.tight_layout()
        plt.show()

    # Return results
    return {
        'lambda_hat': lam_hat,
        'KS_statistic': ks_stat,
        'p_value': p_value,
        'conclusion': conclusion
    }

# check if interarrival times are exponentially distributed
expDist = stats.expon(scale=1/lam_combined)
times = []
t = expDist.rvs()
while t < T:
    times.append(t)
    t += expDist.rvs()

interarrivals = np.diff([0] + times)

# Check if they're exponential
result = check_exponential_fit(interarrivals)
print(result)
