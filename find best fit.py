import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
import pandas as pd

df = pd.read_excel("data/data_assignment.xlsx")
df_expenses = df[df["Income/Expense"] == "Expense"].copy()
df_expenses = df_expenses["Amount"].copy()
df_incomes = df[df["Income/Expense"] == "Income"].copy()
df_incomes = df_incomes["Amount"].copy()
df["Date"] = pd.to_datetime(df["Date"]) # makes sure dates are in datetime format
print(df.describe())


def create_ecdf(data):
    M1 = np.mean(data)
    M2 = np.mean(data ** 2)

    mu = M1
    sigma2 = M2 - M1 ** 2

    xs = np.arange(np.min(data), np.max(data), 0.1)

    # ECDF
    ecdf = ECDF(data)
    plt.figure(figsize=(8,6))
    plt.step(ecdf.x, ecdf.y, color='black', where='post', label="Empirical CDF")

    results = []

    # 1. Normal
    fitNormDist = stats.norm(mu, np.sqrt(sigma2))
    plt.plot(xs, fitNormDist.cdf(xs), 'b', label='Normal')
    ks_stat, pval = stats.kstest(data, fitNormDist.cdf)
    results.append(("Normal", ks_stat, pval))

    # 2. Gamma
    alpha = M1 ** 2 / (M2 - M1 ** 2)
    beta = M1 / (M2 - M1 ** 2)
    fitGammaDist = stats.gamma(alpha, scale=1/beta)
    plt.plot(xs, fitGammaDist.cdf(xs), 'r', label='Gamma')
    ks_stat, pval = stats.kstest(data, fitGammaDist.cdf)
    results.append(("Gamma", ks_stat, pval))

    # 3. Exponential
    lam = 1 / M1
    fitExpDist = stats.expon(scale=1/lam)
    plt.plot(xs, fitExpDist.cdf(xs), 'g', label="Exponential")
    ks_stat, pval = stats.kstest(data, fitExpDist.cdf)
    results.append(("Exponential", ks_stat, pval))

    # 4. Uniform
    a, b = np.min(data), np.max(data)
    fitUnifDist = stats.uniform(loc=a, scale=b - a)
    plt.plot(xs, fitUnifDist.cdf(xs), color='orange', label='Uniform')
    ks_stat, pval = stats.kstest(data, fitUnifDist.cdf)
    results.append(("Uniform", ks_stat, pval))

    # 5. Lognormal
    shape, _, scale = stats.lognorm.fit(data)
    fitLogNorm = stats.lognorm(shape, scale=scale)
    plt.plot(xs, fitLogNorm.cdf(xs), 'purple', label='Lognormal')
    ks_stat, pval = stats.kstest(data, fitLogNorm.cdf)
    results.append(("Lognormal", ks_stat, pval))

    # 6. Log-logistic
    c, _, scale = stats.fisk.fit(data)
    fitLogLogis = stats.fisk(c, scale=scale)
    plt.plot(xs, fitLogLogis.cdf(xs), 'brown', label='Log-logistic')
    ks_stat, pval = stats.kstest(data, fitLogLogis.cdf)
    results.append(("Log-logistic", ks_stat, pval))

    # Finalize plot
    plt.legend()
    plt.title("ECDF with Distribution Fits")
    plt.xlabel("x")
    plt.ylabel("CDF")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.hist(data, bins=15, density=True, alpha=1, color='gray', label='Histogram')

    plt.plot(xs, fitNormDist.pdf(xs), 'b', label='Normal')
    plt.plot(xs, fitGammaDist.pdf(xs), 'r', label='Gamma')
    plt.plot(xs, fitExpDist.pdf(xs), 'g', label='Exponential')
    plt.plot(xs, fitUnifDist.pdf(xs), color='orange', label='Uniform')
    plt.plot(xs, fitLogNorm.pdf(xs), 'purple', label='Lognormal')
    plt.plot(xs, fitLogLogis.pdf(xs), 'brown', label='Log-logistic')

    plt.legend()
    plt.title("Histogram with Fitted Distributions")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.show()

    # Show KS test results
    df = pd.DataFrame(results, columns=["Distribution", "KS Statistic", "p-value"])
    df = df.sort_values(by="KS Statistic")
    print(df)

    # Best distribution
    best = df.iloc[0]
    print("\nBest fit: {} (KS Stat = {:.4f}, p = {:.4f})".format(best['Distribution'], best['KS Statistic'], best['p-value']))

create_ecdf(df_incomes)
create_ecdf(df_expenses)
