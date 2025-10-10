import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.distributions.empirical_distribution import ECDF

df = pd.read_excel("data/data_assignment.xlsx")
df["Date"] = pd.to_datetime(df["Date"]) # makes sure dates are in datetime format
print(df.head())
# print("")
# print(df.describe())
# print("")

cat_mean_std_df = df.groupby("Category")["Amount"].agg(["mean", "std"])
print(cat_mean_std_df)
print(cat_mean_std_df.iloc[0]['mean'])

plt.figure(figsize=(10, 6))
plt.plot(df["Date"], df["Amount"], 'ob')
plt.show()

plt.figure()
plt.hist(df["Amount"], bins=20, rwidth=0.8, density=True)
plt.show()

M1 = np.mean(df['Amount'])      # first moment
M2 = np.mean(df["Amount"]**2)   # second moment

# Estimates for mu and sigma^2
mu = M1
sigma2 = M2 - M1**2
fitNormDist = stats.norm(mu, np.sqrt(sigma2))

# Add theoretical density
xs = np.arange(np.min(df["Amount"]), np.max(df["Amount"]), 0.1)
ys = fitNormDist.pdf(xs)
plt.plot(xs, ys, color='red')
plt.show()

# Method 2: using Python function ECDF
ecdf = ECDF(df["Amount"])
plt.figure()
plt.step(ecdf.x, ecdf.y, color='black', where='post')
plt.plot(xs, fitNormDist.cdf(xs), color='b')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(df['Amount'], fitNormDist.cdf)
print('KS Test Normal distribution: ' + str(tst1))
# Test statistic: 0.346, P-value: 1.903

# Shapiro-Wilk test for normality
tst2 = stats.shapiro(df['Amount'])
print('Shapiro-Wilk Test: ' + str(tst2))
# Test statistic: 0.371, P-value: 5.418



# fit a gamma distribution
alpha = M1**2 / (M2 - M1**2)
beta = M1 / (M2 - M1**2)
fitGammaDist = stats.gamma(alpha, scale=1/beta)
plt.plot(xs, fitGammaDist.cdf(xs), color='r')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(df["Amount"], fitGammaDist.cdf)
print('KS Test Gamma distribution: ' + str(tst1))




# fit an exponential distribution
lam = 1/M1
fitExpDist = stats.expon(scale=1/lam)
plt.plot(xs, fitExpDist.cdf(xs), color='g')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(df["Amount"], fitExpDist.cdf)
print('KS Test Exponential distribution: ' + str(tst1))




# fit a uniform distribution
# (Careful: we are NOT using the method of moments here!!!!!)
a = min(df["Amount"])
b = max(df["Amount"])
fitUnifDist = stats.uniform(loc=a, scale=b - a) # Careful! look at definition!
plt.plot(xs, fitUnifDist.cdf(xs), color='orange')

# Kolmogorov-Smirnov test
tst1 = stats.kstest(df["Amount"], fitUnifDist.cdf)
print('KS Test Uniform distribution: ' + str(tst1))

# Exercise: use the method of moments to estimate a and b.
# Solution:
a = M1 - np.sqrt(3*(M2-M1**2))
b = M1 + np.sqrt(3*(M2-M1**2))

plt.show()
