To run the code:

1. Run find_best_fit.py, this will create the ecdf's for the distribution fitting and it will print the ks-statistic for every distribution.
The first ECDF and histogram belong to the income data, and the latter belong to the expense data. 

2. Run everything_simulation.py, this will run simulations for the income, expense and combined separately. 
Each will plot a graph of the first simulation, and will then return the average and the 95% confidence intervals. 
Then it also runs simulations for our 4 example situations. 
Finally it will make a plot of the interarrival times of the combined simulation to check if the times are actually exponentially distributed or not. 

3. Run eda_expenses_notebook.ipynb and time_exploration.py for all the EDA graphs.