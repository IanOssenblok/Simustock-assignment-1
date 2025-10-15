## How to Run the Code

1. **Dataset setup**  
   Place the dataset inside a folder named `data`, located in the same directory as the following files:   
   - `interarrival_time_eda.py`  
   - `transaction_amount_eda.py`  
   - `final_distribution_fitting.py`  
   - `final_simulation.py`

2. **Generate EDA graphs**  
   Run the `interarrival_time_eda.py` and `transaction_amount_eda.py` scripts to produce all exploratory data analysis (EDA) graphs.

3. **Perform Distribution Fitting**  
Run the `find_best_fit.py` script. This will create the ECDFs for the distribution fitting and it will print the `ks-statistic` for every distribution.
The first ECDF and histogram belong to the income data, and the latter belong to the expense data.

4. **Run Simulations**  
Run the `everything_simulation.py` script.  
This will run simulations for the income, expense and combined separately. 
Each will plot a graph of the first simulation, and will then return the average and the 95% confidence intervals. 
Then it also runs simulations for our 4 scenarios. 
Finally it will make a plot of the interarrival times of the combined simulation to check if the times are actually exponentially distributed or not. 