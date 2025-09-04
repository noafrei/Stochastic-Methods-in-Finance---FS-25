###############################################################################################################################################

# Stochastic Methods in Finance: Group Assignement - Pricing Asian Call Options of Microsoft corporation using Binomial Pricing Model
#
# Import data: Microsoft_2015-2025.csv - data downloaded from https://www.nasdaq.com/ (NASDAQ: MSFT)
# Data description: The dataset contains the daily closing prices of Microsoft Corporation (MSFT) from 2015 to 2025.

# Task: Price an Asian Call Option of Microsoft Corporataion, using a Binomial Tree, do robustness checks, and use also normal approximation.
# Group Memebers: Luca Boschung, Ludovico Sbarra, Odile Zimmerman, Noa Diego Frei

# Instructions: by running this script, you will be able to price an Asian Call Option of Microsoft Corporation using a Binomial Tree and do 
#               robustness checks. You can change the parameters to test for different scenarios. To properly run it you should have the data,
#               and the .py file in the same folder


################################################################################################################################################


##########################################################################################################
# Importing the libraries
##########################################################################################################

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import norm
import pandas as pd
import os

import matplotlib
matplotlib.use('qtagg')  # or 'QtAgg' depending on your Qt installation


##########################################################################################################
# Building the functions needed to price the Asian call option
##########################################################################################################
    


##################################################################################################
# Function: add_children(node, up, down, max_depth)

# Description: Builds a binomial tree of stock prices up to max_depth, tracking the number of 
#              up‐moves and the cumulative sum of prices so that each path’s average price can 
#              be computed! Recursion is used to build the tree layer by layer
##################################################################################################

def add_children(node: List, up: float, down: float, max_depth: int):
    # Unpack current depth
    depth = node[0]

    # Base case: if we’re at the maximum depth, convert sums to averages and stop.
    if depth == max_depth:
        node[3] = node[3] / (max_depth+1)
        return

    # Otherwise, take the arrays of current prices and sums
    prices     = node[2]  # shape = (2^depth,)
    price_sums = node[3]  # shape = (2^depth,)

    # Compute the next layer of prices for an up-move and down-move
    up_prices   = prices * up
    down_prices = prices * down

    # Add those new prices to our running sums
    up_sums   = price_sums + up_prices
    down_sums = price_sums + down_prices

    # Increment depth
    node[0] += 1

    # Now overwrite the node’s arrays by concatenating up‐branch then down‐branch
    node[2] = np.concatenate((up_prices, down_prices))  # new prices array 
    node[3] = np.concatenate((up_sums,   down_sums))    # new sums array 

    # Duplicate the existing ups array so we can add one for the up-move half
    node[1] = np.concatenate((node[1], node[1]))
    ups = np.zeros(len(node[1]), dtype=node[1].dtype)
    # Mark “1” in the first half of the new array, indicating those paths just did an up-move
    ups[:len(node[1])//2] = 1
    # Update the ups count
    node[1] += ups

    # Recurse to build out the next layer
    add_children(node, up, down, max_depth)


##################################################################################################
# Function: price_asian_call_option(leaves, r, q, K)

# Description: Calculates the price of an Asian call option using the leaves of the binomial tree.
#              The function computes the expected payoff of the option at maturity,
#              discounted back to the present value using the risk-free rate (backward induction)
##################################################################################################

def price_asian_call_option(leaves, r: float, q: float, K: float) -> float:
    # Unpack the number of time steps (depth) from the leaves data
    depth = leaves[0]
    # Array of up-move counts for each path
    up_counts = leaves[1]
    # Compute down-move counts per path as total steps minus up-moves
    down_counts = depth - up_counts
    # Array of precomputed average prices along each path
    avg_prices = leaves[3]

    # Compute the probability of each path under the risk-neutral measure:
    # q^number_of_ups * (1 - q)^number_of_downs
    probs = (q ** up_counts) * (1 - q) ** down_counts

    # Compute the payoff at each leaf: max(average_price − strike, 0)
    payoffs = np.maximum(avg_prices - K, 0)
    # Compute the expected payoff as the dot product of probabilities and payoffs
    expectation = np.dot(probs, payoffs)

    # Discount the expected payoff back to present value using (1 + r)^(-depth)
    discount = (1 + r) ** -depth
    return expectation * discount


##################################################################################################
# Function: approximate_asian_call_option(leaves, r, q, K)

# Description: Calculates the price of an Asian call option using a normal approximation.
#              The function computes the expected payoff of the option at maturity,
#              discounted back to the present value using the risk-free rate (backward induction)
#              using the normal distribution.
#              The function computes the mean and standard deviation of the average prices,
#              and then uses the normal distribution to calculate the probabilities of each 
#              average price
##################################################################################################

def approximate_asian_call_option(leaves, r: float, q: float, K: float) -> float:
    # Unpack the number of time steps (depth) from the leaves data
    depth = leaves[0]
    # Array of up‐move counts for each path
    up_counts = leaves[1]
    # Array of precomputed average prices along each path
    avg_prices = leaves[3]

    # Compute the mean and standard deviation of the binomial(up/down) distribution
    mean_up = depth * q                       # expected number of up‐moves
    std_up = np.sqrt(depth * q * (1 - q))     # standard deviation

    # Evaluate the normal PDF at each observed up_count
    normal_probs = norm.pdf(
        up_counts,
        loc=mean_up,
        scale=std_up
    )
    # Normalize so probabilities sum to 1
    normal_probs /= np.sum(normal_probs)

    # Compute the payoff for each path: max(average_price − strike, 0)
    payoffs = np.maximum(avg_prices - K, 0)
    
    # Compute the expected payoff under the approximate normal probability weights
    expectation = np.dot(normal_probs, payoffs)

    # Discount the expected payoff back to present value 
    discount = (1 + r) ** -depth
    return expectation * discount

##################################################################################################
# Function: print_option_details(s0, k, r_ann, sig, t_years, steps, u, d, q_prob, price, approx)

# Description: Prints the parameters and calculated price in a formatted way.
##################################################################################################
def print_option_details(s0, k, r_ann, sig, t_years, steps, u, d, q_prob, price, approx):
    
    dt = t_years / steps
    r_step = r_ann / steps # Or r_ann * dt if r_ann is effective annual

    print("="*50)
    print(" European Arithmetic Asian Call Option Pricing")
    print(" Binomial Tree (Non-Recombining) Results")
    print("="*50)
    print("Input Parameters:")
    print(f"  Initial Stock Price (S0):{s0:>20.2f}")
    print(f"  Strike Price (K):        {k:>20.2f}")
    print(f"  Annual Risk-Free Rate (r): {r_ann:>18.4f} ({r_ann:.2%})")
    print(f"  Volatility (sigma):      {sig:>20.4f}")
    print(f"  Time to Maturity (T):    {t_years:>20.2f} years")
    print(f"  Number of Steps (N):     {steps:>20d}")
    print("\nDerived Model Parameters:")
    print(f"  Time Step (dt):          {dt:>20.6f}")
    print(f"  Interest Rate per Step:  {r_step:>20.6f}")
    print(f"  Up Factor (u):           {u:>20.6f}")
    print(f"  Down Factor (d):         {d:>20.6f}")
    print(f"  Risk-Neutral Prob (q):   {q_prob:>20.6f}")
    print("-"*50)
    print("Calculated Result:")
    print(f"  Option Price:            {price:>20.4f}")
    print(f"  Aprx. Option Price:      {approx:>20.4f}")
    print("="*50)



##################################################################################################
# Function: robustness_check(price_0, strikes, T_years, periods, sigmas, r_annual_list)

# Description: Iterates over different volatilities, rates and strike prices and computes the call 
#              option price.
#              Returns a DataFrame with columns: sigma, r_annual, strike, price_exact, 
#              price_approx (all combinations)
##################################################################################################
def run_robustness_check(
    price_0: float,
    strikes: List[float],
    T_years: float,
    periods: int,
    sigmas: List[float],
    r_annual_list: List[float]
) -> pd.DataFrame:
    
    results = []
    
    for sigma in sigmas:
        # precompute up/down
        u = np.exp(sigma * np.sqrt(T_years/periods)) #need to ask what is correct
        d = np.exp(-sigma * np.sqrt(T_years/periods))
        #u = np.exp(sigma * np.sqrt(1/25))
        #d = np.exp(-sigma * np.sqrt(1/25))
        
        for r_ann in r_annual_list:
            r_act = r_ann * T_years
            gf    = 1 + r_act/periods
            q     = (gf - d) / (u - d)
            
            for K in strikes:
                # build tree
                tree = [0, np.array([0]), np.array([price_0]), np.array([price_0])]
                add_children(tree, u, d, periods)
                
                # price
                exact  = price_asian_call_option(tree, r_act/periods, q, K)
                approx = approximate_asian_call_option(tree, r_act/periods, q, K)
                
                results.append({
                    'sigma':       sigma,
                    'r_annual':    r_ann,
                    'strike':      K,
                    'price_exact': exact,
                    'price_approx': approx
                })
    df = pd.DataFrame(results)
    
    # Round to 3 decimal places
    df[['sigma', 'price_exact', 'price_approx', 'strike']] = df[['sigma', 'price_exact', 'price_approx', 'strike']].round(3)
    return df


##########################################################################################################
# Functions for interpreting the results, plots, etc.
##########################################################################################################


##################################################################################################
# Function: plot_end_leaves_histogram(leaves, num_bins)

# Description: Plots histograms for the final stock prices and final average prices
##################################################################################################
def plot_end_leaves_histogram(leaves, num_bins: int):
    
    if not leaves or len(leaves) < 4:
        print("Error: Invalid 'leaves' data structure provided.")
        return

    final_prices = leaves[2]
    avg_prices = leaves[3]
    depth = leaves[0]

    if final_prices is None or avg_prices is None or len(final_prices) == 0:
        print("Error: No price data found in the leaves.")
        return

    print(f"\nPlotting histograms for {len(final_prices)} leaf nodes (Depth={depth})...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Create a figure with 2 subplots

    # Histogram for Final Stock Prices
    axes[0].hist(final_prices, bins=num_bins, color='blue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Distribution of Final Stock Prices (N={depth})')
    axes[0].set_xlabel('Final Stock Price')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)

    # Histogram for Average Path Prices
    axes[1].hist(avg_prices, bins=num_bins, color='green', edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Distribution of Average Path Prices (N={depth})')
    axes[1].set_xlabel('Average Stock Price Along Path')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()

    
##################################################################################################
# Function: plot_path_count_heatmap(leaves)

# Description: Plots a heatmap of the count of paths by number of up-moves (used only in 
#               the base case)
##################################################################################################
def plot_path_count_heatmap(leaves):
    depth = leaves[0]
    up_counts = leaves[1]

    # Count how many paths end with k up-moves for k in [0..depth]
    counts = np.bincount(up_counts, minlength=depth + 1)

    # Prepare a 2D array for heatmap (1 row, depth+1 columns)
    heatmap_data = counts.reshape(1, -1)

    plt.figure()
    plt.imshow(heatmap_data, aspect='auto', origin='lower')
    plt.yticks([0], ['Count'])  # single row label
    plt.xticks(ticks=np.arange(depth + 1), labels=np.arange(depth + 1))
    plt.xlabel('Number of Up-Moves (k)')
    plt.title('Heatmap of Path Counts by Up-Move Count')
    plt.colorbar(label='Path Count')
    plt.tight_layout()
    plt.show()


##################################################################################################
# Function: plot_price_vs_volatility_at_r(df, r_annual: float)

# Description: Plots option price vs. volatility for a fixed interest rate (r_annual)
##################################################################################################
def plot_price_vs_volatility_at_r(df, r_annual: float):

    # Filter for the chosen rate
    data = df[df['r_annual'] == r_annual]
    if data.empty:
        raise ValueError(f"No data for r_annual={r_annual}")

    plt.figure()
    # Plot one line per strike
    for K, group in data.groupby('strike'):
        plt.plot(group['sigma'], group['price_exact'],
                 marker='o', label=f'K={K:.2f}')
    plt.xlabel('Volatility (σ)')
    plt.ylabel('Exact Asian-Call Price')
    plt.title(f'Price vs. Volatility at r={r_annual:.2%}')
    plt.legend()
    plt.tight_layout()
    plt.show()


##################################################################################################
# Function: plot_3d_surface_price(df, r_annual, value_col='price_exact')

# Description: Plots a 3D surface of option price vs. volatility and strike (K) at fixed r
##################################################################################################
def plot_3d_surface_price(df, r_annual, value_col='price_exact'):
   
    data = df[df['r_annual'] == r_annual]
    pivot = data.pivot(index='sigma', columns='strike', values=value_col)
    σ = pivot.index.values
    K = pivot.columns.values
    Z = pivot.values
    X, Y = np.meshgrid(K, σ)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, label='Exact Call Price')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Volatility (σ)')
    ax.set_zlabel('Exact Call Price')
    ax.set_title(f'3D Surface: {value_col} at r={r_annual:.2%}')
    plt.tight_layout()
    plt.show()



##########################################################################################################
# Main function
##########################################################################################################


if __name__ == "__main__":
    
    
    ##########################################################################################################
    # Importing the data and cleaning it
    ##########################################################################################################
    
    #import the data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path  = os.path.join(script_dir, "./data/Microsoft_2015-2025.csv")
    data = pd.read_csv(csv_path)


    #Cleaning the data

    # Converting the date column to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    
    # Setting the date as the index
    data.set_index(["Date"], inplace=True)
    
    #renaming and polishing the columns 
    data = data.rename(columns={'Close/Last': 'Close'})
    data['Close'] = pd.to_numeric(data['Close'].str.replace('$', '').str.replace(',', ''), errors='coerce')


    ##########################################################################################################
    # Do some calculations for later, volatility, up and down factors, etc. 
    ##########################################################################################################
    
    data['log_returns'] = np.log(data['Close']).diff()
    std = data['log_returns'].std() * np.sqrt(250)
    #print(std)

    price_0 = data.loc['2025-04-28', 'Close']


    ups = 0
    tree = [0, np.array([ups]), np.array([price_0]), np.array([price_0])]
    sigma = std # annualized volatility

    periods = 25 # number of steps in the tree, given in the instructions
    n=periods
    
    T_years = 0.5 #T is always in years, so 0.5 = 6 months, T is the maturity of the option
    up_factor = np.exp(sigma * np.sqrt(T_years/n)) #formula given in the instructions
    down_factor = np.exp(-sigma * np.sqrt(T_years/n)) #formula given in the instructions

    r_annual = 0.01
    r_actual = r_annual * T_years # the total simple‐interest rate earned over the life of the option
    growth_factor = 1 + r_actual / periods
    risk_neutral_prob = (growth_factor - down_factor) / (up_factor - down_factor)

    #we can set as we want - we deiced that the default case is at the money
    strike_price = price_0


    ################################################################################################################################################################################
    # Beginning of Pricing Part - default parameters: S_0=391.16 (from data), K=S_0, rfree_annual=0.01, sigma=0.27(computed), T=0.5, N=25, most of them are given in the instructions
    ################################################################################################################################################################################
    
    #Pricing with the default parameters:
    
    start = time()
    add_children(tree, up_factor, down_factor, n)
    end = time()
    print(f"Success! Building tree took {round(end - start, 3)} seconds")

    call_price = price_asian_call_option(tree, r_actual/n, risk_neutral_prob, strike_price)
    aprx_call_price = approximate_asian_call_option(tree, r_actual/n, risk_neutral_prob, strike_price)

    print_option_details(
            s0=price_0,
            k=strike_price,
            r_ann=r_annual,
            sig=sigma,
            t_years=T_years,
            steps=periods,
            u=up_factor,
            d=down_factor,
            q_prob=risk_neutral_prob,
            price=call_price,
            approx=aprx_call_price
        )
    
    #############################################################################################################################################################
    # Robustness check - play around with the parameters
    #############################################################################################################################################################
    
    # Robustness check:
    vol_grid = [sigma*0.6, sigma*0.7,sigma*0.8, sigma, sigma*1.2, sigma*1.4, sigma*1.6]        # ±60% around your estimated σ
    rate_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]            # 0.5%, 1%, 2% annual
    strikes = [                # we only choose 3 strikes for the robustness check, would be better to have more
        0.8 * strike_price,    # deep out-of-the-money
        strike_price,          # at-the-money
        1.2 * strike_price     # deep in-the-money
    ]


    df = run_robustness_check(
        price_0=price_0,
        strikes=strikes,
        T_years=T_years,
        periods=n,
        sigmas=vol_grid,
        r_annual_list=rate_grid
    )
    print(df.to_string(index=False))

    # Export to robustness table to csv file
    out_csv = os.path.join(script_dir, 'robustness_results.csv')
    df.to_csv(out_csv, index=False)

    
#############################################################################################################################################################
# Plots and result interpretation - uncomment to plot them
#############################################################################################################################################################
    

    plot_end_leaves_histogram(tree, num_bins=100)

    #plot_path_count_heatmap(tree)
    '''
    for r in rate_grid:
    plot_path_count_heatmap(tree)
    plot_price_vs_volatility_at_r(df, r_annual=r)
    plot_3d_surface_price(df, r_annual=r)
    '''


#############################################################################################################################################################
# End of Script
#############################################################################################################################################################
