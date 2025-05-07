import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# Function to build the binomial tree for stock prices
def build_stock_tree(S_0, u, d, n):
    tree = [[S_0]]  # Root node with the initial stock price
    
    # Fill the tree with stock prices for each level
    for i in range(1, n + 1):
        level = []
        for j in range(2 ** i):  # Each level has 2^i nodes
            up_moves = bin(j).count('1')  # Count the number of up moves
            down_moves = i - up_moves
            price = S_0 * (u ** up_moves) * (d ** down_moves)
            level.append(price)
        tree.append(level)

    #print the tree for debugging
    #for i in range(len(tree)):
        #print(f"Level {i}: {tree[i]}")
        
    return tree

# Function to calculate the average price at each node in the binomial tree
def build_average_tree(tree, n):
    averages = []
    averages.append(tree[0][0])# Root node has the average equal to the initial price
    for i in range(1, n + 1):
        level_averages = []
        for j in range(2 ** i):
            # The average price at this node is the average of the stock prices along the path to this node
            path_prices = []
            # Start with the current node and move up the tree to collect stock prices
            current_level = i
            current_index = j
            while current_level >= 0:
                path_prices.append(tree[current_level][current_index])
                # Move to the parent node at the previous level
                current_level -= 1
                current_index = current_index // 2  # to to one level down

            # Calculate the average price for this node
            average_price = np.mean(path_prices)
            level_averages.append(average_price)
        averages.append(level_averages)
    
    
    #for i in range(len(averages)):
        #print(f"Level {i}: {averages[i]}")  # Debugging: print the average prices at each level
        
    return averages


# Function to calculate the payoff at the terminal nodes based on the average prices
def calculate_payoffs_at_terminal(averages, K, n):
    payoffs = []
    for j in range(2 ** n):
        avg_price = averages[n][j]  # Average price at the final node
        payoff = max(avg_price - K, 0)  # Payoff for Asian call option
        payoffs.append(payoff)
    
    return payoffs

# Function to perform backward induction to calculate the option price
def backward_induction(tree, payoffs, p_risk_neutral, q_risk_neutral, r, n):
    # Initialize option tree with the correct number of nodes per level
    option_tree = [[] for _ in range(n + 1)]  # Create a list of n+1 empty lists for each level
    
    #debug print r daily
    print(f"r_daily: {r}")  # Debugging: print the risk-free rate

    # Step 1: Initialize the option values at the terminal nodes using the payoffs
    # Fill the last level (terminal nodes) with the payoffs
    option_tree[n] = payoffs  # Payoffs are the values at the last level

    # Step 2: Perform backward induction: Calculate option prices at earlier nodes
    for i in range(n - 1, -1, -1):  # Loop backward from level n-1 to level 0
        for j in range(2 ** i):  # Loop through each node at the current time step
            # Calculate the expected payoff at this node from the future
            expected_payoff = (p_risk_neutral * option_tree[i + 1][j + 1] + q_risk_neutral * option_tree[i + 1][j])
            
            #print(f"Expected Payoff at node ({i}, {j}): {expected_payoff}")  # Debugging: print expected payoffs
            
            # Discount the expected payoff to the present

            option_tree[i].append(np.exp(-r * (i + 1)) * expected_payoff) # Apply continuous discounting 

    # The price of the option at the root node (t=0) is the final result
    return option_tree[0][0]


# Final function that takes parameters and returns the Asian option price
def asian_option_price_binomial(S_0, K, T, n, u, d, r):
    
    dt = T / n  # Time step size
    
    # Calculate risk-neutral probabilities
    p_risk_neutral = (np.exp(r * dt) - d) / (u - d)
    q_risk_neutral = 1 - p_risk_neutral
    
    #debugging
    print(f"p_risk_neutral: {p_risk_neutral}, q_risk_neutral: {q_risk_neutral}")

    
    # Step 1: Build the stock price tree
    tree = build_stock_tree(S_0, u, d, n)
    
    # Step 2: Build the average price tree
    averages = build_average_tree(tree, n)
    
    #debugging
    print("Exit average function")
    

    # Step 3: Calculate payoffs at the terminal nodes
    payoffs = calculate_payoffs_at_terminal(averages, K, n)
    
    #debugging
    print("Exit payoff at termninal function")
    
    # Step 4: Perform backward induction to calculate the option price
    option_price = backward_induction(tree, payoffs, p_risk_neutral, q_risk_neutral, r, n)
    
    print("Exit backward induction function")
    
    return option_price


def main():
    # Read the data
    data = pd.read_csv("/Users/noadiegofrey/Documents/Studio/HSG/Study Material/Bachelor/4. Semester/Stochastic Methodes in Finance/Assignement/Microsoft_2015-2025.csv")
    
    # Convert the date column to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index(["Date"], inplace=True)
    
    # We start from the last 5 years of data
    data = data[data.index >= '2020-01-01']
    
    # Remove dollar signs and commas, then convert to numeric
    data['Close/Last'] = data['Close/Last'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    data['Open'] = data['Open'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    data['High'] = data['High'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    data['Low'] = data['Low'].replace({'\$': '', ',': ''}, regex=True).astype(float)
    data['Volume'] = data['Volume'].replace({',': ''}, regex=True).astype(int)
    
    # Calculate the daily return (percentage change)
    data['Return'] = data['Close/Last'].pct_change()
    
    # Handle NaN value (which will be in the first row due to pct_change)
    data['Return'] = data['Return'].fillna(0)
    
    # Set the parameters for the Asian call option
    S_0 = data.loc['2025-04-28', 'Close/Last']  # Initial stock price (on April 28, 2025)
    
    # Parameters for the binomial model
    n = 5  # Number of steps
    T = 125  # Time to maturity (in days)
    dt = T / n  # Time step
    
    # Volatility and drift
    sigma_daily = data['Return'].std()  # Daily standard deviation (volatility)
    sigma_5days = sigma_daily * np.sqrt(dt)  # 5-day volatility
    r_annual = 0.01  # Annual risk-free rate
    r_daily = (1 + r_annual)**(1/250) - 1  # Convert annual rate to daily rate
    r_5days = (1 + r_daily) ** dt - 1  # 5-day risk-free rate
    
    # Debugging: print the risk-free rate for 5 days
    print(f"r_5days: {r_5days}") 
    
    K = S_0  # Strike price equal to the initial stock price

    # Up/down factors for the binomial model
    u = np.exp(sigma_5days)  # Up factor
    d = np.exp(-sigma_5days)  # Down factor
    
    # Call the function to calculate the Asian call option price
    asian_option_price = asian_option_price_binomial(S_0, K, T, n, u, d, r_5days)
    
    # Print the result
    print(f"Asian Call Option Price: {asian_option_price:.2f}")
    

# Ensure the main function runs only when the script is executed directly
if __name__ == "__main__":
    main()

