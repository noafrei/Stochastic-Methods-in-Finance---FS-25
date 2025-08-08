from typing import List
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import norm

import matplotlib
matplotlib.use('qtagg')  # or 'QtAgg' depending on your Qt installation

def add_children(node: List, up : float, down : float, max_depth : int):
    # layer of the tree is as follows:
    # [depth, [ups], [prices], [avg prices]]
    depth = node[0]
    if depth == max_depth:
        # convert sum to average
        node[3] = node[3] / (max_depth+1)
        return

    prices = node[2]
    price_sums = node[3]

    up_prices = prices * up
    down_prices = prices * down

    up_sums = price_sums + up_prices
    down_sums = price_sums + down_prices

    node[0] += 1
    node[2] = np.concatenate((up_prices, down_prices))
    node[3] = np.concatenate((up_sums, down_sums))
    node[1] = np.concatenate((node[1], node[1]))

    ups = np.zeros(len(node[1]), dtype=node[1].dtype)
    ups[:len(node[1])//2] = 1

    node[1] += ups

    add_children(node, up, down, max_depth)

def price_asian_call_option(leaves, r: float, q: float, K: float) -> float:
    depth = leaves[0]
    up_counts = leaves[1]       # Number of up moves in each path
    down_counts = depth - up_counts  # Number of down moves in each path
    avg_prices = leaves[3]

    probs = (q ** up_counts) *  (1 - q) ** down_counts

    payoffs = np.maximum(avg_prices - K, 0)
    expectation = np.dot(probs, payoffs)

    discount = (1 + r) ** -depth
    return expectation * discount

def approximate_asian_call_option(leaves, r: float, q: float, K: float) -> float:
    depth = leaves[0]
    up_counts = leaves[1]
    avg_prices = leaves[3]

    mean_up = depth * q
    std_up = np.sqrt(depth * q * (1-q))

    normal_probs = norm.pdf(up_counts, loc=mean_up, scale=std_up)
    normal_probs = normal_probs / np.sum(normal_probs)

    payoffs = np.maximum(avg_prices - K, 0)
    expectation = np.dot(normal_probs, payoffs)

    discount = (1 + r) ** -depth
    return expectation * discount

def print_option_details(s0, k, r_ann, sig, t_years, steps, u, d, q_prob, price, approx):
    """Prints the parameters and calculated price in a formatted way."""
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

def plot_end_leaves_histogram(leaves, num_bins: int):
    """
    Plots histograms for the final stock prices and final average prices
    at the end leaves of the binomial tree.

    Args:
        leaves: The tree structure after add_children has run.
                Expected format: [depth, [ups], [final_prices], [avg_prices]]
        num_bins: The number of bins to use in the histograms.
    """
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
    plt.savefig("./histogram.pdf")

price_0 = 391.16
ups = 0
tree = [0, np.array([ups]), np.array([price_0]), np.array([price_0])]
sigma = 0.2713948019202162
T_years = 0.5
up_factor = np.exp(sigma * np.sqrt(T_years/25))
down_factor = np.exp(-sigma * np.sqrt(T_years/25))
periods = 25

r_annual = 0.01
r_actual = r_annual * T_years
growth_factor = 1 + r_actual / periods
risk_neutral_prob = (growth_factor - down_factor) / (up_factor - down_factor)

strike_price = 400

start = time()
add_children(tree, up_factor, down_factor, periods)
end = time()
print(f"Success! Building tree took {round(end - start, 3)} seconds")

call_price = price_asian_call_option(tree, r_actual/periods, risk_neutral_prob, strike_price)
aprx_call_price = approximate_asian_call_option(tree, r_actual/periods, risk_neutral_prob, strike_price)

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

avg_prices = tree[3]
print(np.mean(avg_prices), np.std(avg_prices))
number_of_bins = 100
plot_end_leaves_histogram(tree, num_bins=number_of_bins)
