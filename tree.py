from typing import List
import numpy as np
from time import time

def add_children(node: List, up : float, down : float, max_depth : int):
    # layer of the tree is as follows:
    # [depth, [ups], [prices], [avg prices]]
    depth = node[0]
    if depth == max_depth:
        return

    prices = node[2]
    avg_prices = node[3]

    up_prices = prices * up
    down_prices = prices * down

    price_sums = depth * avg_prices

    up_averages = (price_sums + up_prices) / (depth + 1)
    down_averages = (price_sums + down_prices) / (depth + 1)

    node[0] += 1
    node[2] = np.concatenate((up_prices, down_prices))
    node[3] = np.concatenate((up_averages, down_averages))
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

def print_option_details(s0, k, r_ann, sig, t_years, steps, u, d, q_prob, price):
    """Prints the parameters and calculated price in a formatted way."""
    dt = t_years / steps
    r_step = r_ann / steps # Or r_ann * dt if r_ann is effective annual

    print("="*50)
    print(" European Arithmetic Asian Call Option Pricing")
    print(" Binomial Tree (Non-Recombining) Results")
    print("="*50)
    print("Input Parameters:")
    print(f"  Initial Stock Price (S0): {s0:>20.2f}")
    print(f"  Strike Price (K):       {k:>20.2f}")
    print(f"  Annual Risk-Free Rate (r):{r_ann:>18.4f} ({r_ann:.2%})")
    print(f"  Volatility (sigma):     {sig:>20.4f}")
    print(f"  Time to Maturity (T):   {t_years:>20.2f} years")
    print(f"  Number of Steps (N):    {steps:>20d}")
    print("\nDerived Model Parameters:")
    print(f"  Time Step (dt):         {dt:>20.6f}")
    print(f"  Interest Rate per Step: {r_step:>20.6f}")
    print(f"  Up Factor (u):          {u:>20.6f}")
    print(f"  Down Factor (d):        {d:>20.6f}")
    print(f"  Risk-Neutral Prob (q):  {q_prob:>20.6f}")
    print("-"*50)
    print("Calculated Result:")
    print(f"  Option Price:           {price:>20.4f}")
    print("="*50)


price_0 = 391.16
ups = 0
tree = [0, np.array([ups]), np.array([price_0]), np.array([price_0])]
sigma = 0.2713948019202162
up_factor = np.exp(sigma * np.sqrt(1/25))
down_factor = np.exp(-sigma * np.sqrt(1/25))
periods = 25
T_years = 1.0

r_annual = 0.05
growth_factor = 1 + r_annual / periods
risk_neutral_prob = (growth_factor - down_factor) / (up_factor - down_factor)

strike_price = 400

start = time()
add_children(tree, up_factor, down_factor, periods)
end = time()
print(f"Success! Building tree took {round(end - start, 3)} seconds")

call_price = price_asian_call_option(tree, r_annual/periods, risk_neutral_prob, strike_price)

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
        price=call_price
    )

