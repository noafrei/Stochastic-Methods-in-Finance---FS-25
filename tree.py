from typing import List
import numpy as np
from time import time


def add_children(node: List, up : float, down : float, max_depth : int):
    # layer of the tree is as follows:
    # [depth, [prices], [avg prices]]
    depth = node[0]
    if depth == max_depth:
        return

    prices = node[1]
    avg_prices = node[2]

    up_prices = prices * up
    down_prices = prices * down

    price_sums = depth * avg_prices

    up_averages = (price_sums + up_prices) / (depth + 1)
    down_averages = (price_sums + down_prices) / (depth + 1)

    node[0] += 1
    node[1] = np.concatenate((up_prices, down_prices))
    node[2] = np.concatenate((up_averages, down_averages))
    add_children(node, up, down, max_depth)


tree = [0, np.array([100]), np.array([100])]
std = 0.2713948019202162
up_factor = np.exp(std * np.sqrt(1/25))
down_factor = np.exp(-std * np.sqrt(1/25))
max_depth = 25

strike_price = 120

start = time()
add_children(tree, up_factor, down_factor, max_depth)
end = time()
print(f"Success! Took {round(end - start, 3)} seconds")

print(tree)

payoffs = tree[2] - strike_price
payoffs = payoffs.clip(0)

print(payoffs)
