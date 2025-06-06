import numpy as np
import matplotlib.pyplot as plt
import random




def plot_CIs(
    CI_list, 
    n: int = 20, 
    seed: int = 0, 
    title: str = "Random Confidence Intervals"):
    """
    Plot randomly selected n confidence intervals from a list of dicts.
    Each dict should contain:
        - "lower": np.ndarray
        - "upper": np.ndarray
        - "covers?": bool

    Averages bounds if multidimensional.

    Parameters:
    - CI_list: list of dicts with keys "lower", "upper", and "covers?"
    - n: number of intervals to sample
    - seed: random seed
    - title: plot title
    """
    if n > len(CI_list):
        raise ValueError(f"Requested {n} intervals, but only {len(CI_list)} available.")

    rng = np.random.default_rng(seed=seed)
    sampled = rng.sample(CI_list, n)

    avg_lowers = []
    avg_uppers = []

    for d in sampled:
        lower = d["lower"]
        upper = d["upper"]
        avg_lowers.append(np.mean(lower))
        avg_uppers.append(np.mean(upper))

    centers = [(l + u) / 2 for l, u in zip(avg_lowers, avg_uppers)]
    half_widths = [(u - l) / 2 for l, u in zip(avg_lowers, avg_uppers)]

    plt.errorbar(range(n), centers, yerr=half_widths, fmt='o', capsize=5)
    plt.xticks(ticks=range(n), labels=[f"CI {i}" for i in range(n)], rotation=45, ha='right')
    plt.ylabel("Confidence Interval")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    
    

