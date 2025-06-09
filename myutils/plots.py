import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random




def plot_CIs(
    *,
    CI_dict: dict, 
    truth: float,
    n: int = 5, 
    seed: int = 0, 
    title: str = ""):
    """
    Plot randomly selected n confidence intervals from a dict of lists of dicts.
    Each dict should contain:
        - "linear": list
        - "random_forest": list
        - "kernel": list
        - "xgboost": list
        - "neural_net": list
    Each value is a list of subdicts,
    where each subdict should contain:
        - "lower": np.ndarray
        - "upper": np.ndarray
        - "covers?": bool

    Averages bounds if multidimensional.

    Parameters:
    - CI_dict: dict of lists of dicts
    - n: number of intervals to sample
    - seed: random seed
    - title: plot title
    """
    rng = np.random.default_rng(seed=seed)
    colors = sns.color_palette('pastel', n_colors=len(CI_dict))
    
    total_plots = n * len(CI_dict)
    bar_height = 0.8 
    y_base = 0
    
    plt.figure(dpi=600)
    plt.figure(figsize=(8, total_plots * 0.25 + 2))  # adjust height for visibility
    
    for i, (CI_list_name, CI_list) in enumerate(CI_dict.items()):
        if n > len(CI_list):
            raise ValueError(f"Requested {n} intervals, but {CI_list_name} has only {len(CI_list)} available.")
            
        sampled = rng.choice(CI_list, n, replace=False)

        for j, d in enumerate(sampled):
            lower = np.mean(d["lower"])
            upper = np.mean(d["upper"])
            width = upper - lower
            y = y_base + j
            
            plt.broken_barh(
                [(lower, width)],           # (x_start, width)
                (y - bar_height / 2, bar_height),  # (y_start, height)
                facecolors=colors[i],
                edgecolors='none',
                label=CI_list_name if j == 0 else None
            )

        y_base += n
        
    # Add vertical line for truth
    plt.axvline(
        x=truth, 
        color='black', 
        linestyle='dotted', 
        linewidth=1.5, 
        label='Truth'
    )
    
    plt.xlabel("Average Confidence Interval")
    plt.title(title)
    plt.yticks([])  # remove y-axis ticks
    plt.gca().spines['left'].set_visible(False)  # remove left border
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    

