import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os




def plot_CI_by_conditions(results, save_dir=None, dpi=300, file_format="png"):
    """
    For each (bias_func, bias_scale) combination, create line plots of coverage
    and width vs. n_real, n_expanded, r_expanded, colored by r0_learner_name.

    Parameters:
        results: list of dicts containing experiment output
        save_dir: if provided, saves each plot to this directory
        dpi: dots per inch for saved figures (default 300 for print quality)
        file_format: 'png', 'pdf', or 'svg' (for LaTeX, use 'pdf' or 'svg')
    """
    df = pd.DataFrame(results)
    sns.set(style="whitegrid")
    sns.set_context("talk")

    combos = df[["bias_func", "bias_scale"]].drop_duplicates()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for _, row in combos.iterrows():
        bias_func = row["bias_func"]
        bias_scale = row["bias_scale"]

        subset = df[
            (df["bias_func"] == bias_func) &
            (df["bias_scale"] == bias_scale)
        ]

        fig, axes = plt.subplots(2, 3, figsize=(24, 13))
        axes = axes.flatten()  # flatten to 1D array

        fig.suptitle(f"CI Metrics for bias_func={bias_func}, bias_scale={bias_scale}", fontsize='large')

        # Plot 1: Coverage vs. n_real
        sns.lineplot(data=subset, x="n_real", y="coverage", hue="r0_learner_name", marker="o", ci=None, ax=axes[0])
        axes[0].set_title("Coverage vs. n_real")

        # Plot 2: Coverage vs. n_expanded
        sns.lineplot(data=subset, x="n_expanded", y="coverage", hue="r0_learner_name", marker="o", ci=None, ax=axes[1])
        axes[1].set_title("Coverage vs. n_expanded")

        # Plot 3: Coverage vs. r_expanded
        sns.lineplot(data=subset, x="r_expanded", y="coverage", hue="r0_learner_name", marker="o", ci=None, ax=axes[2])
        axes[2].set_title("Coverage vs. r_expanded")

        # Plot 4: Width vs. n_real
        sns.lineplot(data=subset, x="n_real", y="avg_me", hue="r0_learner_name", marker="o", ci=None, ax=axes[3])
        axes[3].set_title("Width vs. n_real")

        # Plot 5: Width vs. n_expanded
        sns.lineplot(data=subset, x="n_expanded", y="avg_me", hue="r0_learner_name", marker="o", ci=None, ax=axes[4])
        axes[4].set_title("Width vs. n_expanded")

        # Plot 6: Width vs. r_expanded
        sns.lineplot(data=subset, x="r_expanded", y="avg_me", hue="r0_learner_name", marker="o", ci=None, ax=axes[5])
        axes[5].set_title("Width vs. r_expanded")

        for ax in axes:
            ax.get_legend().remove()

        # Add a single shared legend below all plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            title="Learner",
            loc="lower center",
            ncol=len(labels),                 
            bbox_to_anchor=(0.5, -0.12),      
            columnspacing=1.2,
            handletextpad=0.5,
            fontsize='medium',
            title_fontsize='medium'
        )

        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        if save_dir:
            filename = f"CI_metrics_{bias_func}_{bias_scale}.{file_format}".replace(" ", "_")
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
        
        plt.show()
        
        
            
            

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
    
    
    

