import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def evaluate_synthetic_data(real_returns, synthetic_returns):
    """
    Compares the distribution of real market returns and synthetic returns 
    using the Kolmogorov-Smirnov test and visualizes volatility clusters.
    """
    # 1. Kolmogorov-Smirnov Test
    # The KS test compares the underlying continuous distributions of two independent samples.
    # Null hypothesis: both samples are drawn from the same continuous distribution.
    ks_stat, p_value = ks_2samp(real_returns, synthetic_returns)
    
    print("--- Kolmogorov-Smirnov Test ---")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"P-value:      {p_value:.4e}")
    if p_value < 0.05:
        print("Result: Reject the null hypothesis (Distributions are likely significantly different).")
    else:
        print("Result: Fail to reject the null hypothesis (Distributions appear similar).")
        
    # 2. Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # A. Distribution comparison (Histogram / Density)
    axes[0].hist(real_returns, bins=50, alpha=0.5, density=True, label='Real Returns')
    axes[0].hist(synthetic_returns, bins=50, alpha=0.5, density=True, label='Synthetic Returns')
    axes[0].set_title('Density Distribution of Returns')
    axes[0].set_xlabel('Return')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # B. Volatility Clustering Visualization
    # Volatility clustering is often observed by looking at the absolute returns over time
    axes[1].plot(np.abs(real_returns), alpha=0.7, label='Real |Returns|', linewidth=1)
    axes[1].plot(np.abs(synthetic_returns), alpha=0.6, label='Synthetic |Returns|', linewidth=1)
    axes[1].set_title('Volatility Clustering (|Returns| Over Time)')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Absolute Return (Volatility Proxy)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    print("Saved evaluation plots to 'evaluation_results.png'")
    
    try:
        plt.show()
    except Exception as e:
        print("Could not display plot interactively. Please check 'evaluation_results.png'.")

if __name__ == "__main__":
    # --- Dummy Data Generation for Testing ---
    # In practice, you would pass your actual `real_returns` and `model.sample(...)` outputs here.
    
    print("Generating dummy data with simulated volatility clustering...")
    n_steps = 1000
    
    # Simulate a process with volatility clustering (similar to GARCH)
    real_vol = np.zeros(n_steps)
    real_vol[0] = 0.01
    for i in range(1, n_steps):
        # Current volatility depends on previous volatility and a random shock
        real_vol[i] = np.sqrt(0.00001 + 0.1 * (np.random.randn() * real_vol[i-1])**2 + 0.85 * real_vol[i-1]**2)
    
    real_returns = np.random.randn(n_steps) * real_vol
    
    # Simulate synthetic returns (assume the model learned it reasonably well, but with minor noise)
    synthetic_returns = np.random.randn(n_steps) * real_vol * 1.02 + 0.0005
    
    evaluate_synthetic_data(real_returns, synthetic_returns)
