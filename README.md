# DeepLOB: Generative Diffusion for Synthetic Market Microstructure

## Project Overview
This research project implements a **conditional generative diffusion framework** designed to synthesize high-fidelity Limit Order Book (LOB) dynamics. By leveraging non-equilibrium thermodynamics-inspired modeling, the simulator reconstructs intraday liquidity patterns and price formation processes that characterize modern electronic markets.

## Scientific Objective
The primary goal is to address the limitations of traditional continuous-time Markov processes in LOB modeling, which often struggle to capture the **Non-Markovian** nature and heavy-tailed return distributions observed in high-frequency data. 

This framework specifically investigates:
* **Rough Volatility Regimes:** Modeling volatility as a fractional process where the Hurst Index ($H$) is strictly less than 0.5.
* **Stylized Facts Reproduction:** Validating synthetic data against empirical benchmarks, including volatility clustering, autocorrelation decay, and the leverage effect.
* **Systemic Risk & Algorithmic Coupling:** Analyzing how homogenized execution strategies ($\rho$) lead to percolation phase transitions and liquidity collapse.



## Technical Architecture
* **Generative Core:** A **Deep Diffusion Model (DDM)** implemented in **PyTorch**. The model learns the joint distribution of price levels and order volume by reversing a Gaussian noise injection process over $T$ diffusion steps.
* **Stochastic Parameterization:** Users can modulate the roughness of the volatility manifold via the CLI, allowing for the simulation of "Flash Crash" scenarios or stable, high-liquidity regimes.
* **Data Pipeline:** An automated ETL workflow that consumes public equity data (via `yfinance` or LOBSTER formats) to condition the generative process on real-world macroeconomic indicators.

## Mathematical Foundation
The simulation engine utilizes a reverse-time Stochastic Differential Equation (SDE) to transform Gaussian noise into structured LOB states. By conditioning the score function on historical volatility and volume imbalance, the model captures the path-dependency essential for accurate backtesting of execution algorithms.



## Key Parameters
| Parameter | Symbol | Scientific Significance |
| :--- | :--- | :--- |
| **Hurst Exponent** | $H$ | Controls the "roughness" and long-term memory of the volatility path. |
| **Coupling Coeff.** | $\rho$ | Measures the degree of algorithmic resonance among market participants. |
| **Diffusion Steps** | $T$ | Determines the resolution of the generative reverse-SDE process. |

## Research Applications
1.  **PhD Research:** Provides a modular environment for testing **Stochastic Optimal Control** theories and neural operators under non-Markovian volatility.
2.  **Quantitative Trading:** Enables high-fidelity backtesting of execution algorithms (TWAP/VWAP) by generating statistically consistent market scenarios.
3.  **Risk Management:** Facilitates "Stress-Testing-as-a-Service," evaluating portfolio sensitivity to sudden shifts in liquidity density.

---
*Developed by Eliott Elkeslassy — 2026*
