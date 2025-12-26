
## Unsupervised Market Regime Detection (U.S. Equities)

This project implements an end‑to‑end **unsupervised** market regime detection pipeline for U.S. financial markets, using cross‑asset and macro features to identify latent bull, bear, and high‑volatility states.  The goal is to show how clustering and dimensionality reduction can be used to extract economically meaningful regimes that align with realized return and risk characteristics.[3][2][4][1]

## Problem Statement

Traditional regime labels (e.g., “bull” vs “bear”) are often defined ex‑post and rely on arbitrary thresholds that obscure intraregime dynamics.  This project instead learns regimes directly from data, clustering daily market conditions into statistically distinct states that can later be used for risk management, asset allocation, or signal conditioning.[4][3][1]

## Data and Feature Set

The pipeline operates on daily data from 2010–2024, aggregating:

- Equity and volatility:
  - S&P 500 index (^GSPC) OHLCV from Yahoo Finance  
  - VIX index (^VIX) level and intraday range as an implied volatility proxy[3][4]
- Rates and credit:
  - 10‑year Treasury yield (^TNX)  
  - 13‑week T‑bill rate (^IRX) as a short‑rate / Fed funds proxy  
  - Baa corporate bond yields and credit spreads from FRED[3]
- Macro indicators (FRED via `pandas_datareader`):
  - GDP, unemployment, industrial production, nonfarm payrolls  
  - Effective Fed funds rate, M2 money stock, 10‑year breakeven inflation, housing starts[3]

Representative engineered features include:

- Equity / volatility:
  - Daily S&P 500 log returns, 21‑day rolling volatility, 10‑day momentum  
  - First differences and percentage changes in VIX  
- Rates:
  - Level and daily changes in 10‑year yield and short‑rate proxy  
- Technicals:
  - MACD and signal line on S&P 500 (via `ta`)  
- All numeric features are standardized (z‑scores via `StandardScaler`) before PCA and clustering.[3]

## Methodology

The pipeline is implemented in Python (pandas, scikit‑learn, `pandas_datareader`, `ta`) and follows these steps:

- Data preparation
  - Download market data from Yahoo Finance for 2010–2024.  
  - Align all series on a business‑day calendar; forward‑fill macro series to daily frequency.  
  - Merge all sources into a unified panel and drop rows with missing values to obtain a clean feature matrix.[3]

- Dimensionality reduction
  - Apply PCA to the standardized feature matrix.  
  - Retain the top 3–5 principal components that explain most of the variance, and use them both for visualization and as inputs to clustering models.[1][3]

- Clustering models
  - K‑Means:
    - Fit models for \(k \in \{2, \dots, 6\}\).  
    - Evaluate clustering quality using silhouette scores, selecting the best‑performing \(k\) as the baseline regime count.[3]
  - Gaussian Mixture Models (GMM):
    - Fit GMMs over the same range of \(k\).  
    - Use log‑likelihood and Bayesian Information Criterion (BIC) for model selection, allowing soft regime assignments and non‑spherical clusters.[1][3]
  - Optional: Agglomerative clustering for hierarchical inspection and regime stability checks.[3]

## Regime Characterization and Diagnostics

For each identified regime, the notebook computes:

- Return and risk statistics:
  - Mean and standard deviation of daily S&P 500 returns  
  - Annualized volatility and hit ratio (share of positive days)  
- State descriptors:
  - Average VIX level and volatility  
  - Average changes in rates and short‑rate proxy  
  - Mean macro conditions (e.g., unemployment, GDP growth, credit spreads)[3]

These summaries are then mapped to intuitive labels such as:

- “Bull / risk‑on” regime: positive average returns, lower volatility, subdued VIX, supportive macro backdrop.  
- “Bear / high‑stress” regime: negative mean returns, elevated volatility, high VIX, widening credit spreads, weaker macro prints.  
- “Sideways / transition” regime: near‑zero returns, moderate volatility, mixed macro signals.[1][3]

Clustering performance is evaluated via:

- Silhouette scores across \(k\) for K‑Means (often peaking around 3 regimes in equity‑focused applications).[3]
- GMM log‑likelihood and BIC to balance fit and model complexity.[1][3]

Placeholders like \(S_k\), \(\mu_{\text{bull}}\), and \(\sigma_{\text{bull}}\) in the notebook can be directly populated from the regime summary tables once the pipeline is run on the latest data.
