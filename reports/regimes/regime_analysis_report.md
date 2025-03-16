# Strategy Performance Analysis Across Market Regimes

This report analyzes how the trading strategy performs across different market regimes identified by three different detection methods: Feature-based, Hidden Markov Model (HMM), and Transformer.

## 1. Executive Summary

I've performed a comprehensive analysis of the trading strategy performance across different market regimes identified by three detection methods: Feature-based, HMM, and Transformer models. The analysis reveals that this strategy has distinct performance characteristics across different market regimes, with the potential for significant performance improvements through regime-aware trading adjustments.

### Feature-based Model
- **Highest Cumulative Return**: Neutral regime (159.26%)
- **Highest Average Return**: Mean Reversion regime (0.068% per period)
- **Best Sharpe Ratio**: Mean Reversion regime (0.066)
- **Highest Win Rate**: Volatile regime (57.03%)
- **Regime Prevalence**: Neutral periods dominate (56.41% of time)
- **Key Insight**: Strategy performs poorly in downtrends (-16.02% cumulative return)

### HMM Model
- **Only Detected Two Regimes**: Uptrend (64.04% of time) and Downtrend (35.95% of time)
- **Better Performance in Uptrends**: 127.77% cumulative return vs. 41.96% in downtrends
- **Higher Sharpe Ratio in Uptrends**: 0.045 vs. 0.017 in downtrends
- **Lower Volatility in Uptrends**: 0.68% vs. 1.03% in downtrends
- **Similar Win Rates**: 51.90% in uptrends vs. 50.94% in downtrends

### Transformer Model
- **Detected Three Regimes**: Uptrend (64.01%), Downtrend (35.70%), and Neutral (0.29%)
- **Best Performance in Uptrends**: 128.33% cumulative return
- **Poor Performance in Neutral**: -1.52% cumulative return (but only 19 periods)
- **Higher Sharpe in Uptrends**: 0.046 vs. 0.018 in downtrends
- **Lower Volatility in Uptrends**: 0.67% vs. 1.03% in downtrends
- **Minor Win Rate Differences**: 51.87% in uptrends vs. 51.01% in downtrends

## 2. Detailed Analysis

### 2.1 Feature-based Regime Analysis

The feature-based model detected five distinct regimes with significant performance differences:

| Regime | % of Time | Avg Return (%) | Cum Return (%) | Volatility (%) | Sharpe | Win Rate (%) | Max DD (%) |
|--------|-----------|----------------|----------------|----------------|--------|--------------|------------|
| Neutral | 56.41 | 0.043 | 159.26 | 0.67 | 0.065 | 52.78 | 14.82 |
| Uptrend | 19.59 | 0.0004 | 0.57 | 0.81 | 0.0005 | 51.57 | 35.65 |
| Downtrend | 17.72 | -0.014 | -16.02 | 1.08 | -0.013 | 46.70 | 65.62 |
| Mean Reversion | 4.31 | 0.068 | 19.03 | 1.03 | 0.066 | 52.86 | 16.57 |
| Volatile | 1.97 | 0.054 | 6.88 | 1.41 | 0.038 | 57.03 | 11.50 |

**Key Observations**:
1. The strategy performs exceptionally well during neutral market conditions, which represent the majority of the time (56.41%).
2. Interestingly, despite being labeled as "uptrend," the strategy barely generates any returns in uptrend regimes (only 0.57% cumulatively).
3. The strategy shows negative performance in downtrend regimes (-16.02%).
4. The highest average returns come from mean reversion regimes (0.068% per period).
5. The most favorable win rate occurs during volatile regimes (57.03%).
6. The worst drawdowns occur in downtrend regimes (65.62%).

This suggests the strategy might be designed to capture small, consistent moves during relatively calm markets rather than trending markets.

### 2.2 HMM Regime Analysis

The HMM model simplified the market into just two regimes:

| Regime | % of Time | Avg Return (%) | Cum Return (%) | Volatility (%) | Sharpe | Win Rate (%) | Max DD (%) |
|--------|-----------|----------------|----------------|----------------|--------|--------------|------------|
| Uptrend | 64.04 | 0.031 | 127.77 | 0.68 | 0.045 | 51.90 | 23.04 |
| Downtrend | 35.96 | 0.018 | 41.96 | 1.03 | 0.017 | 50.94 | 43.35 |

**Key Observations**:
1. The HMM model simplifies the market into just two states: uptrend (64.04% of time) and downtrend (35.96% of time).
2. The strategy performs positively in both regimes, but much better in uptrends (127.77% vs. 41.96% cumulative returns).
3. Volatility is significantly higher in downtrend regimes (1.03% vs. 0.68%).
4. The win rates are similar in both regimes, suggesting consistency in the strategy's approach.
5. Maximum drawdowns are much worse in downtrend regimes (43.35% vs. 23.04%).

The HMM model provides a clearer distinction between high and low-performance periods compared to the feature-based approach.

### 2.3 Transformer Regime Analysis

The Transformer model was very similar to the HMM model:

| Regime | % of Time | Avg Return (%) | Cum Return (%) | Volatility (%) | Sharpe | Win Rate (%) | Max DD (%) |
|--------|-----------|----------------|----------------|----------------|--------|--------------|------------|
| Uptrend | 64.01 | 0.031 | 128.33 | 0.67 | 0.046 | 51.87 | 22.88 |
| Downtrend | 35.70 | 0.018 | 42.92 | 1.03 | 0.018 | 51.01 | 42.47 |
| Neutral | 0.29 | -0.080 | -1.52 | 0.80 | -0.100 | 47.37 | 2.69 |

**Key Observations**:
1. The Transformer model's regime classification is very similar to the HMM model, with almost identical time allocations for uptrends and downtrends.
2. The strategy performs best in uptrend regimes (128.33% cumulative return), similar to the HMM findings.
3. The neutral regime is rarely detected (only 0.29% of the time) but shows negative performance when identified.
4. Volatility patterns match the HMM model, with higher volatility in downtrends.
5. Win rates are relatively consistent across uptrend and downtrend regimes.

The similarity between the HMM and Transformer classifications suggests both models are capturing similar underlying patterns in the market.

## 3. Comparative Analysis

### 3.1 Model Agreement

The HMM and Transformer models show high agreement in their regime classifications, while the feature-based approach differs significantly:

- **Feature vs. HMM/Transformer**: The feature-based model detects five distinct regimes, while HMM and Transformer primarily focus on up/down trends.
- **HMM vs. Transformer**: These models show remarkable similarity, suggesting they're identifying the same underlying market dynamics.

### 3.2 Strategy Performance Patterns

Across all models, some consistent patterns emerge:

1. **Lower volatility in uptrends**: All models show that uptrend or neutral periods have lower volatility than downtrends or volatile periods.
2. **Higher Sharpe ratios in uptrends**: All models indicate better risk-adjusted returns during uptrend periods.
3. **Larger drawdowns in downtrends**: The strategy consistently experiences its worst drawdowns during downtrend periods.
4. **Win rate consistency**: Win rates don't vary dramatically across regimes, suggesting the strategy maintains its edge regardless of market conditions.

### 3.3 Model Usefulness for Trading

- **Feature-based Model**: Provides the most granular classification but may overfit to specific patterns. Its "neutral" regime classification appears most profitable for this strategy.
- **HMM Model**: Offers a simpler, more robust classification that clearly delineates between higher and lower performance periods.
- **Transformer Model**: Performs very similarly to HMM but occasionally identifies a rare "neutral" regime that shows poor performance.

## 4. Strategic Implications

Based on the analysis, here are potential strategic adjustments that could be made:

1. **Regime-based Position Sizing**: Increase position sizes during uptrends/neutral regimes and reduce exposure during downtrends. Specifically, increase position size during HMM/Transformer uptrend regimes or Feature-based neutral regimes.

2. **Risk Management**: Implement tighter stop-losses during downtrend regimes where drawdowns are typically larger.

3. **Strategy Filtering**: Consider only trading during specific regimes where performance is strongest or adjusting the strategy parameters based on the detected regime.

4. **Model Selection**: The HMM model provides the clearest signal for trading decisions due to its simplicity and clear performance differentiation. For practical implementation, this model offers the best balance of simplicity and clarity in differentiating between regimes.

5. **Volatility Adjustment**: The strategy performs better in lower volatility environments (particularly uptrends in HMM/Transformer models or neutral periods in feature-based model).

## 5. Conclusion

The strategy shows distinct performance characteristics across different market regimes. It performs best during periods identified as uptrends by the HMM and Transformer models, or neutral periods by the feature-based model. 

The worst performance occurs during downtrends (feature-based model) or the rare neutral regime (Transformer model). Volatility is consistently higher during downtrend periods across all models.

The analysis reveals that this strategy has distinct performance characteristics across different market regimes, with the potential for significant performance improvements through regime-aware trading adjustments. 