# Adverse Selection Detection Algorithm

## Overview
This notebook implements a comprehensive machine learning algorithm to classify trades as **informed** vs **uninformed**, addressing the critical problem of adverse selection in market making and electronic trading.

## Problem Statement
In financial markets, **adverse selection** occurs when market makers consistently trade against informed traders who possess superior information. This leads to:
- Systematic losses for market makers
- Wider bid-ask spreads
- Reduced market liquidity
- Increased trading costs for all participants

## Objectives
1. **Develop** a robust classification system to identify informed trading activity in real-time
2. **Engineer** comprehensive features capturing market microstructure signals
3. **Implement** ensemble machine learning models for improved prediction accuracy
4. **Enable** dynamic risk management and spread adjustment strategies
5. **Provide** actionable insights for market making algorithms

## Methodology

### Feature Engineering
- **Trade-level features**: Size, timing, price impact characteristics
- **Market microstructure**: Bid-ask spreads, quote dynamics, trade direction
- **Volume patterns**: Rolling statistics, trade frequency, size distributions
- **Volatility metrics**: Short-term vs long-term volatility ratios
- **Momentum indicators**: Price trends and moving average relationships
- **Order book features**: Depth imbalances and order flow dynamics

### Machine Learning Models
**Ensemble approach** combining three complementary algorithms:
- **RandomForestClassifier**: Captures non-linear feature interactions through bagging
- **GradientBoostingClassifier**: Sequential error correction with adaptive learning
- **XGBoost**: Optimized gradient boosting with regularization

### Labeling Strategy
Informed trades are identified using:
- **Price impact analysis**: Trades with significant post-trade price movements
- **Size-impact correlation**: Large trades with persistent market impact
- **Momentum continuation**: Trades predicting future price direction

## Expected Results

### Performance Metrics
- **ROC-AUC Score**: Target > 0.70 for practical trading applications
- **Precision**: Minimize false positives to avoid unnecessary spread widening
- **Recall**: Capture majority of informed trades to protect against adverse selection
- **F1-Score**: Balance between precision and recall for optimal trading performance

### Business Impact
- **Risk Reduction**: Early detection of informed trading reduces market maker losses
- **Dynamic Pricing**: Real-time spread adjustment based on informed trade probability
- **Regulatory Compliance**: Enhanced surveillance for unusual trading patterns
- **Market Quality**: Improved liquidity provision through better risk management

### Real-Time Application
The trained model enables:
- **Live trade classification** with sub-millisecond latency
- **Adaptive market making** strategies
- **Risk-adjusted position sizing**
- **Intelligent order routing** decisions

---

**Train multiple models for ensemble prediction:**
 - RandomForestClassifier
 - GradientBoostingClassifier
 - XGBoost

