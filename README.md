# Options Analytics

This project provides a simple yet powerful pipeline for financial options analysis. It is designed to handle the two primary stages of quantitative options research:
- `retriever.py`: This file contains the modules responsible for connecting to and retrieving options data from the data provider, ThetaData.
- `greeks.py`: This file reads the raw market options data and calculates all the Greeks (Delta, Gamma, Theta, Vega, Rho) for every option contract using the Black-Scholes approach.

The ultimate goal is to provide a clean, modular foundation for building more complex trading strategies, risk management systems, or academic research models.

Other projects in this repository include:
- Adverse Selection Detection Algorithm
- Delta Hedging Stock Simulation 


# Options Market Microstructure Analysis

This repository provides tools and notebooks for analyzing the microstructure of the options market. Topics include order flow dynamics, bid-ask behavior, implied volatility surfaces, and liquidity patterns. Designed for quantitative researchers and traders interested in (intraday) market mechanics.

# Bid/Ask Spread

The bid-ask spread in options trading represents the difference between the highest price a buyer is willing to pay (the bid) and the lowest price a seller is willing to accept (the ask) for 1  an option contract. This spread essentially reflects the transaction cost for traders, as buyers will typically pay the ask price, and sellers will receive the bid price. The width of the bid-ask spread is a key indicator of an option's liquidity, with narrower spreads suggesting higher liquidity and lower transaction costs, while wider spreads indicate lower liquidity and potentially higher costs. 

In the Figure below we show the bid ask spread in dollars as a function of the moneyness of the option

$$ \text{Moneyness} = \log \left( \frac{K}{S_t} \right) $$ 

on the left panel we show the spread for the options with 1 day to expiration (1DTE), while on the right panel for 10 DTE. Intraday quotes for SPX option on the date 2024-04-23.

![image](https://github.com/user-attachments/assets/84c4a485-aa16-4fee-93db-ccd09fc25e57)


In the figure below we show the spread in Percentage (%) of the mid price (bid+ask)/2.  When we look specifically at out-of-the-money (OTM) options, their minimum bid-ask spread tends to be quite close to that of at-the-money (ATM) options. On the other hand, the deep in-the-money (ITM) options generally exhibit tighter spreads.

![image](https://github.com/user-attachments/assets/88dfc7cc-e8fa-4419-b657-935988c2393e)


# Trading Volume

Options near the at-the-money (ATM) options have the highest trading volume because they exhibit the greatest sensitivity to changes in the underlying asset's price, attracting both speculators seeking to capitalize on potential price movements and hedgers looking for effective risk management tools. This high level of interest and activity around ATM options leads to tighter bid-ask spreads and greater liquidity compared to in-the-money or out-of-the-money options

![image](https://github.com/user-attachments/assets/179347c1-726c-499e-a268-0731b3f7bdc2)


# Put Call Parity

Put-call parity is a fundamental principle in options pricing theory that establishes a crucial relationship between the prices of European call and put options with the same underlying asset, strike price, and expiration  date. The put-call parity equation is given by

$$ C - P = S - K e^{-r(T-t)}$$

This no-arbitrage condition dictates that a specific portfolio consisting of these options and the underlying asset must yield a return equivalent to the risk-free rate. Understanding put-call parity is essential for option traders and investors as it provides a benchmark for identifying potential arbitrage opportunities and for constructing various hedging and speculation strategies.

The reference rates are obtained from:
https://www.newyorkfed.org/markets/reference-rates/sofr

![image](https://github.com/user-attachments/assets/4b5081b9-8029-4f20-9fc9-2839a8501a3d)


# Implied Volatility and the Greeks

The following table summarizes the theoretical price and the key Greeks (i.e., sensitivities to various parameters) for European call and put options, derived using the Black-Scholes-Merton model with continuous dividend yield.

| Greek / Price | Call Option Formula                                                                                 | Put Option Formula                                                                                  |
|---------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **Fair Value**| $$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$                                                           | $$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$                                                         |
| **Delta (Δ)** | $$\Delta_C = e^{-qT} N(d_1)$$                                                                         | $$\Delta_P = e^{-qT} (N(d_1) - 1)$$                                                                   |
| **Gamma (Γ)** | $$\Gamma = \frac{e^{-qT} N'(d_1)}{S \sigma \sqrt{T}}$$                                                | $$\Gamma = \frac{e^{-qT} N'(d_1)}{S \sigma \sqrt{T}}$$ (same as call)                               |
| **Theta (Θ)** | $$\Theta_C = -\frac{S e^{-qT} N'(d_1) \sigma}{2 \sqrt{T}} + q S e^{-qT} N(d_1) - r K e^{-rT} N(d_2)$$ | $$\Theta_P = -\frac{S e^{-qT} N'(d_1) \sigma}{2 \sqrt{T}} - q S e^{-qT} N(-d_1) + r K e^{-rT} N(-d_2)$$ |
| **Vega (ν)**  | $$\nu = S e^{-qT} N'(d_1) \sqrt{T}$$                                                                  | $$\nu = S e^{-qT} N'(d_1) \sqrt{T}$$ (same as call)                                                 |
| **Rho (ρ)**   | $$\rho_C = K T e^{-rT} N(d_2)$$                                                                       | $$\rho_P = -K T e^{-rT} N(-d_2)$$                                                                    |

The parameters $d_1$ and $d_2$ in the Black-Scholes model are given by:

$$ d_1 = \frac{\ln\left(\frac{S}{K}\right) + \left(r - q + \frac{\sigma^2}{2}\right)T}{\sigma \sqrt{T}} $$

$$d_2 = d_1 - \sigma \sqrt{T}$$


![image](https://github.com/user-attachments/assets/54bd1ba2-d214-43b3-8996-adf44990855d)

![image](https://github.com/user-attachments/assets/07ecff2b-730e-416f-a1f3-76e260174ac6)


In the figure below we shoe the term structure for call options on the SPX. The term structure of implied volatility (IV) often reflects shifting market expectations. When short-term IV spikes, it usually signals immediate uncertainty, such as upcoming economic data releases, earnings, or recent market stress—leading to heightened demand for short-dated options as protection. 

In contrast, intermediate-term IV tends to dip, indicating that traders foresee a period of stabilization or reduced volatility once near-term risks subside. Interestingly, long-term IV often rises again, capturing broader concerns like recession risks, inflation, or political uncertainty. This uptick also reflects structural demand from institutions seeking long-dated hedges, contributing to higher pricing for options with extended maturities.


![image](https://github.com/user-attachments/assets/71727c98-afda-4007-801c-01cd32b013e7)

The figures below show the bid prices of SPX calls and puts as a function of time to expiration. Bid prices increase almost monotonically with days to expiration, demonstrating the fundamental concept of *time value* in options pricing (theta is always negative). 

Longer expiration periods provide greater opportunity for favorable price movements, making options with more time inherently more valuable. This relationship is consistent across all strike prices, with the steepest increases occurring in the first 30-60 days, reflecting the non-linear nature of time decay in options valuation.

![image](https://github.com/user-attachments/assets/80037d55-a58c-4c99-a6d0-2a9ea19508fa)

The figure below shows the same as before but with mid prices instead of bid. Here the variations becomes less noticeable.

![image](https://github.com/user-attachments/assets/d8543cbc-43d9-46a8-aec3-9fec0f85e3f1)








