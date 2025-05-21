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

$$ C - P = S $$

This no-arbitrage condition dictates that a specific portfolio consisting of these options and the underlying asset must yield a return equivalent to the risk-free rate. Understanding put-call parity is essential for option traders and investors as it provides a benchmark for identifying potential arbitrage opportunities and for constructing various hedging and speculation strategies.

The reference rates are obtained from:
https://www.newyorkfed.org/markets/reference-rates/sofr

![image](https://github.com/user-attachments/assets/b37ec7eb-a622-4e8f-bf7e-e2af3549566f)

# Volatility Smile

![image](https://github.com/user-attachments/assets/54bd1ba2-d214-43b3-8996-adf44990855d)



# Spread for Index Futures








