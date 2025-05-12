# Options Market Microstructure Analysis

This repository provides tools and notebooks for analyzing the microstructure of the options market. Topics include order flow dynamics, bid-ask behavior, implied volatility surfaces, and liquidity patterns. Designed for quantitative researchers and traders interested in (intraday) market mechanics.

# Bid/Ask Spread

In the Figure below we show the bid ask spread in dollars as a function of the moneyness of the option

$$ \text{Moneyness} = \log \left( \frac{K}{S_t} \right) $$ 

on the left panel we show the spread for the options with 1 day to expiration (1DTE), while on the right panel for 10 DTE. Intraday quotes for SPX option on the date 2024-04-23.




![image](https://github.com/user-attachments/assets/c9c0b77d-a5c0-44d1-8a7f-7958e5f4cca7)

![image](https://github.com/user-attachments/assets/07c18612-9fce-495e-a2b7-6850540a917e)

![image](https://github.com/user-attachments/assets/f0aed7c9-0970-4995-be4c-fd201f54b367)



Percentage spread: When we look specifically at out-of-the-money (OTM) options, their minimum bid-ask spread tends to be quite close to that of at-the-money (ATM) options. On the other hand, the deep in-the-money (ITM) options generally exhibit tighter spreads.

# Trading Volume

![image](https://github.com/user-attachments/assets/179347c1-726c-499e-a268-0731b3f7bdc2)


# Put Call Parity

The reference rates are obtained from:
https://www.newyorkfed.org/markets/reference-rates/sofr

![image](https://github.com/user-attachments/assets/b37ec7eb-a622-4e8f-bf7e-e2af3549566f)








