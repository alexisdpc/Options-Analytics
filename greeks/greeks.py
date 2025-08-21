import numpy as np
import polars as pl
import scipy.stats as ss

from scipy.optimize import fsolve

def theo_price(S, K, T, r, q, sigma, CallPut):
    """Returns the price of a Call/Put option

    Arguments:
        S (float): Stock price
        K (float): Strike price
        T (float): Expiration time
        r (float): Drift rate
        q (float): Dividend rate
        sigma (float): Volatility
        CallPut (str): This is a string that can either be 'Call' or 'Put'

    Returns:
        Numpy array with the option price at each point in time.
    """
    
    # Fixed parameters
    eps = 1e-5

    # We use the analytic solution for the Call and Put pricing
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2.)*T)/(sigma*np.sqrt(T) + eps)
    d2 = d1 - sigma*np.sqrt(T)

    # For Call option:
    if CallPut == 'C':
        return S*np.exp(-q*T)*ss.norm.cdf(d1) - K*np.exp(-r*T)*ss.norm.cdf(d2)
    
    # For Put option:
    elif CallPut == 'P':
        return K*np.exp(-r*T)*ss.norm.cdf(-d2) - S*np.exp(-q*T)*ss.norm.cdf(-d1)
    
    else:
        raise ValueError("CallPut must be 'C' for Call or 'P' for Put.")

 
def delta(S, K, T, r, q, sigma, CallPut):
    """Returns the Delta of a Call/Put option.
       We use the analytical expression for Delta
       
    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        q (float): Dividend yield
        sigma (float): Volatility of the underlying asset
        CallPut (str): 'C' for Call option, 'P' for Put option

    Returns:
        float: Implied volatility (sigma)
    """
    eps = 1e-5
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2.)*T)/(sigma*np.sqrt(T) + eps)

    # For Call option:
    if CallPut == 'C':
        return ss.norm.cdf(d1)*np.exp(-q*T)
    
    # For Put option:
    elif CallPut == 'P':
        return ss.norm.cdf(d1)*np.exp(-q*T) - 1.0 

    else:
        raise ValueError("CallPut must be 'C' for Call or 'P' for Put.")
    

def gamma(S, K, T, r, q, sigma):
    """Returns the Gamma of a Call/Put option.
       Gamma is the same for both Call and Put options.
       
    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        q (float): Dividend yield
        sigma (float): Volatility of the underlying asset

    Returns:
        float: Gamma of the option
    """

    eps = 1e-5 # Avoids divergences
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2.)*T)/(sigma*np.sqrt(T) + eps)
    Nprime = np.exp(-d1**2./2.)/np.sqrt(2.*np.pi)  # Standard normal probability density function evaluated at d1

    # Same for both Call and Put options
    return Nprime/(S*sigma*np.sqrt(T) + eps)


def vega(S, K, T, r, q, sigma):
    """Returns the Vega of a Call/Put option.
       Vega is the same for both Call and Put options.
       
    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        q (float): Dividend yield
        sigma (float): Volatility of the underlying asset

    Returns:
        float: Vega of the option
    """
    eps = 1e-5 # Avoids divergences
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2.)*T)/(sigma*np.sqrt(T) + eps)
    Nprime = np.exp(-d1**2./2.)/np.sqrt(2.*np.pi) # Standard normal probability density function evaluated at d1

    # Same for both Call and Put options
    vega_value = S*Nprime*np.sqrt(T)*np.exp(-q*T)

    # We divide by 100 (1 vol point change)
    return vega_value/100.0


def theta(S, K, T, r, q, sigma, CallPut):
    """
    Calculate the theta of a Call/Put option using the Black-Scholes model.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        q (float): Dividend yield
        sigma (float): Volatility of the underlying asset
        CallPut (str): 'C' for Call option, 'P' for Put option

    Returns:
        float: Theta of the option
    """
    eps = 1e-5 # Avoids divergences
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2.)*T)/(sigma*np.sqrt(T) + eps)
    d2 = d1 - sigma*np.sqrt(T)
    Nprime = np.exp(-d1**2./2.)/np.sqrt(2.*np.pi)
    term1 = -S*Nprime*sigma/(2.*np.sqrt(T)) * np.exp(-q*T)  

    # For Call option:
    # Do I need to convert to per-day theta?? theta/365.0    
    if CallPut == 'C':
        theta_call  =  term1 - r*K*np.exp(-r*T)*ss.norm.cdf(d2) + q*S*np.exp(-q*T)*ss.norm.cdf(d1)
        return theta_call/252.0 # Daily value

    # For Put option:
    # Do I need to convert to per-day theta?? theta/365.0
    elif CallPut == 'P':
        theta_put = term1 + r*K*np.exp(-r*T)*ss.norm.cdf(-d2) - q*S*np.exp(-q*T)*ss.norm.cdf(-d1)
        return theta_put/252.0 # Daily value
    
    else:
        raise ValueError("CallPut must be 'C' for Call or 'P' for Put.")


def rho(S, K, T, r, q, sigma, CallPut):
    """
    Calculate the rho of a Call/Put option using the Black-Scholes model.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        q (float): Dividend yield
        sigma (float): Volatility of the underlying asset
        CallPut (str): 'C' for Call option, 'P' for Put option

    Returns:
        float: Rho of the option
    """
    eps = 1e-5  # Avoids divergences
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + eps)
    d2 = d1 - sigma * np.sqrt(T)

    # For Call option:
    if CallPut == 'C':
        rho_call = K * T * np.exp(-r * T) * ss.norm.cdf(d2)
        # (1 basis point rate change)
        return rho_call/10000.0

    # For Put option:
    elif CallPut == 'P':
        rho_put = -K * T * np.exp(-r * T) * ss.norm.cdf(-d2)
        # (1 basis point rate change)
        return rho_put/10000.0

    else:
        raise ValueError("CallPut must be 'C' for Call or 'P' for Put.") 


def iv_solver(S, K, T, r, q, CallPut, market_price):
    """
    Solves for the implied volatility of an option using the Black-Scholes model.

    Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        q (float): Dividend yield
        CallPut (str): 'C' for Call option, 'P' for Put option
        market_price (float): Observed market price of the option

    Returns:
        float: Implied volatility (sigma)
    """

    # Check if any parameter is None
    if any(param is None for param in [S, K, T, r, q, CallPut, market_price]):
        return None

    def iv_equation(sigma):
        """
        Calculates the difference between the Black-Scholes price and the market price
        for a given volatility (sigma). This equation is used to find the root (implied volatility).
        """
        # Ensure sigma is positive
        if sigma <= 0:
            return float('inf')
        
        # Calculate the option price using the given parameters
        price = theo_price(S, K, T, r, q, sigma, CallPut)

        # Return the difference between the calculated price and the market price
        return price - market_price

    # Use a reasonable initial guess for implied volatility (e.g., 0.01 to 1.0)
    initial_guess = 0.2  # Typical starting point for implied volatility

    root = fsolve(iv_equation,  initial_guess)
    return root[0]


def iv_calculator(options_df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate implied volatility (IV) for each option in the DataFrame.
    The IV is calculated using the Black-Scholes model.
    The IV is calculated for both the bid and ask prices.

    Parameters:
        options_df (DataFrame): DataFrame containing option data.

    Returns:
        options_df (pl.DataFrame): Updated DataFrame with implied volatility column.
    """

    # Initialize lists to store results
    iv_list = []
    delta_list = []
    theta_list = []
    rho_list = []

    for i in range(0, len(options_df)):
    
        if i % 10000 == 0:
            print(f"Processing row {i} of {len(options_df)}")

        # Extracting the values from the DataFrame
        mid_price = (options_df[i, "bid"] + options_df[i, "ask"]) / 2.0
        S = options_df[i, "spx_close"]  #open
        K = options_df[i, "strike"]
        T = options_df[i, "days_to_exp"]/252.
        r = options_df[i, "interest_rate"]*1e-2  #sofr_rate
        q = 1.5*1e-2 
        CallPut = options_df[i, "right"]

        # This print can be used to debug the values
        #print(f"{i} - {bid}, {ask}, {S}, {K}, {T}, {r}, {q}, {CallPut}")

        iv = iv_solver(S, K, T, r, q, CallPut, mid_price)
        delta_mid = delta(S, K, T, r, q, iv, CallPut)
        theta_mid = theta(S, K, T, r, q, iv, CallPut)
        rho_mid = rho(S, K, T, r, q, iv, CallPut)

        iv_list.append(iv)
        delta_list.append(delta_mid)
        theta_list.append(theta_mid)
        rho_list.append(rho_mid)

    options_df = options_df.with_columns(pl.Series(name="iv", values=iv_list))  
    options_df = options_df.with_columns(pl.Series(name="delta", values=delta_list))     
    options_df = options_df.with_columns(pl.Series(name="theta", values=theta_list))
    options_df = options_df.with_columns(pl.Series(name="rho", values=rho_list))

    return options_df

def gamma_calculator(options_df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized calculation of Gamma for each option in the DataFrame.
    Call and Puts have the same Gamma.

    Parameters:
        options_df (DataFrame): DataFrame containing option data.

    Returns:
        options_df (pl.DataFrame): Updated DataFrame with implied volatility column.
    """
    options_df = options_df.with_columns(
        gamma = gamma(
            pl.col("spx_close"),
            pl.col("strike"),
            pl.col("days_to_exp") / 252.0,
            pl.col("interest_rate") * 1e-2,
            1.5 * 1e-2,
            pl.col("iv"),
        )
    )

    return options_df

def vega_calculator(options_df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized calculation of Vega for each option in the DataFrame.
    Call and Puts have the same Vega.

    Parameters:
        options_df (DataFrame): DataFrame containing option data.

    Returns:
        options_df (pl.DataFrame): Updated DataFrame with implied volatility column.
    """
    options_df = options_df.with_columns(
        vega = vega(
            pl.col("spx_close"),
            pl.col("strike"),
            pl.col("days_to_exp") / 252.0,
            pl.col("interest_rate") * 1e-2,
            1.5 * 1e-2,
            pl.col("iv"),
        )
    )

    return options_df

def get_sofr():
    """Fetches the SOFR interest rates data from a CSV file.

    Returns:
        DataFrame: A Polars DataFrame containing the SOFR interest rates data.
    """
    sofr = pl.read_csv(
        "../data/sofr-rates.csv",
        try_parse_dates=True,
        infer_schema_length=1000000,
    )

    return sofr
