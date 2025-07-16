import polars as pl
import pandas as pd
import httpx
import csv
import datetime as dt
import QuantLib as ql
import yfinance as yf
import httpx  # install via pip install httpx
import csv
import polars as pl

minute_to_milliseconds = {
    "1m": 1 * 60 * 1000,
    "3m": 3 * 60 * 1000,
    "5m": 5 * 60 * 1000,
    "10m": 10 * 60 * 1000,
    "15m": 15 * 60 * 1000,
    "30m": 30 * 60 * 1000,
    "60m": 60 * 60 * 1000,
}

strike_multiplier = 1000

hist_quotes_schema = {
    "root": pl.String,
    "expiration": pl.String,
    "strike": pl.Int64,
    "right": pl.String,
    "ms_of_day": pl.Int64,
    "bid_size": pl.Float64,
    "bid_exchange": pl.Int64,
    "bid": pl.Float64,
    "bid_condition": pl.Int64,
    "ask_size": pl.Float64,
    "ask_exchange": pl.Int64,
    "ask": pl.Float64,
    "ask_condition": pl.Int64,
    "date": pl.String,
}

thetadata_multiplier = 1000

hist_quotes_filter_columns = [
    "root",
    "expiration",
    "strike",
    "right",
    "bid_size",
    "bid_exchange",
    "bid",
    "ask_size",
    "ask_exchange",
    "ask",
    "datetime",
]


def interval_dates(
    underlyig_data: pd.DataFrame,
    start_date: dt.datetime,
    end_date: dt.datetime,
    chunk_size=10,
) -> list:
    i = 0
    list_of_intervals = []
    while i < len(underlyig_data):
        if i + chunk_size > len(underlyig_data):
            data_chunked = underlyig_data[
                (underlyig_data["Date"] >= start_date)
                & (underlyig_data["Date"] <= end_date)
            ]["Date"].iloc[i:]
        else:
            data_chunked = underlyig_data[
                (underlyig_data["Date"] >= start_date)
                & (underlyig_data["Date"] <= end_date)
            ]["Date"].iloc[i : i + chunk_size]
        try:
            list_of_intervals.append((data_chunked.iloc[0], data_chunked.iloc[-1]))
        except:
            pass
        i += chunk_size
    return list_of_intervals


class retriever:

    def __init__(self, tic: str):
        self.ticker = tic
        self.base_url = "http://127.0.0.1:25510/v2"
        self.url_expiration = self.base_url + "/list/expirations"
        self.url_quotes = self.base_url + "/bulk_hist/option/quote"
        self.opt_exp_schedule = None
        self.skip_frwd = None
        self.frwd_exp = None
        self.strikes_put = None
        self.strikes_call = None
        self.bulk_hist_quotes = None
        self.quotes_data = pl.DataFrame()

    def get_expirations(self, start: dt.datetime, end: dt.datetime) -> list:
        list_expirations = []
        url = self.url_expiration

        TICKER = self.ticker

        params = {"root": TICKER, "use_csv": "true"}

        while url is not None:
            response = httpx.get(url, params=params, timeout=60)
            response.raise_for_status()
            csv_reader = csv.reader(response.text.split("\n"))

            for row in csv_reader:
                if len(row) != 0 and row != ["date"]:
                    list_expirations.append(row)

            if (
                "Next-Page" in response.headers
                and response.headers["Next-Page"] != "null"
            ):
                url = response.headers["Next-Page"]
                params = None
            else:
                url = None
        self.opt_exp_schedule = [
            ql.Date(int(dt[0][6:]), int(dt[0][4:6]), int(dt[0][:4]))
            for dt in list_expirations
        ]

    def get_quotes(
        self,
        start_date: dt.datetime,
        end_date: dt.date,
        data: pd.DataFrame,
        ref_price_underlying: str | None = None,
        timeout: int = 900000,
        ivl="1m",
        option_exp_mapping: tuple | None = None,
    ) -> pl.DataFrame:
        retrieved_data = []
        skip_frwd = self.skip_frwd
        frwd_exp = self.frwd_exp
        schedule = self.opt_exp_schedule

        expiration_dates = [
            dt.datetime(dt_.year(), dt_.month(), dt_.dayOfMonth()).date()
            for dt_ in schedule
        ]
        expiration_dates = pd.DataFrame(index=expiration_dates).sort_index()
        expiration_dates.reset_index(inplace=True)
        list_mkt_date = [
            date_.date() for date_ in pd.date_range(start_date, end_date, freq="B")
        ]

        for retrieve_date in list_mkt_date:
            exp_dt = expiration_dates[expiration_dates["index"] >= retrieve_date][
                skip_frwd:frwd_exp
            ]
            if ref_price_underlying == "Close":
                exp_dt = expiration_dates[expiration_dates["index"] >= retrieve_date][
                    1 + skip_frwd : frwd_exp + 1
                ]
            for df_index, fix_date in exp_dt.iterrows():
                error_no_market_data = False
                if ref_price_underlying is None:
                    ref_price_underlying = "Open"
                elif ref_price_underlying == "Open":
                    try:
                        ref_price = data[
                            (
                                data["Date"]
                                == expiration_dates.loc[
                                    df_index + skip_frwd - frwd_exp
                                ].values[0]
                            )
                        ]["Open"].values[0]
                    except:
                        print(f"ERROR IN GETTING THE OPEN PRICE - {retrieve_date}")
                elif ref_price_underlying == "Close":
                    try:
                        ref_price = data[
                            (
                                data["Date"]
                                == expiration_dates.loc[
                                    df_index + skip_frwd - frwd_exp
                                ].values[0]
                            )
                        ]["Close"].values[0]
                    except:
                        print(f"ERROR IN GETTING THE CLOSE PRICE - {retrieve_date}")
                list_strikes = []
                for strike in [(self.strikes_put, "P"), (self.strikes_call, "C")]:
                    if (
                        not strike[0] is None
                        and isinstance(strike[0], tuple)
                        and len(strike[0]) == 2
                    ):
                        try:
                            list_strikes.append(
                                (
                                    ref_price * (1 + strike[0][0]),
                                    ref_price * (1 + strike[0][1]),
                                    strike[1],
                                )
                            )
                        except:
                            pass
                    else:
                        continue

                TICKER = self.ticker
                EXP_DATE = fix_date.values[0].strftime("%Y%m%d")
                START_DATE = retrieve_date.strftime("%Y%m%d")
                # END_DATE = date.strftime('%Y%m%d')
                END_DATE = START_DATE
                if ivl in minute_to_milliseconds.keys():
                    CANDLE_INTERVAL = minute_to_milliseconds[ivl]
                else:
                    CANDLE_INTERVAL = ivl

                params = {
                    "root": TICKER,
                    "exp": EXP_DATE,
                    "start_date": START_DATE,
                    "end_date": END_DATE,
                    "use_csv": "true",
                    "ivl": CANDLE_INTERVAL,
                }

                url = self.url_quotes
                api_response_list = []
                while url is not None:
                    response = httpx.get(url, params=params, timeout=timeout)
                    if response.status_code == 472:
                        url = None
                        error_no_market_data = True
                        break
                    if response.status_code == 474:
                        print("DISCONNECTED")
                        response.raise_for_status()
                    # response.raise_for_status()
                    csv_reader = csv.reader(response.text.split("\n"))

                    for row in csv_reader:
                        if len(row) != 0:
                            api_response_list.append(row)
                    # check the Next-Page header to see if we have more data
                    if (
                        "Next-Page" in response.headers
                        and response.headers["Next-Page"] != "null"
                    ):
                        url = response.headers["Next-Page"]
                        params = None
                    else:
                        url = None
                if error_no_market_data:
                    continue
                options_df = pl.DataFrame(api_response_list, orient="row")
                headers = options_df.head(1).to_dicts().pop()
                options_df = options_df.rename(headers)
                options_df.unique(maintain_order=True)
                options_df = options_df[1:, :]
                try:
                    options_df = options_df.cast(hist_quotes_schema)
                except:
                    print("ERROR IN CASTING THE DATA")
                options_df = options_df.with_columns(
                    pl.col("strike").truediv(strike_multiplier)
                )

                if option_exp_mapping is not None:
                    options_df = options_df.with_columns(
                        pl.lit(
                            option_exp_mapping[
                                (df_index - exp_dt.iloc[-1].name) + (len(exp_dt) - 1)
                            ]
                        ).alias("label")
                    )

                for strikes in list_strikes:
                    retrieved_data.append(
                        options_df.filter(
                            (pl.col("strike") >= strikes[0])
                            & (pl.col("strike") <= strikes[1])
                            & (pl.col("right") == strikes[2])
                        )
                    )
        try:
            self.quotes_data = pl.concat(retrieved_data, how="vertical")
            self.quotes_data = self.quotes_data.with_columns(
                pl.col("expiration").str.to_date("%Y%m%d")
            )
            self.quotes_data = self.quotes_data.with_columns(
                pl.col("date").str.to_date("%Y%m%d")
            )
            self.quotes_data = self.quotes_data.with_columns(
                pl.col("date").dt.day().alias("day")
            )
            self.quotes_data = self.quotes_data.with_columns(
                pl.col("date").dt.month().alias("month")
            )
            self.quotes_data = self.quotes_data.with_columns(
                pl.col("date").dt.year().alias("year")
            )
            self.quotes_data = self.quotes_data.with_columns(
                pl.col("ms_of_day").truediv(1000).alias("total_seconds")
            )
            self.quotes_data = self.quotes_data.with_columns(
                (pl.col("total_seconds") // 3600).alias("hours")
            )
            self.quotes_data = self.quotes_data.with_columns(
                (pl.col("total_seconds") % 3600).alias("remaining_seconds")
            )
            self.quotes_data = self.quotes_data.with_columns(
                (pl.col("remaining_seconds") // 60).alias("minutes")
            )
            self.quotes_data = self.quotes_data.with_columns(
                (pl.col("remaining_seconds") % 60).alias("seconds")
            )
            self.quotes_data = self.quotes_data.with_columns(
                pl.datetime(
                    pl.col("year"),
                    pl.col("month"),
                    pl.col("day"),
                    pl.col("hours"),
                    pl.col("minutes"),
                    pl.col("seconds"),
                    time_zone="America/New_York",
                )
            )
            if option_exp_mapping is not None:
                self.quotes_data = self.quotes_data.select(
                    hist_quotes_filter_columns + ["label"]
                )
            else:
                self.quotes_data = self.quotes_data.select(hist_quotes_filter_columns)
        except:
            print(
                f"ERROR IN CONCATENATING THE DATA AND DOING POLARS OPERATIONS: {retrieve_date} --> {EXP_DATE}"
            )


spxw_retriever = retriever(tic="SPXW")
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
start = ql.Date(31, 12, 2019)
end = ql.Date(9, 5, 2025)
yf_ticker = yf.Ticker("^SPX")

data = yf_ticker.history(
    start=dt.datetime(start.year(), start.month(), start.dayOfMonth()).date(),
    end=dt.datetime(end.year(), end.month(), end.dayOfMonth()).date()
    + dt.timedelta(days=1),
)
# data = yf_ticker.history(period="max")

data.index = data.index.tz_convert(None)
data.reset_index(inplace=True)
data["Date"] = data["Date"].dt.date
tenor = ql.Period(1, ql.Days)  # for 0DTE ql.Period(1, ql.Days)
rule = ql.DateGeneration.Forward
convention = ql.Following
terminalDateConvention = ql.Following

spxw_retriever.opt_exp_schedule = ql.MakeSchedule(
    calendar=calendar,
    effectiveDate=start,
    terminationDate=end,
    tenor=tenor,
    rule=rule,
    convention=ql.Following,
    terminalDateConvention=terminalDateConvention,
)
# spxw_retriever.skip_frwd = 0

spxw_retriever.frwd_exp = 1  # for 0DTE 1
spxw_retriever.strikes_put = (-0.05, +0.05)
spxw_retriever.strikes_call = (-0.05, +0.05)

for dt_1, dt_2 in interval_dates(
    underlyig_data=data,
    start_date=dt.datetime(2020, 1, 2).date(),
    end_date=dt.datetime(2025, 5, 12).date(),
):
    spxw_retriever.get_quotes(
        start_date=dt_1,
        end_date=dt_2,
        data=data,
        ref_price_underlying="Open",
        timeout=900000,
        ivl="30m",
    )
    spxw_retriever.quotes_data.write_parquet(f"spxw_0DTE_{str(dt_2)}.parquet")


BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

# set params
params = {"root": "SPX", "exp": "20250523", "use_csv": "true"}
#
# This is the non-streaming version, and the entire response
# will be held in memory.
#
url = BASE_URL + "/bulk_snapshot/option/open_interest"

while url is not None:
    response = httpx.get(url, params=params, timeout=900)  # make the request
    response.raise_for_status()  # make sure the request worked

    # read the entire response, and parse it as CSV
    csv_reader = csv.reader(response.text.split("\n"))

    for row in csv_reader:
        print(row)  # do something with the data

    # check the Next-Page header to see if we have more data
    if "Next-Page" in response.headers and response.headers["Next-Page"] != "null":
        url = response.headers["Next-Page"]
        params = None
    else:
        url = None


#
# This is the streaming version, and will read line-by-line
#
url = BASE_URL + "/bulk_snapshot/option/open_interest"

while url is not None:
    with httpx.stream("GET", url, params=params, timeout=60) as response:
        response.raise_for_status()  # make sure the request worked
        for line in response.iter_lines():
            print(line)  # do something with the data

    # check the Next-Page header to see if we have more data
    if "Next-Page" in response.headers and response.headers["Next-Page"] != "null":
        url = response.headers["Next-Page"]
        params = None
    else:
        url = None
