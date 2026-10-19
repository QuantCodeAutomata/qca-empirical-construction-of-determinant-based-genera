"""
data/loader.py
==============
Downloads and caches daily stock Close prices from Yahoo Finance using yfinance.

Portfolio groups are fixed three-asset baskets; all prices are aligned to
common trading dates before being returned.
"""

import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PORTFOLIOS: dict[str, list[str]] = {
    "group1": ["KO", "AAPL", "TSLA"],
    "group2": ["GOOGL", "MSFT", "META"],
    "group3": ["JPM", "NVDA", "AMZN"],
}

START_DATE: str = "2020-01-01"
END_DATE: str = "2024-01-01"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Download daily stock prices from Yahoo Finance.

    Parameters
    ----------
    tickers:
        Ticker symbols, e.g. ``['KO', 'AAPL', 'TSLA']``.
    start:
        Start date string in ``'YYYY-MM-DD'`` format (inclusive).
    end:
        End date string in ``'YYYY-MM-DD'`` format (exclusive).
    price_col:
        Column to extract from the raw download; defaults to ``'Close'``.
        When ``auto_adjust=False`` the raw unadjusted close is labelled
        ``'Close'`` and the split/dividend-adjusted close is ``'Adj Close'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with *tickers* as columns and a ``DatetimeIndex``.
        No NaN-dropping is performed here; callers are responsible for
        alignment and cleaning.

    Raises
    ------
    ValueError
        If ``price_col`` cannot be found in the downloaded data.
    RuntimeError
        If the download returns an empty DataFrame.
    """
    ticker_list: list[str] = [tickers] if isinstance(tickers, str) else list(tickers)
    logger.info(
        "Downloading %s for %s tickers from %s to %s",
        price_col,
        len(ticker_list),
        start,
        end,
    )

    # Prefer raw (unadjusted) Close prices; fall back gracefully if the
    # keyword argument is not accepted by the installed yfinance version.
    raw: pd.DataFrame
    try:
        raw = yf.download(
            ticker_list,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
    except TypeError:
        logger.warning(
            "auto_adjust=False not accepted by installed yfinance; "
            "retrying without the argument."
        )
        raw = yf.download(ticker_list, start=start, end=end, progress=False)

    if raw.empty:
        raise RuntimeError(
            f"yfinance returned an empty DataFrame for tickers {ticker_list} "
            f"between {start} and {end}."
        )

    # yfinance >= 0.2 always returns a MultiIndex (Price, Ticker).
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        if price_col not in level0:
            raise ValueError(
                f"'{price_col}' not found in MultiIndex level-0 labels {level0}. "
                f"Available: {level0}"
            )
        prices: pd.DataFrame = raw[price_col]
    else:
        # Flat columns — single price type was returned (older API or
        # single-asset download with certain settings).
        if price_col in raw.columns:
            prices = raw[[price_col]]
        else:
            prices = raw

    # Reorder / filter columns to match the requested ticker ordering.
    available = [t for t in ticker_list if t in prices.columns]
    missing = set(ticker_list) - set(available)
    if missing:
        logger.warning("Tickers not found in download result: %s", sorted(missing))
    prices = prices[available].astype(np.float64)

    logger.info("Downloaded shape: %s", prices.shape)
    return prices


def load_portfolio_prices(
    group_name: str,
    start: str = START_DATE,
    end: str = END_DATE,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Load and clean prices for a named portfolio group.

    Rows where **any** column contains a NaN are dropped so that all assets
    share the same set of trading dates (common-date alignment).

    Parameters
    ----------
    group_name:
        Key in :data:`PORTFOLIOS`, e.g. ``'group1'``.
    start:
        Start date (inclusive), defaults to :data:`START_DATE`.
    end:
        End date (exclusive), defaults to :data:`END_DATE`.
    price_col:
        Price column to extract; defaults to ``'Close'``.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame, tickers as columns, dates as index.

    Raises
    ------
    KeyError
        If *group_name* is not in :data:`PORTFOLIOS`.
    """
    if group_name not in PORTFOLIOS:
        raise KeyError(
            f"Unknown portfolio group '{group_name}'. "
            f"Valid groups: {list(PORTFOLIOS.keys())}"
        )

    tickers = PORTFOLIOS[group_name]
    logger.info("Loading portfolio '%s': %s", group_name, tickers)

    prices = download_prices(tickers, start, end, price_col)

    n_before = len(prices)
    prices = prices.dropna(how="any")
    n_dropped = n_before - len(prices)
    if n_dropped:
        logger.warning(
            "Portfolio '%s': dropped %d row(s) containing NaN values.",
            group_name,
            n_dropped,
        )

    if prices.empty:
        raise RuntimeError(
            f"No valid price rows remain for portfolio '{group_name}' "
            f"after dropping NaN rows."
        )

    logger.info(
        "Portfolio '%s' final shape: %s  [%s → %s]",
        group_name,
        prices.shape,
        prices.index[0].date(),
        prices.index[-1].date(),
    )
    return prices


def load_all_portfolios(
    start: str = START_DATE,
    end: str = END_DATE,
    price_col: str = "Close",
) -> dict[str, pd.DataFrame]:
    """Load and clean prices for every portfolio group defined in :data:`PORTFOLIOS`.

    Parameters
    ----------
    start:
        Start date (inclusive), defaults to :data:`START_DATE`.
    end:
        End date (exclusive), defaults to :data:`END_DATE`.
    price_col:
        Price column to extract; defaults to ``'Close'``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of *group_name* → cleaned price DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for group_name in PORTFOLIOS:
        result[group_name] = load_portfolio_prices(group_name, start, end, price_col)
    return result


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    portfolios = load_all_portfolios()
    for name, df in portfolios.items():
        print(
            f"{name}: shape={df.shape}  tickers={list(df.columns)}  "
            f"dates=[{df.index[0].date()} → {df.index[-1].date()}]"
        )
