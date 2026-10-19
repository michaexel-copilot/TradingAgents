"""Daily coin series for the sentiment report (tradingagents.sentiment_report).

Every fetcher returns a frame indexed by naive daily dates (or raises
``SourceError``), so the report can say which sources were missing instead of
failing. Only free endpoints are used; Santiment needs a free API key.

  price/volume   Yahoo Finance (``<COIN>-USD``), CoinGecko as fallback
  funding rate   Bybit linear perpetual, Binance as fallback (8h rates, daily mean)
  open interest  Bybit linear perpetual, daily
  fear & greed   alternative.me (whole crypto market, mostly Bitcoin)
  social/onchain Santiment GraphQL (``SANTIMENT_API_KEY``): social volume,
                 weighted sentiment, daily active addresses
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests

from tradingagents.dataflows.errors import VendorError

logger = logging.getLogger(__name__)

TIMEOUT = 20.0
_UA = {"User-Agent": "tradingagents-sentiment-report/1.0"}


class SourceError(VendorError):
    """A source could not deliver data for the requested coin and window."""


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def _daily_index(values: dict, name: str) -> pd.Series:
    series = pd.Series(values, name=name, dtype="float64")
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _get_json(url: str, params: dict | None = None, headers: dict | None = None):
    try:
        response = requests.get(url, params=params, headers={**_UA, **(headers or {})}, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SourceError(f"{url.split('/')[2]}: {type(exc).__name__}: {exc}") from None


# --- price ---------------------------------------------------------------


def fetch_price_yahoo(coin: str, start: date, end: date) -> pd.DataFrame:
    import yfinance as yf

    symbol = f"{coin.upper()}-USD"
    try:
        data = yf.download(
            symbol,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
    except Exception as exc:  # yfinance raises a wide range of errors
        raise SourceError(f"Yahoo Finance: {type(exc).__name__}: {exc}") from None
    if data is None or data.empty:
        raise SourceError(f"Yahoo Finance: keine Daten für {symbol}")
    data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
    frame = data.rename(columns={"Close": "close", "High": "high", "Low": "low", "Volume": "volume"})
    return frame[["close", "high", "low", "volume"]].astype("float64")


def coingecko_id(coin: str) -> str:
    """The CoinGecko id for a ticker: the highest-ranked search hit with that symbol."""
    data = _get_json("https://api.coingecko.com/api/v3/search", {"query": coin}, _coingecko_headers())
    hits = [c for c in data.get("coins", []) if c.get("symbol", "").upper() == coin.upper()]
    if not hits:
        raise SourceError(f"CoinGecko: kein Coin mit Symbol {coin}")
    hits.sort(key=lambda c: c.get("market_cap_rank") or 10**9)
    return hits[0]["id"]


def _coingecko_headers() -> dict:
    key = os.environ.get("COINGECKO_API_KEY")
    return {"x-cg-demo-api-key": key} if key else {}


def fetch_price_coingecko(coin: str, start: date, end: date) -> pd.DataFrame:
    cg_id = coingecko_id(coin)
    data = _get_json(
        f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart/range",
        {"vs_currency": "usd", "from": _ms(start) // 1000, "to": _ms(end + timedelta(days=1)) // 1000},
        _coingecko_headers(),
    )
    prices = {pd.Timestamp(ts, unit="ms").normalize(): v for ts, v in data.get("prices", [])}
    volumes = {pd.Timestamp(ts, unit="ms").normalize(): v for ts, v in data.get("total_volumes", [])}
    if not prices:
        raise SourceError(f"CoinGecko: keine Kursdaten für {cg_id}")
    close = _daily_index(prices, "close")
    frame = pd.DataFrame({"close": close, "high": close, "low": close})
    frame["volume"] = _daily_index(volumes, "volume")
    return frame


def fetch_price(coin: str, start: date, end: date) -> tuple[pd.DataFrame, str]:
    errors = []
    for name, fetch in (("Yahoo Finance", fetch_price_yahoo), ("CoinGecko", fetch_price_coingecko)):
        try:
            return fetch(coin, start, end), name
        except SourceError as exc:
            errors.append(str(exc))
    raise SourceError("; ".join(errors))


# --- derivatives -----------------------------------------------------------


def _perp_symbol(coin: str) -> str:
    return f"{coin.upper()}USDT"


def fetch_funding_bybit(coin: str, start: date, end: date) -> pd.Series:
    rates: dict[pd.Timestamp, list[float]] = {}
    cursor_end = _ms(end + timedelta(days=1))
    start_ms = _ms(start)
    for _ in range(50):  # 200 rates per page, 3 per day: ~9 years at most
        data = _get_json(
            "https://api.bybit.com/v5/market/funding/history",
            {"category": "linear", "symbol": _perp_symbol(coin), "startTime": start_ms,
             "endTime": cursor_end, "limit": 200},
        )
        if data.get("retCode") != 0:
            raise SourceError(f"Bybit: {data.get('retMsg')}")
        rows = data.get("result", {}).get("list", [])
        if not rows:
            break
        for row in rows:
            ts = int(row["fundingRateTimestamp"])
            rates.setdefault(pd.Timestamp(ts, unit="ms").normalize(), []).append(float(row["fundingRate"]))
        oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
        if oldest <= start_ms or len(rows) < 200:
            break
        cursor_end = oldest - 1
    if not rates:
        raise SourceError(f"Bybit: keine Funding-Daten für {_perp_symbol(coin)}")
    # Percent per 8h interval, averaged over the day.
    return _daily_index({k: 100 * sum(v) / len(v) for k, v in rates.items()}, "funding")


def fetch_funding_binance(coin: str, start: date, end: date) -> pd.Series:
    data = _get_json(
        "https://fapi.binance.com/fapi/v1/fundingRate",
        {"symbol": _perp_symbol(coin), "startTime": _ms(start), "endTime": _ms(end + timedelta(days=1)),
         "limit": 1000},
    )
    if not isinstance(data, list) or not data:
        raise SourceError(f"Binance: keine Funding-Daten für {_perp_symbol(coin)}")
    rates: dict[pd.Timestamp, list[float]] = {}
    for row in data:
        rates.setdefault(pd.Timestamp(int(row["fundingTime"]), unit="ms").normalize(), []).append(
            float(row["fundingRate"])
        )
    return _daily_index({k: 100 * sum(v) / len(v) for k, v in rates.items()}, "funding")


def fetch_funding(coin: str, start: date, end: date) -> tuple[pd.Series, str]:
    errors = []
    for name, fetch in (("Bybit", fetch_funding_bybit), ("Binance", fetch_funding_binance)):
        try:
            return fetch(coin, start, end), name
        except SourceError as exc:
            errors.append(str(exc))
    raise SourceError("; ".join(errors))


def fetch_open_interest(coin: str, start: date, end: date) -> tuple[pd.Series, str]:
    values: dict[pd.Timestamp, float] = {}
    cursor = ""
    for _ in range(20):
        params = {"category": "linear", "symbol": _perp_symbol(coin), "intervalTime": "1d",
                  "startTime": _ms(start), "endTime": _ms(end + timedelta(days=1)), "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = _get_json("https://api.bybit.com/v5/market/open-interest", params)
        if data.get("retCode") != 0:
            raise SourceError(f"Bybit: {data.get('retMsg')}")
        result = data.get("result", {})
        for row in result.get("list", []):
            values[pd.Timestamp(int(row["timestamp"]), unit="ms").normalize()] = float(row["openInterest"])
        cursor = result.get("nextPageCursor") or ""
        if not cursor:
            break
    if not values:
        raise SourceError(f"Bybit: kein Open Interest für {_perp_symbol(coin)}")
    # In coins; the report converts to USD with the close price.
    return _daily_index(values, "oi_coins"), "Bybit"


# --- market sentiment ------------------------------------------------------


def fetch_fear_greed(start: date, end: date) -> tuple[pd.Series, str]:
    data = _get_json("https://api.alternative.me/fng/", {"limit": 0, "format": "json"})
    values = {
        pd.Timestamp(int(row["timestamp"]), unit="s").normalize(): float(row["value"])
        for row in data.get("data", [])
    }
    series = _daily_index(values, "fng")
    series = series[(series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))]
    if series.empty:
        raise SourceError("alternative.me: keine Fear-&-Greed-Werte im Zeitraum")
    return series, "alternative.me"


# --- Santiment -------------------------------------------------------------

SANTIMENT_URL = "https://api.santiment.net/graphql"
SANTIMENT_METRICS = {
    "social_volume_total": "social_volume",
    "sentiment_weighted_total": "sentiment",
    "daily_active_addresses": "active_addresses",
}


def _santiment(query: str, key: str) -> dict:
    try:
        response = requests.post(
            SANTIMENT_URL, json={"query": query},
            headers={**_UA, "Authorization": f"Apikey {key}"}, timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise SourceError(f"Santiment: {type(exc).__name__}") from None
    if payload.get("errors"):
        raise SourceError(f"Santiment: {payload['errors'][0].get('message', 'Fehler')}")
    return payload.get("data") or {}


def santiment_slug(coin: str, key: str) -> str:
    data = _santiment("{ allProjects { slug ticker } }", key)
    hits = [p["slug"] for p in data.get("allProjects", []) if (p.get("ticker") or "").upper() == coin.upper()]
    if not hits:
        raise SourceError(f"Santiment: kein Projekt mit Ticker {coin}")
    return hits[0]


def fetch_santiment(coin: str, start: date, end: date, slug: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Social volume, weighted sentiment and active addresses, as far as the plan allows.

    The free plan hides the most recent 30 days of some metrics; those come
    back as an error per metric, listed in the second return value.
    """
    key = os.environ.get("SANTIMENT_API_KEY")
    if not key:
        raise SourceError("Santiment: SANTIMENT_API_KEY nicht gesetzt (kostenloser Key: app.santiment.net)")
    slug = slug or santiment_slug(coin, key)
    columns, problems = {}, []
    for metric, column in SANTIMENT_METRICS.items():
        query = (
            f'{{ getMetric(metric: "{metric}") {{ timeseriesData(slug: "{slug}", '
            f'from: "{start.isoformat()}T00:00:00Z", to: "{end.isoformat()}T23:59:59Z", '
            f'interval: "1d") {{ datetime value }} }} }}'
        )
        try:
            rows = _santiment(query, key)["getMetric"]["timeseriesData"]
        except (SourceError, KeyError, TypeError) as exc:
            problems.append(f"{metric}: {exc}")
            continue
        values = {pd.Timestamp(r["datetime"]).tz_localize(None).normalize(): r["value"] for r in rows}
        if values:
            columns[column] = _daily_index(values, column)
    if not columns:
        raise SourceError("Santiment: " + "; ".join(problems))
    return pd.DataFrame(columns), problems


# --- current social snapshot (not archived) --------------------------------


def social_snapshot(coin: str, end: date) -> dict[str, str]:
    """StockTwits and Reddit posts of the last 7 days, from TradingAgents' own fetchers.

    These feeds only serve recent posts, so the snapshot is only taken when the
    report window ends within the last week.
    """
    if (date.today() - end).days > 7:
        return {}
    from tradingagents.dataflows.vendors.reddit import CRYPTO_SUBREDDITS, fetch_reddit_posts
    from tradingagents.dataflows.vendors.stocktwits import fetch_stocktwits_messages

    # Spelled out rather than as "<COIN>-USD": the shared symbol rules only know
    # the large coins and would send an altcoin to the stock feeds.
    coin = coin.upper()
    window = {"start_date": (end - timedelta(days=7)).isoformat(), "end_date": end.isoformat()}
    return {
        "stocktwits": fetch_stocktwits_messages(f"{coin}.X", limit=30, **window),
        "reddit": fetch_reddit_posts(coin, CRYPTO_SUBREDDITS, **window),
    }


# --- coin universe ---------------------------------------------------------

# The list behind the bubbles on cryptobubbles.net: the top 1000 coins by market cap.
CRYPTOBUBBLES_URL = "https://cryptobubbles.net/backend/data/bubbles1000.usd.json"

# Stablecoins have no sentiment cycle worth trading. cryptobubbles marks them
# with ``stable``; this list catches the large ones when the flag is missing
# (and in the CoinGecko fallback, which has no flag).
STABLECOINS = frozenset({
    "USDT", "USDC", "DAI", "USDE", "USDS", "FDUSD", "TUSD", "PYUSD", "USDD", "FRAX", "BUSD", "USDP",
    "GUSD", "LUSD", "USD0", "USD1", "RLUSD", "USDG", "USDX", "USDB", "USDY", "USTC", "CRVUSD", "GHO",
    "SUSD", "SUSDE", "SUSDS", "SDAI", "EURC", "EURS", "EURT", "EURI", "AEUR", "XSGD", "BRZ", "USDF",
    "DOLA", "MIM", "USDL", "AUSD", "USDTB", "BUIDL", "USYC", "OUSG",
})


def parse_cryptobubbles(data) -> list[dict]:
    """``[{id, name, rank}]`` from the cryptobubbles JSON, one entry per ticker, by rank,
    without stablecoins."""
    rows = data.get("data", []) if isinstance(data, dict) else data
    coins: dict[str, dict] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        name = str(row.get("name") or "").strip()
        rank = row.get("rank")
        if not symbol or not name or not isinstance(rank, int | float):
            continue
        if row.get("stable") or symbol in STABLECOINS:
            continue
        entry = {"id": symbol, "name": name, "rank": int(rank)}
        # Two coins can share a ticker; the larger one keeps it.
        if symbol not in coins or entry["rank"] < coins[symbol]["rank"]:
            coins[symbol] = entry
    return sorted(coins.values(), key=lambda c: c["rank"])


def fetch_coin_universe() -> tuple[list[dict], str]:
    """Coins to pick from: cryptobubbles.net, or CoinGecko's top 1000 when it is unreachable."""
    errors = []
    try:
        coins = parse_cryptobubbles(_get_json(os.environ.get("CRYPTOBUBBLES_URL", CRYPTOBUBBLES_URL)))
        if coins:
            return coins, "cryptobubbles.net"
        errors.append("cryptobubbles.net: unerwartetes Format")
    except SourceError as exc:
        errors.append(str(exc))
    rows = []
    try:
        for page in range(1, 5):
            rows += _get_json(
                "https://api.coingecko.com/api/v3/coins/markets",
                {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page},
                _coingecko_headers(),
            )
    except SourceError as exc:
        errors.append(str(exc))
    coins = parse_cryptobubbles([
        {"symbol": r.get("symbol"), "name": r.get("name"), "rank": r.get("market_cap_rank")}
        for r in rows if isinstance(r, dict)
    ])
    if coins:
        return coins, "CoinGecko (cryptobubbles.net nicht erreichbar)"
    raise SourceError("; ".join(errors))
