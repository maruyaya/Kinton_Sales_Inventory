import pandas as pd
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STOCK_FILE = RAW_DIR / "stock_history.csv"
PURCHASE_FILE = RAW_DIR / "purchase_history.csv"

OUTPUT_ORDER_FILE = PROCESSED_DIR / "order_suggestion.csv"


# ============================================================
# Order timing
# ============================================================
ORDER_DATE = pd.Timestamp("2026-05-19")
RECEIVE_DATE = pd.Timestamp("2026-05-20")


# ============================================================
# Settings
# ============================================================
PERIOD = "W"
LOOKBACK_PERIODS = 4

MANUAL_REQUIRED_STOCK = {
    "curry": 0,
    "edamame": 2,
    "inari": 10,
    "ito togarashi": 2,
    "kizami nori": 2,
    "menma bamboo shoot": 50,
    "naruto": 0,
    "nori": 0,
    "oi ocha": 24,
    "pork broth": 4,
    "red ginger": 2,
    "tonkatsu sauce": 0,
    "wood ear mushroom": 24,
    "yuzu juice": 0,
    "sencha": 1,
}

ORDER_MULTIPLE = {
    # "oi ocha": 24,
}


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(".", "", regex=False)
    )
    return df


def clean_text(value):
    if pd.isna(value):
        return value

    value = str(value).strip().lower()

    aliases = {
        "wood ear mashroom": "wood ear mushroom",
        "wood ear": "wood ear mushroom",
        "menma": "menma bamboo shoot",
    }

    return aliases.get(value, value)


def require_columns(df: pd.DataFrame, required_cols: list[str], file_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("\nCurrent columns found in file:")
        print(df.columns.tolist())
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def round_up_to_multiple(value, multiple):
    if pd.isna(value) or value <= 0:
        return 0
    if multiple <= 0:
        return value
    return int(np.ceil(value / multiple) * multiple)


def load_stock() -> pd.DataFrame:
    stock = pd.read_csv(STOCK_FILE, encoding="utf-8-sig")
    stock = clean_column_names(stock)

    require_columns(stock, ["ingredient", "stock", "date"], STOCK_FILE.name)

    stock["ingredient"] = stock["ingredient"].apply(clean_text)
    stock["stock"] = pd.to_numeric(stock["stock"], errors="coerce")
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")

    return stock[
        stock["ingredient"].notna()
        & stock["stock"].notna()
        & stock["date"].notna()
    ].copy()


def load_purchase() -> pd.DataFrame:
    purchase = pd.read_csv(PURCHASE_FILE, encoding="utf-8-sig")
    purchase = clean_column_names(purchase)

    # Extra compatibility for common variations
    rename_map = {
        "ordered_quantity": "ordered_qty",
        "order_qty": "ordered_qty",
        "qty_ordered": "ordered_qty",
        "expected_delivery_date": "expected_delivery",
        "delivery_date": "expected_delivery",
        "received": "received_date",
    }
    purchase = purchase.rename(columns={k: v for k, v in rename_map.items() if k in purchase.columns})

    require_columns(
        purchase,
        ["date", "ingredient", "ordered_qty", "expected_delivery"],
        PURCHASE_FILE.name
    )

    purchase["ingredient"] = purchase["ingredient"].apply(clean_text)
    purchase["ordered_qty"] = pd.to_numeric(purchase["ordered_qty"], errors="coerce").fillna(0)
    purchase["date"] = pd.to_datetime(purchase["date"], errors="coerce")
    purchase["expected_delivery"] = pd.to_datetime(purchase["expected_delivery"], errors="coerce")

    # Empty received_date means this order is still pending.
    if "received_date" not in purchase.columns:
        purchase["received_date"] = pd.NaT
    else:
        purchase["received_date"] = pd.to_datetime(purchase["received_date"], errors="coerce")

    return purchase[purchase["ingredient"].notna()].copy()


def build_stock_based_consumption(stock: pd.DataFrame, purchase: pd.DataFrame) -> pd.DataFrame:
    stock = stock.sort_values(["ingredient", "date"]).copy()
    rows = []

    for ingredient, group in stock.groupby("ingredient"):
        group = group.sort_values("date").reset_index(drop=True)

        for i in range(1, len(group)):
            prev_date = group.loc[i - 1, "date"]
            curr_date = group.loc[i, "date"]
            prev_stock = group.loc[i - 1, "stock"]
            curr_stock = group.loc[i, "stock"]

            received_qty = purchase[
                (purchase["ingredient"] == ingredient)
                & (purchase["received_date"].notna())
                & (purchase["received_date"] > prev_date)
                & (purchase["received_date"] <= curr_date)
            ]["ordered_qty"].sum()

            estimated_consumption = prev_stock + received_qty - curr_stock
            consumption_for_average = max(estimated_consumption, 0)
            days_between_counts = max((curr_date - prev_date).days, 1)

            rows.append({
                "ingredient": ingredient,
                "period_start": curr_date.to_period(PERIOD).start_time,
                "consumption_for_average": consumption_for_average,
                "avg_daily_consumption_interval": consumption_for_average / days_between_counts,
                "adjustment_flag": estimated_consumption < 0,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "ingredient",
            "period_start",
            "consumption_for_average",
            "avg_daily_consumption_interval",
            "adjustment_flag",
        ])

    return pd.DataFrame(rows)


def build_recent_consumption_summary(stock_consumption: pd.DataFrame) -> pd.DataFrame:
    if stock_consumption.empty:
        return pd.DataFrame(columns=[
            "ingredient",
            "avg_daily_consumption",
            "avg_period_consumption",
            "total_recent_consumption",
            "periods_used",
            "adjustment_count_recent",
        ])

    period_df = (
        stock_consumption
        .groupby(["period_start", "ingredient"], as_index=False)
        .agg(
            consumption_for_average=("consumption_for_average", "sum"),
            avg_daily_consumption=("avg_daily_consumption_interval", "mean"),
            adjustment_count=("adjustment_flag", "sum"),
        )
    )

    recent_periods = (
        period_df["period_start"]
        .drop_duplicates()
        .sort_values()
        .tail(LOOKBACK_PERIODS)
    )

    recent = period_df[period_df["period_start"].isin(recent_periods)].copy()

    return (
        recent
        .groupby("ingredient", as_index=False)
        .agg(
            avg_daily_consumption=("avg_daily_consumption", "mean"),
            avg_period_consumption=("consumption_for_average", "mean"),
            total_recent_consumption=("consumption_for_average", "sum"),
            periods_used=("period_start", "nunique"),
            adjustment_count_recent=("adjustment_count", "sum"),
        )
    )


def build_order_suggestion(stock: pd.DataFrame, purchase: pd.DataFrame) -> pd.DataFrame:
    stock_consumption = build_stock_based_consumption(stock, purchase)
    recent_consumption = build_recent_consumption_summary(stock_consumption)

    latest_stock = (
        stock.sort_values("date")
        .groupby("ingredient", as_index=False)
        .tail(1)
        .loc[:, ["ingredient", "stock", "date"]]
        .rename(columns={"stock": "current_stock", "date": "stock_count_date"})
    )

    pending_by_receive = purchase[
        purchase["received_date"].isna()
        & purchase["expected_delivery"].notna()
        & (purchase["expected_delivery"] <= RECEIVE_DATE)
    ].copy()

    pending_summary = (
        pending_by_receive
        .groupby("ingredient", as_index=False)
        .agg(
            pending_arriving_by_receive_qty=("ordered_qty", "sum"),
            next_expected_delivery=("expected_delivery", "min"),
        )
    )

    result = pd.DataFrame({
        "ingredient": list(MANUAL_REQUIRED_STOCK.keys()),
        "required_stock": list(MANUAL_REQUIRED_STOCK.values()),
    })

    result = result.merge(latest_stock, on="ingredient", how="left")
    result = result.merge(pending_summary, on="ingredient", how="left")
    result = result.merge(recent_consumption, on="ingredient", how="left")

    fill_zero_cols = [
        "current_stock",
        "pending_arriving_by_receive_qty",
        "avg_daily_consumption",
        "avg_period_consumption",
        "total_recent_consumption",
        "periods_used",
        "adjustment_count_recent",
    ]

    for col in fill_zero_cols:
        result[col] = result[col].fillna(0)

    result["periods_used"] = result["periods_used"].astype(int)
    result["adjustment_count_recent"] = result["adjustment_count_recent"].astype(int)

    result["order_date"] = ORDER_DATE
    result["receive_date"] = RECEIVE_DATE

    result["days_from_stock_count_to_receive"] = (
        RECEIVE_DATE - result["stock_count_date"]
    ).dt.days.clip(lower=0)

    result["predicted_consumption_until_receive"] = (
        result["avg_daily_consumption"]
        * result["days_from_stock_count_to_receive"]
    )

    result["projected_stock_on_receive_before_new_order"] = (
        result["current_stock"]
        + result["pending_arriving_by_receive_qty"]
        - result["predicted_consumption_until_receive"]
    ).clip(lower=0)

    result["raw_order_suggestion"] = (
        result["required_stock"]
        - result["projected_stock_on_receive_before_new_order"]
    ).clip(lower=0)

    result["order_multiple"] = result["ingredient"].map(ORDER_MULTIPLE).fillna(1)

    result["suggested_order_qty"] = result.apply(
        lambda row: round_up_to_multiple(row["raw_order_suggestion"], row["order_multiple"]),
        axis=1
    )

    result["order_status"] = np.where(
        result["suggested_order_qty"] > 0,
        "Order",
        "OK"
    )

    result = result[
        [
            "ingredient",
            "order_status",
            "suggested_order_qty",
            "order_date",
            "receive_date",
            "current_stock",
            "stock_count_date",
            "days_from_stock_count_to_receive",
            "pending_arriving_by_receive_qty",
            "next_expected_delivery",
            "avg_daily_consumption",
            "predicted_consumption_until_receive",
            "projected_stock_on_receive_before_new_order",
            "required_stock",
            "raw_order_suggestion",
            "avg_period_consumption",
            "total_recent_consumption",
            "periods_used",
            "adjustment_count_recent",
            "order_multiple",
        ]
    ].sort_values(
        ["order_status", "suggested_order_qty", "ingredient"],
        ascending=[True, False, True]
    )

    return result


def main():
    print("Loading data...")
    stock = load_stock()
    purchase = load_purchase()

    print("Building order suggestion...")
    order_df = build_order_suggestion(stock, purchase)
    order_df.to_csv(OUTPUT_ORDER_FILE, index=False, encoding="utf-8-sig")

    print("\nCompleted.")
    print(f"Order suggestion output: {OUTPUT_ORDER_FILE}")
    print("\nPreview:")
    print(order_df.head(30))


if __name__ == "__main__":
    main()
