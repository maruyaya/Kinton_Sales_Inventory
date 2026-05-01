import pandas as pd
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SALES_FILE = PROCESSED_DIR / "sales_clean_combined.csv"
STOCK_FILE = RAW_DIR / "stock_history.csv"
PURCHASE_FILE = RAW_DIR / "purchase_history.csv"

OUTPUT_ORDER_FILE = PROCESSED_DIR / "order_suggestion.csv"
OUTPUT_PERIOD_FILE = PROCESSED_DIR / "period_sales_stock_consumption.csv"


# ============================================================
# Settings
# ============================================================
PERIOD = "W"

# Manual required stock levels
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
    "wood ear mashroom": 24,
    "yuzu juice": 5,
    "sencha": 1,
}

# Optional: order by fixed case/bag size
ORDER_MULTIPLE = {
    # "oi ocha": 24,
    # "pork broth": 1,
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
    return str(value).strip().lower()


def require_columns(df: pd.DataFrame, required_cols: list[str], file_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def round_up_to_multiple(value, multiple):
    if pd.isna(value) or value <= 0:
        return 0
    if multiple <= 0:
        return value
    return int(np.ceil(value / multiple) * multiple)


def load_sales() -> pd.DataFrame:
    sales = pd.read_csv(SALES_FILE, encoding="utf-8-sig")
    sales = clean_column_names(sales)

    require_columns(sales, ["product_name", "qty_sold", "date"], SALES_FILE.name)

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales["qty_sold"] = pd.to_numeric(sales["qty_sold"], errors="coerce").fillna(0)
    sales["product_name"] = sales["product_name"].apply(clean_text)

    sales = sales[sales["date"].notna()].copy()
    return sales


def load_stock() -> pd.DataFrame:
    stock = pd.read_csv(STOCK_FILE, encoding="utf-8-sig")
    stock = clean_column_names(stock)

    require_columns(stock, ["ingredient", "stock", "date"], STOCK_FILE.name)

    stock["ingredient"] = stock["ingredient"].apply(clean_text)
    stock["stock"] = pd.to_numeric(stock["stock"], errors="coerce")
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")

    stock = stock[
        stock["ingredient"].notna()
        & stock["stock"].notna()
        & stock["date"].notna()
    ].copy()

    return stock


def load_purchase() -> pd.DataFrame:
    purchase = pd.read_csv(PURCHASE_FILE, encoding="utf-8-sig")
    purchase = clean_column_names(purchase)

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

    purchase = purchase[purchase["ingredient"].notna()].copy()
    return purchase


def build_stock_based_consumption(stock: pd.DataFrame, purchase: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate ingredient consumption from stock movement.

    estimated_consumption = previous_stock + received_qty - current_stock

    This is for monitoring only.
    Required stock is controlled by MANUAL_REQUIRED_STOCK.
    """
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
            adjustment_flag = estimated_consumption < 0
            consumption_for_average = max(estimated_consumption, 0)
            days_between_counts = max((curr_date - prev_date).days, 1)

            rows.append({
                "ingredient": ingredient,
                "period_start": curr_date.to_period(PERIOD).start_time,
                "previous_count_date": prev_date,
                "current_count_date": curr_date,
                "previous_stock": prev_stock,
                "current_stock": curr_stock,
                "received_qty_between_counts": received_qty,
                "estimated_consumption": estimated_consumption,
                "consumption_for_average": consumption_for_average,
                "days_between_counts": days_between_counts,
                "avg_daily_consumption_interval": consumption_for_average / days_between_counts,
                "adjustment_flag": adjustment_flag,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "ingredient",
            "period_start",
            "previous_count_date",
            "current_count_date",
            "previous_stock",
            "current_stock",
            "received_qty_between_counts",
            "estimated_consumption",
            "consumption_for_average",
            "days_between_counts",
            "avg_daily_consumption_interval",
            "adjustment_flag",
        ])

    return pd.DataFrame(rows)


def build_period_output(stock_consumption: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    if stock_consumption.empty:
        ingredient_period = pd.DataFrame(columns=[
            "period_start",
            "ingredient",
            "estimated_consumption",
            "consumption_for_average",
            "avg_daily_consumption",
            "count_intervals",
            "adjustment_count",
        ])
    else:
        ingredient_period = (
            stock_consumption
            .groupby(["period_start", "ingredient"], as_index=False)
            .agg(
                estimated_consumption=("estimated_consumption", "sum"),
                consumption_for_average=("consumption_for_average", "sum"),
                avg_daily_consumption=("avg_daily_consumption_interval", "mean"),
                count_intervals=("current_count_date", "count"),
                adjustment_count=("adjustment_flag", "sum")
            )
        )

    sales = sales.copy()
    sales["period_start"] = sales["date"].dt.to_period(PERIOD).dt.start_time

    sales_period = (
        sales
        .groupby("period_start", as_index=False)
        .agg(
            total_sold_qty=("qty_sold", "sum"),
            product_count=("product_name", "nunique")
        )
    )

    result = ingredient_period.merge(sales_period, on="period_start", how="left")
    result["total_sold_qty"] = result["total_sold_qty"].fillna(0)
    result["product_count"] = result["product_count"].fillna(0).astype(int)

    return result.sort_values(["period_start", "ingredient"])


def build_order_suggestion(
    stock: pd.DataFrame,
    purchase: pd.DataFrame,
    period_df: pd.DataFrame
) -> pd.DataFrame:
    latest_stock = (
        stock.sort_values("date")
        .groupby("ingredient", as_index=False)
        .tail(1)
        .loc[:, ["ingredient", "stock", "date"]]
        .rename(columns={"stock": "current_stock", "date": "stock_count_date"})
    )

    # Pending orders = purchase rows where received_date is blank.
    pending_orders = purchase[purchase["received_date"].isna()].copy()

    pending_summary = (
        pending_orders
        .groupby("ingredient", as_index=False)
        .agg(
            pending_order_qty=("ordered_qty", "sum"),
            next_expected_delivery=("expected_delivery", "min")
        )
    )

    if period_df.empty:
        avg_consumption = pd.DataFrame(columns=[
            "ingredient",
            "avg_period_consumption",
            "avg_daily_consumption",
            "total_recent_consumption",
            "periods_used",
            "adjustment_count_recent",
        ])
    else:
        recent_periods = (
            period_df["period_start"]
            .drop_duplicates()
            .sort_values()
            .tail(4)
        )

        recent = period_df[period_df["period_start"].isin(recent_periods)].copy()

        avg_consumption = (
            recent
            .groupby("ingredient", as_index=False)
            .agg(
                avg_period_consumption=("consumption_for_average", "mean"),
                avg_daily_consumption=("avg_daily_consumption", "mean"),
                total_recent_consumption=("consumption_for_average", "sum"),
                periods_used=("period_start", "nunique"),
                adjustment_count_recent=("adjustment_count", "sum")
            )
        )

    # Start from manual required stock list, so every target item appears.
    result = pd.DataFrame({
        "ingredient": list(MANUAL_REQUIRED_STOCK.keys()),
        "required_stock": list(MANUAL_REQUIRED_STOCK.values())
    })

    result = result.merge(latest_stock, on="ingredient", how="left")
    result = result.merge(pending_summary, on="ingredient", how="left")
    result = result.merge(avg_consumption, on="ingredient", how="left")

    fill_zero_cols = [
        "current_stock",
        "pending_order_qty",
        "avg_period_consumption",
        "avg_daily_consumption",
        "total_recent_consumption",
        "periods_used",
        "adjustment_count_recent",
    ]

    for col in fill_zero_cols:
        result[col] = result[col].fillna(0)

    result["periods_used"] = result["periods_used"].astype(int)
    result["adjustment_count_recent"] = result["adjustment_count_recent"].astype(int)

    result["effective_stock"] = result["current_stock"] + result["pending_order_qty"]
    result["raw_order_suggestion"] = (
        result["required_stock"] - result["effective_stock"]
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
            "current_stock",
            "pending_order_qty",
            "effective_stock",
            "required_stock",
            "raw_order_suggestion",
            "avg_daily_consumption",
            "avg_period_consumption",
            "total_recent_consumption",
            "periods_used",
            "stock_count_date",
            "next_expected_delivery",
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
    sales = load_sales()
    stock = load_stock()
    purchase = load_purchase()

    print("Estimating ingredient consumption from stock history...")
    stock_consumption = build_stock_based_consumption(stock, purchase)

    print("Building period-level sales and stock-consumption output...")
    period_df = build_period_output(stock_consumption, sales)
    period_df.to_csv(OUTPUT_PERIOD_FILE, index=False, encoding="utf-8-sig")

    print("Building order suggestion output with manual required stock...")
    order_df = build_order_suggestion(stock, purchase, period_df)
    order_df.to_csv(OUTPUT_ORDER_FILE, index=False, encoding="utf-8-sig")

    print("\nCompleted.")
    print(f"Period output: {OUTPUT_PERIOD_FILE}")
    print(f"Order suggestion output: {OUTPUT_ORDER_FILE}")
    print("\nPreview:")
    print(order_df.head(30))

    flagged = period_df[period_df["adjustment_count"] > 0]
    if not flagged.empty:
        print("\nNote:")
        print("Some stock intervals had negative estimated consumption.")
        print("This usually means stock correction or received items were not recorded with received_date.")
        print("Check period_sales_stock_consumption.csv and the adjustment_count column.")


if __name__ == "__main__":
    main()
