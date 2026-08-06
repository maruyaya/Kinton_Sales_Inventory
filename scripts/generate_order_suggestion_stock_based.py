import pandas as pd
from pathlib import Path
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STOCK_FILE = RAW_DIR / "stock_history.csv"
PURCHASE_FILE = RAW_DIR / "purchase_history.csv"
MAPPING_FILE = RAW_DIR / "ingredient_product_mapping_with_estimate.csv"
SALES_FILE = PROCESSED_DIR / "sales_clean_combined.csv"

OUTPUT_ORDER_FILE = PROCESSED_DIR / "order_suggestion.csv"


# ============================================================
# Order timing
# ============================================================
ORDER_DATE = pd.Timestamp("2026-08-06")
RECEIVE_DATE = pd.Timestamp("2026-08-07")


# ============================================================
# Settings
# ============================================================
PERIOD = "W"
LOOKBACK_PERIODS = 2
STD_LOOKBACK_PERIODS = 4
NEXT_WEEK_DAYS = 7
SALES_LOOKBACK_DAYS = 7

MANUAL_REQUIRED_STOCK = {
    "edamame": 2,
    "inari": 10,
    "ito togarashi": 2,
    "kizami nori": 2,
    "menma bamboo shoot": 45,
    # "naruto": 0,
    # "nori": 0,
    "oi ocha": 24,
    "pork broth": 4,
    "red ginger": 2,
    "tonkatsu sauce": 0,
    "wood ear mushroom": 24,
    "yuzu juice": 0,
    "sencha": 1,
    "hondashi": 1,
    "sesame dressing": 3,
    "wasabi": 0.5,
}

ORDER_MULTIPLE = {
    "pork broth": 1,
    "wood ear mushroom": 6,
    "menma bamboo shoot": 10,
    "edamame": 1,
    "kizami nori": 1,
}

MAX_STOCK_AFTER_DELIVERY = {
    "pork broth": 4,
    "wood ear mushroom": 24,
    "menma bamboo shoot": 45,
    "edamame": 2,
    "kizami nori": 2,
}

REORDER_POINT = {
    "edamame": 1.1,
    "kizami nori": 1.4,
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


def require_columns(df, required_cols, file_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("\nCurrent columns found in file:")
        print(df.columns.tolist())
        raise ValueError(f"{file_name} is missing required columns: {missing}")


def round_order_qty(value, multiple, ingredient):
    if pd.isna(value) or value <= 0:
        return 0

    if multiple <= 0:
        multiple = 1

    if ingredient in MAX_STOCK_AFTER_DELIVERY:
        return int(np.floor(value / multiple) * multiple)

    return int(np.ceil(value / multiple) * multiple)


def load_stock():
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


def load_purchase():
    purchase = pd.read_csv(PURCHASE_FILE, encoding="utf-8-sig")
    purchase = clean_column_names(purchase)

    rename_map = {
        "ordered_quantity": "ordered_qty",
        "order_qty": "ordered_qty",
        "qty_ordered": "ordered_qty",
        "expected_delivery_date": "expected_delivery",
        "delivery_date": "expected_delivery",
        "received": "received_date",
    }
    purchase = purchase.rename(
        columns={k: v for k, v in rename_map.items() if k in purchase.columns}
    )

    require_columns(
        purchase,
        ["date", "ingredient", "ordered_qty", "expected_delivery"],
        PURCHASE_FILE.name
    )

    purchase["ingredient"] = purchase["ingredient"].apply(clean_text)
    purchase["ordered_qty"] = pd.to_numeric(
        purchase["ordered_qty"], errors="coerce"
    ).fillna(0)
    purchase["date"] = pd.to_datetime(purchase["date"], errors="coerce")
    purchase["expected_delivery"] = pd.to_datetime(
        purchase["expected_delivery"], errors="coerce"
    )

    if "received_date" not in purchase.columns:
        purchase["received_date"] = pd.NaT
    else:
        purchase["received_date"] = pd.to_datetime(
            purchase["received_date"], errors="coerce"
        )

    return purchase[purchase["ingredient"].notna()].copy()


def load_sales():
    sales = pd.read_csv(SALES_FILE, encoding="utf-8-sig")
    sales = clean_column_names(sales)

    require_columns(sales, ["product_name", "qty_sold", "date"], SALES_FILE.name)

    sales["product"] = sales["product_name"].apply(clean_text)
    sales["qty_sold"] = pd.to_numeric(sales["qty_sold"], errors="coerce").fillna(0)
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")

    return sales[
        sales["product"].notna()
        & sales["date"].notna()
    ].copy()


def load_mapping():
    mapping = pd.read_csv(MAPPING_FILE, encoding="utf-8-sig")
    mapping = clean_column_names(mapping)

    require_columns(
        mapping,
        ["ingredient", "product", "estimated_qty_per_product"],
        MAPPING_FILE.name
    )

    mapping["ingredient"] = mapping["ingredient"].apply(clean_text)
    mapping["product"] = mapping["product"].apply(clean_text)
    mapping["estimated_qty_per_product"] = pd.to_numeric(
        mapping["estimated_qty_per_product"],
        errors="coerce"
    )

    return mapping[
        mapping["ingredient"].notna()
        & mapping["product"].notna()
        & mapping["estimated_qty_per_product"].notna()
    ].copy()


def build_stock_based_consumption(stock, purchase):
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
                "avg_daily_consumption_interval": (
                    consumption_for_average / days_between_counts
                ),
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


def build_recent_consumption_summary(stock_consumption):
    if stock_consumption.empty:
        return pd.DataFrame(columns=[
            "ingredient",
            "avg_daily_consumption",
            "std_daily_consumption",
            "std_weekly_consumption",
            "next_week_expected_consumption",
            "next_week_consumption_lower",
            "next_week_consumption_upper",
            "avg_period_consumption",
            "total_recent_consumption",
            "periods_used",
            "std_periods_used",
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

    std_periods = (
        period_df["period_start"]
        .drop_duplicates()
        .sort_values()
        .tail(STD_LOOKBACK_PERIODS)
    )

    recent = period_df[period_df["period_start"].isin(recent_periods)].copy()
    recent_for_std = period_df[period_df["period_start"].isin(std_periods)].copy()

    avg_summary = (
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

    std_summary = (
        recent_for_std
        .groupby("ingredient", as_index=False)
        .agg(
            std_daily_consumption=("avg_daily_consumption", "std"),
            std_periods_used=("period_start", "nunique"),
        )
    )

    result = avg_summary.merge(std_summary, on="ingredient", how="left")

    result["std_daily_consumption"] = result["std_daily_consumption"].fillna(0)
    result["std_periods_used"] = result["std_periods_used"].fillna(0).astype(int)

    result["std_weekly_consumption"] = (
        result["std_daily_consumption"] * np.sqrt(NEXT_WEEK_DAYS)
    )

    result["next_week_expected_consumption"] = (
        result["avg_daily_consumption"] * NEXT_WEEK_DAYS
    )

    result["next_week_consumption_lower"] = (
        result["next_week_expected_consumption"]
        - result["std_weekly_consumption"]
    ).clip(lower=0)

    result["next_week_consumption_upper"] = (
        result["next_week_expected_consumption"]
        + result["std_weekly_consumption"]
    )

    return result


def build_sales_based_consumption(sales, mapping):
    sales_start_date = ORDER_DATE - pd.Timedelta(days=SALES_LOOKBACK_DAYS - 1)

    recent_sales = sales[
        (sales["date"] >= sales_start_date)
        & (sales["date"] <= ORDER_DATE)
    ].copy()

    sales_by_product = (
        recent_sales
        .groupby("product", as_index=False)
        .agg(
            recent_sales_qty=("qty_sold", "sum"),
            sales_days=("date", "nunique")
        )
    )

    mapped_sales = mapping.merge(
        sales_by_product,
        on="product",
        how="left"
    )

    mapped_sales["recent_sales_qty"] = mapped_sales["recent_sales_qty"].fillna(0)
    mapped_sales["sales_based_consumption"] = (
        mapped_sales["recent_sales_qty"]
        * mapped_sales["estimated_qty_per_product"]
    )

    summary = (
        mapped_sales
        .groupby("ingredient", as_index=False)
        .agg(
            related_recent_sales_qty=("recent_sales_qty", "sum"),
            sales_based_period_consumption=("sales_based_consumption", "sum"),
            mapped_product_count=("product", "nunique"),
        )
    )

    summary["sales_based_daily_consumption"] = (
        summary["sales_based_period_consumption"] / SALES_LOOKBACK_DAYS
    )

    return summary


def build_order_suggestion(stock, purchase, sales, mapping):
    stock_consumption = build_stock_based_consumption(stock, purchase)
    stock_consumption_summary = build_recent_consumption_summary(stock_consumption)
    sales_consumption_summary = build_sales_based_consumption(sales, mapping)

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
    result = result.merge(stock_consumption_summary, on="ingredient", how="left")
    result = result.merge(sales_consumption_summary, on="ingredient", how="left")

    fill_zero_cols = [
        "current_stock",
        "pending_arriving_by_receive_qty",
        "avg_daily_consumption",
        "std_daily_consumption",
        "std_weekly_consumption",
        "next_week_expected_consumption",
        "next_week_consumption_lower",
        "next_week_consumption_upper",
        "avg_period_consumption",
        "total_recent_consumption",
        "periods_used",
        "std_periods_used",
        "adjustment_count_recent",
        "related_recent_sales_qty",
        "sales_based_period_consumption",
        "sales_based_daily_consumption",
        "mapped_product_count",
    ]

    for col in fill_zero_cols:
        result[col] = result[col].fillna(0)

    result["periods_used"] = result["periods_used"].astype(int)
    result["std_periods_used"] = result["std_periods_used"].astype(int)
    result["adjustment_count_recent"] = result["adjustment_count_recent"].astype(int)
    result["mapped_product_count"] = result["mapped_product_count"].astype(int)

    result["effective_daily_consumption"] = np.maximum(
        result["avg_daily_consumption"],
        result["sales_based_daily_consumption"]
    )

    result["order_date"] = ORDER_DATE
    result["receive_date"] = RECEIVE_DATE

    result["days_from_stock_count_to_receive"] = (
        RECEIVE_DATE - result["stock_count_date"]
    ).dt.days.clip(lower=0)

    result["predicted_consumption_until_receive"] = (
        result["effective_daily_consumption"]
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

    result["max_stock_after_delivery"] = result["ingredient"].map(
        MAX_STOCK_AFTER_DELIVERY
    )

    result["max_allowed_order_qty"] = (
        result["max_stock_after_delivery"]
        - result["projected_stock_on_receive_before_new_order"]
    )

    result["max_allowed_order_qty"] = result["max_allowed_order_qty"].clip(lower=0)

    for ingredient, reorder_point in REORDER_POINT.items():
        mask = result["ingredient"] == ingredient

        result.loc[
            mask
            & (result["projected_stock_on_receive_before_new_order"] >= reorder_point),
            "raw_order_suggestion"
        ] = 0

        result.loc[
            mask
            & (result["projected_stock_on_receive_before_new_order"] < reorder_point),
            "raw_order_suggestion"
        ] = (
            result.loc[mask, "max_stock_after_delivery"]
            - result.loc[mask, "projected_stock_on_receive_before_new_order"]
        )

    result["raw_order_suggestion"] = np.where(
        result["max_stock_after_delivery"].notna(),
        np.minimum(
            result["raw_order_suggestion"],
            result["max_allowed_order_qty"]
        ),
        result["raw_order_suggestion"]
    )

    result["order_multiple"] = result["ingredient"].map(ORDER_MULTIPLE).fillna(1)

    result["suggested_order_qty"] = result.apply(
        lambda row: round_order_qty(
            row["raw_order_suggestion"],
            row["order_multiple"],
            row["ingredient"]
        ),
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
            "std_daily_consumption",
            "std_weekly_consumption",
            "next_week_expected_consumption",
            "next_week_consumption_lower",
            "next_week_consumption_upper",

            "sales_based_daily_consumption",
            "effective_daily_consumption",

            "predicted_consumption_until_receive",
            "projected_stock_on_receive_before_new_order",
            "required_stock",
            "max_stock_after_delivery",
            "max_allowed_order_qty",
            "raw_order_suggestion",

            "avg_period_consumption",
            "total_recent_consumption",
            "related_recent_sales_qty",
            "sales_based_period_consumption",
            "mapped_product_count",
            "periods_used",
            "std_periods_used",
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
    sales = load_sales()
    mapping = load_mapping()

    print("Building order suggestion...")
    order_df = build_order_suggestion(stock, purchase, sales, mapping)
    order_df.to_csv(OUTPUT_ORDER_FILE, index=False, encoding="utf-8-sig")

    print("\nCompleted.")
    print(f"Order suggestion output: {OUTPUT_ORDER_FILE}")
    print("\nPreview:")
    print(order_df.head(30))


if __name__ == "__main__":
    main()