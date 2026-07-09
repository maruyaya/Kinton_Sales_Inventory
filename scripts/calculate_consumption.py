from unittest import result

import pandas as pd
from pathlib import Path
import numpy as np


# ============================================================
# File paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

STOCK_FILE = RAW_DIR / "stock_history.csv"
PURCHASE_FILE = RAW_DIR / "purchase_history.csv"
MAPPING_FILE = RAW_DIR / "ingredient_product_mapping.csv"

OUTPUT_FILE = PROCESSED_DIR / "consumption_analysis.csv"


# ============================================================
# Date range
# Change these dates when you want to analyze another period.
# ============================================================
START_DATE = pd.Timestamp("2026-05-25")
END_DATE = pd.Timestamp("2026-05-31")


# ============================================================
# Helper functions
# ============================================================
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


# ============================================================
# Load stock history
# ============================================================
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


# ============================================================
# Load purchase history
# ============================================================
def load_purchase() -> pd.DataFrame:
    if not PURCHASE_FILE.exists():
        return pd.DataFrame(columns=[
            "date",
            "ingredient",
            "ordered_qty",
            "expected_delivery",
            "received_date",
        ])

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

    if "received_date" not in purchase.columns:
        purchase["received_date"] = pd.NaT
    else:
        purchase["received_date"] = pd.to_datetime(purchase["received_date"], errors="coerce")

    return purchase[purchase["ingredient"].notna()].copy()


# ============================================================
# Find header row in Sales by Item Detail CSV
# ============================================================
def find_header_row(file_path: Path) -> int:
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            line_clean = line.strip().lower()
            if "item" in line_clean and "date" in line_clean:
                return i

    raise ValueError(f"Could not find header row in file: {file_path.name}")


# ============================================================
# Transform one Sales by Item Detail file
# ============================================================
def transform_sales_file(file_path: Path) -> pd.DataFrame:
    header_row = find_header_row(file_path)

    df = pd.read_csv(
        file_path,
        skiprows=header_row,
        encoding="utf-8-sig"
    )
    df.columns = df.columns.str.strip()

    required_cols = ["Item", "Date", "Qty. Sold"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_path.name} is missing columns: {missing_cols}")

    is_product_header = (
        df["Item"].notna()
        & df["Date"].isna()
        & ~df["Item"].astype(str).str.startswith("Total -", na=False)
    )

    df["product"] = df["Item"].where(is_product_header)
    df["product"] = df["product"].ffill()

    sales = df[df["Date"].notna()].copy()
    sales["date"] = pd.to_datetime(sales["Date"], errors="coerce")
    sales["qty_sold"] = pd.to_numeric(sales["Qty. Sold"], errors="coerce").fillna(0)
    sales["product"] = sales["product"].apply(clean_text)
    sales["source_file"] = file_path.name

    sales = sales[
        sales["date"].notna()
        & sales["product"].notna()
    ].copy()

    return sales[["date", "product", "qty_sold", "source_file"]]


# ============================================================
# Load all Sales by Item Detail files
# ============================================================
def load_sales() -> pd.DataFrame:
    sales_files = sorted(RAW_DIR.glob("SalesbyItemDetail*.csv"))

    if not sales_files:
        raise FileNotFoundError(f"No SalesbyItemDetail*.csv files found in {RAW_DIR}")

    all_sales = []

    for file_path in sales_files:
        try:
            sales = transform_sales_file(file_path)
            all_sales.append(sales)
            print(f"Loaded sales file: {file_path.name}, rows: {len(sales)}")
        except Exception as e:
            print(f"Skipped sales file: {file_path.name}, reason: {e}")

    if not all_sales:
        raise ValueError("No sales files were successfully loaded.")

    sales = pd.concat(all_sales, ignore_index=True).drop_duplicates()

    sales = sales[
        (sales["date"] >= START_DATE)
        & (sales["date"] <= END_DATE)
    ].copy()

    return sales


# ============================================================
# Load ingredient-product mapping
# ============================================================
def load_mapping() -> pd.DataFrame:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(
            f"{MAPPING_FILE} was not found. "
            "Create data/raw/ingredient_product_mapping.csv with columns: ingredient,product"
        )

    mapping = pd.read_csv(MAPPING_FILE, encoding="utf-8-sig")
    mapping = clean_column_names(mapping)

    require_columns(mapping, ["ingredient", "product"], MAPPING_FILE.name)

    mapping["ingredient"] = mapping["ingredient"].apply(clean_text)
    mapping["product"] = mapping["product"].apply(clean_text)

    mapping = mapping[
        mapping["ingredient"].notna()
        & mapping["product"].notna()
    ].drop_duplicates().copy()

    return mapping


# ============================================================
# Stock-based consumption
# ============================================================
def calculate_stock_consumption(stock: pd.DataFrame, purchase: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate stock-based consumption for each ingredient in the date range.

    Logic:
        start_stock = latest stock count on or before START_DATE
        end_stock   = latest stock count on or before END_DATE
        received_qty = purchase received between START_DATE and END_DATE

        estimated_consumption = start_stock + received_qty - end_stock

    If there is no received_date column or it is blank, received_qty will be 0.
    """
    ingredients = sorted(stock["ingredient"].dropna().unique())
    rows = []

    for ingredient in ingredients:
        ingredient_stock = stock[stock["ingredient"] == ingredient].copy()

        start_rows = ingredient_stock[ingredient_stock["date"] <= START_DATE].sort_values("date")
        end_rows = ingredient_stock[ingredient_stock["date"] <= END_DATE].sort_values("date")

        if start_rows.empty or end_rows.empty:
            continue

        start_row = start_rows.tail(1).iloc[0]
        end_row = end_rows.tail(1).iloc[0]

        start_stock = start_row["stock"]
        end_stock = end_row["stock"]
        start_stock_date = start_row["date"]
        end_stock_date = end_row["date"]

        received_qty = purchase[
            (purchase["ingredient"] == ingredient)
            & (purchase["received_date"].notna())
            & (purchase["received_date"] >= START_DATE)
            & (purchase["received_date"] <= END_DATE)
        ]["ordered_qty"].sum()

        estimated_consumption = start_stock + received_qty - end_stock

        rows.append({
            "ingredient": ingredient,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "start_stock_date": start_stock_date,
            "end_stock_date": end_stock_date,
            "start_stock": start_stock,
            "received_qty": received_qty,
            "end_stock": end_stock,
            "stock_based_consumption": estimated_consumption,
            "stock_consumption_for_average": max(estimated_consumption, 0),
            "stock_adjustment_flag": estimated_consumption < 0,
        })

    return pd.DataFrame(rows)


# ============================================================
# Sales-based related product quantity
# ============================================================
def calculate_related_sales(sales: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate related product sales by ingredient.

    This does NOT estimate exact ingredient usage unless qty_per_product exists.
    It shows how many related products were sold.
    """
    sales_by_product = (
        sales
        .groupby("product", as_index=False)
        .agg(
            related_product_sold_qty=("qty_sold", "sum"),
            sales_rows=("qty_sold", "count")
        )
    )

    related = mapping.merge(sales_by_product, on="product", how="left")
    related["related_product_sold_qty"] = related["related_product_sold_qty"].fillna(0)
    related["sales_rows"] = related["sales_rows"].fillna(0).astype(int)

    # If mapping has qty_per_product in the future, calculate sales-based consumption.
    if "qty_per_product" in mapping.columns:
        related["qty_per_product"] = pd.to_numeric(related["qty_per_product"], errors="coerce")
        related["sales_based_consumption"] = (
            related["related_product_sold_qty"]
            * related["qty_per_product"].fillna(0)
        )
    else:
        related["qty_per_product"] = pd.NA
        related["sales_based_consumption"] = pd.NA

    summary = (
        related
        .groupby("ingredient", as_index=False)
        .agg(
            related_product_sold_qty=("related_product_sold_qty", "sum"),
            matched_product_count=("product", "nunique"),
            matched_sales_rows=("sales_rows", "sum"),
            sales_based_consumption=("sales_based_consumption", "sum")
        )
    )

    if "qty_per_product" not in mapping.columns:
        summary["sales_based_consumption"] = pd.NA

    return summary


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data...")
    stock = load_stock()
    purchase = load_purchase()
    sales = load_sales()
    mapping = load_mapping()

    print("Calculating stock-based consumption...")
    stock_consumption = calculate_stock_consumption(stock, purchase)

    print("Calculating related sales by ingredient...")
    related_sales = calculate_related_sales(sales, mapping)

    result = stock_consumption.merge(
        related_sales,
        on="ingredient",
        how="outer"
    )

    result["start_date"] = result["start_date"].fillna(START_DATE)
    result["end_date"] = result["end_date"].fillna(END_DATE)

    fill_zero_cols = [
        "start_stock",
        "received_qty",
        "end_stock",
        "stock_based_consumption",
        "stock_consumption_for_average",
        "related_product_sold_qty",
        "matched_product_count",
        "matched_sales_rows",
    ]

    for col in fill_zero_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    result = result.sort_values("ingredient")

    # Estimate ingredient usage per related sold product
    result["estimated_qty_per_product"] = np.where(
        result["related_product_sold_qty"] > 0,
        result["stock_consumption_for_average"] / result["related_product_sold_qty"],
        pd.NA
    )

    # Add estimated qty back to ingredient_product_mapping
    mapping_with_estimate = mapping.merge(
        result[["ingredient", "estimated_qty_per_product"]],
        on="ingredient",
        how="left"
    )

    mapping_with_estimate.to_csv(
        RAW_DIR / "ingredient_product_mapping_with_estimate.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nCompleted.")
    print(f"Output file: {OUTPUT_FILE}")
    print("\nPreview:")
    print(result.head(30))


if __name__ == "__main__":
    main()
