import pandas as pd
from pathlib import Path
import re


# ================================
# File paths
# ================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "sales_clean_combined.csv"


# ================================
# Find the real header row in CSV
# ================================
def find_header_row(file_path: Path) -> int:
    """
    Find the row index where the actual table header starts.
    Expected header contains columns like: Item, Date, Qty. Sold, Revenue
    """
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        for i, line in enumerate(f):
            line_clean = line.strip().lower()
            if "item" in line_clean and "date" in line_clean:
                return i

    raise ValueError(f"Could not find header row in file: {file_path.name}")


# ================================
# Product classification
# ================================
def classify_type(product_name: str) -> str:
    if pd.isna(product_name):
        return "Unknown"

    product_name = str(product_name).strip()
    name = product_name.lower()

    # Products starting with "$" mean extra toppings
    if product_name.startswith("$"):
        return "Topping"

    drink_keywords = [
        "coke", "water", "tea", "sake", "ramune",
        "lemonade", "san pellegrino", "ginger ale", "sprite"
    ]
    dessert_keywords = [
        "cheesecake", "ice cream", "mochi"
    ]
    side_keywords = [
        "gyoza", "edamame", "karaage", "takoyaki", "rice", "gohan"
    ]
    main_keywords = [
        "ramen", "donburi", "don", "miso", "shoyu",
        "tonkotsu", "spicy", "original", "beef",
        "chicken", "pork"
    ]

    if any(k in name for k in drink_keywords):
        return "Drink"
    if any(k in name for k in dessert_keywords):
        return "Dessert"
    if any(k in name for k in side_keywords):
        return "Side"
    if any(k in name for k in main_keywords):
        return "Main"

    return "Other"


# ================================
# Transform one file
# ================================
def transform_one_file(file_path: Path) -> pd.DataFrame:
    header_row = find_header_row(file_path)

    df = pd.read_csv(
        file_path,
        skiprows=header_row,
        encoding="utf-8-sig"
    )
    df.columns = df.columns.str.strip()

    # Rename original Type column to avoid conflict
    if "Type" in df.columns:
        df = df.rename(columns={"Type": "Transaction Type"})

    # Check required columns
    required_cols = ["Item", "Date", "Qty. Sold"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {file_path.name}: {missing_cols}"
        )

    # Detect product header rows
    is_product_header = (
        df["Item"].notna()
        & df["Date"].isna()
        & ~df["Item"].astype(str).str.startswith("Total -", na=False)
    )

    # Create product name column and forward fill
    df["Product name"] = df["Item"].where(is_product_header)
    df["Product name"] = df["Product name"].ffill()

    # Keep only actual sales rows
    sales = df[df["Date"].notna()].copy()

    # Parse date column
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    sales = sales[sales["Date"].notna()].copy()

    # Date-related columns
    sales["Day"] = sales["Date"].dt.day
    sales["Month"] = sales["Date"].dt.month
    sales["Year"] = sales["Date"].dt.year
    sales["Weekday"] = sales["Date"].dt.day_name()

    # Clean numeric columns
    sales["Qty. Sold"] = pd.to_numeric(sales["Qty. Sold"], errors="coerce")

    if "Revenue" in sales.columns:
        sales["Revenue"] = (
            sales["Revenue"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.strip()
        )
        sales["Revenue"] = pd.to_numeric(sales["Revenue"], errors="coerce")
    else:
        sales["Revenue"] = pd.NA

    # Create category column
    sales["Category"] = sales["Product name"].apply(classify_type)

    # Remove leading "$" from product names for readability
    sales["Product name"] = (
        sales["Product name"]
        .astype(str)
        .str.replace(r"^\$", "", regex=True)
        .str.strip()
    )

    # Keep source file name for traceability
    sales["Source File"] = file_path.name

    # Reorder columns
    result = sales[[
        "Category",
        "Product name",
        "Qty. Sold",
        "Revenue",
        "Date",
        "Day",
        "Month",
        "Year",
        "Weekday",
        "Document Number",
        "Transaction Type",
        "Source File"
    ]].copy()

    return result


# ================================
# Main process: transform all files
# ================================
def main():
    csv_files = sorted(RAW_DIR.glob("SalesbyItemDetail*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No matching CSV files found in: {RAW_DIR}")

    all_dataframes = []

    for file_path in csv_files:
        print(f"Processing: {file_path.name}")
        try:
            transformed_df = transform_one_file(file_path)
            all_dataframes.append(transformed_df)
            print(f"  -> rows: {len(transformed_df)}")
        except Exception as e:
            print(f"  -> skipped due to error: {e}")

    if not all_dataframes:
        raise ValueError("No files were successfully processed.")

    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Optional: remove duplicates
    combined_df = combined_df.drop_duplicates()

    # Sort by date
    combined_df = combined_df.sort_values(
        by=["Date", "Product name"],
        ascending=[True, True]
    )

    # Save combined result
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nTransformation completed.")
    print(f"Processed files: {len(all_dataframes)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(combined_df.head(10))


if __name__ == "__main__":
    main()