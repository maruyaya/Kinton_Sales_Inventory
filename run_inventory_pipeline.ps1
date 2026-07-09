# ============================================================
# Run inventory order suggestion pipeline
# ============================================================
# Run from project root:
#   .\run_inventory_pipeline.ps1
#
# If execution policy error appears:
#   powershell -ExecutionPolicy Bypass -File .\run_inventory_pipeline.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host "Inventory Order Suggestion Pipeline"
Write-Host "========================================"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "Project root:"
Write-Host $ProjectRoot

Write-Host ""
Write-Host "Step 1: Transform sales data..."
python .\scripts\transform_sales_data.py

Write-Host ""
Write-Host "Step 2: Check input date ranges..."

$CheckDateRangesCode = @'
import pandas as pd
from pathlib import Path

base = Path.cwd()
raw = base / "data" / "raw"
processed = base / "data" / "processed"

sales_file = processed / "sales_clean_combined.csv"
stock_file = raw / "stock_history.csv"
purchase_file = raw / "purchase_history.csv"

def clean_columns(df):
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

print("")
print("----- Date range check -----")

# Sales date range
if sales_file.exists():
    sales = pd.read_csv(sales_file, encoding="utf-8-sig")
    sales = clean_columns(sales)

    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
        sales_dates = sales["date"].dropna()

        if not sales_dates.empty:
            print(f"Sales file: {sales_file}")
            print(f"Sales date range: {sales_dates.min().date()} to {sales_dates.max().date()}")
            print(f"Sales rows: {len(sales)}")
        else:
            print(f"Sales file exists, but no valid dates found: {sales_file}")
    else:
        print(f"Sales file exists, but 'date' column was not found: {sales_file}")
else:
    print(f"Sales file not found: {sales_file}")

# Raw SalesbyItemDetail files
sales_raw_files = sorted(raw.glob("SalesbyItemDetail*.csv"))
print("")
print("Raw SalesbyItemDetail files:")
if sales_raw_files:
    for f in sales_raw_files:
        print(f"  - {f.name}")
else:
    print("  No SalesbyItemDetail*.csv files found.")

# Stock date range
print("")
if stock_file.exists():
    stock = pd.read_csv(stock_file, encoding="utf-8-sig")
    stock = clean_columns(stock)

    if "date" in stock.columns:
        stock["date"] = pd.to_datetime(stock["date"], errors="coerce")
        stock_dates = stock["date"].dropna()

        if not stock_dates.empty:
            print(f"Stock file: {stock_file}")
            print(f"Stock date range: {stock_dates.min().date()} to {stock_dates.max().date()}")
            print(f"Latest stock count date: {stock_dates.max().date()}")
            print(f"Stock rows: {len(stock)}")
        else:
            print(f"Stock file exists, but no valid dates found: {stock_file}")
    else:
        print(f"Stock file exists, but 'date' column was not found: {stock_file}")
else:
    print(f"Stock file not found: {stock_file}")

# Purchase date range
print("")
if purchase_file.exists():
    purchase = pd.read_csv(purchase_file, encoding="utf-8-sig")
    purchase = clean_columns(purchase)

    rename_map = {
        "ordered_quantity": "ordered_qty",
        "order_qty": "ordered_qty",
        "qty_ordered": "ordered_qty",
        "expected_delivery_date": "expected_delivery",
        "delivery_date": "expected_delivery",
        "received": "received_date",
    }
    purchase = purchase.rename(columns={k: v for k, v in rename_map.items() if k in purchase.columns})

    if "date" in purchase.columns:
        purchase["date"] = pd.to_datetime(purchase["date"], errors="coerce")
        order_dates = purchase["date"].dropna()

        if not order_dates.empty:
            print(f"Purchase file: {purchase_file}")
            print(f"Purchase order date range: {order_dates.min().date()} to {order_dates.max().date()}")
            print(f"Purchase rows: {len(purchase)}")
        else:
            print(f"Purchase file exists, but no valid order dates found: {purchase_file}")
    else:
        print(f"Purchase file exists, but 'date' column was not found: {purchase_file}")

    if "expected_delivery" in purchase.columns:
        purchase["expected_delivery"] = pd.to_datetime(purchase["expected_delivery"], errors="coerce")
        expected_dates = purchase["expected_delivery"].dropna()
        if not expected_dates.empty:
            print(f"Expected delivery date range: {expected_dates.min().date()} to {expected_dates.max().date()}")

    if "received_date" in purchase.columns:
        purchase["received_date"] = pd.to_datetime(purchase["received_date"], errors="coerce")
        received_dates = purchase["received_date"].dropna()
        if not received_dates.empty:
            print(f"Received date range: {received_dates.min().date()} to {received_dates.max().date()}")
        else:
            print("Received date range: no received dates found")
    else:
        print("Received date column: not found")
else:
    print(f"Purchase file not found: {purchase_file}")

print("----------------------------")
'@

$TempCheckScript = Join-Path $env:TEMP "check_inventory_date_ranges.py"
Set-Content -Path $TempCheckScript -Value $CheckDateRangesCode -Encoding UTF8

python $TempCheckScript

Remove-Item $TempCheckScript -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Step 3: Generate order suggestion..."
python .\scripts\generate_order_suggestion_stock_based.py

Write-Host ""
Write-Host "Done."
Write-Host "Output:"
Write-Host "  data\processed\order_suggestion.csv"
Write-Host ""
