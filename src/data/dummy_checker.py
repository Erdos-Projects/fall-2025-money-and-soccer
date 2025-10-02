import pandas as pd

def main():
    merged_csv = "data/processed/merged/players_transfer_outcomes.csv"
    merged = pd.read_csv(merged_csv)

    target = "DeclineFlag"
    subset = merged[merged[target].notnull()]

    # --- shape ---
    print(f"\nDataset shape (all rows): {merged.shape}")
    print(f"Dataset shape (subset where {target} not null): {subset.shape}")

    # --- target distribution ---
    print(f"\n=== Distribution of target '{target}' ===")
    print(subset[target].value_counts(dropna=False))
    print("\n=== Distribution (normalized %) ===")
    print(subset[target].value_counts(normalize=True).round(3) * 100)

    # Ensure pandas prints all columns
    pd.set_option("display.max_rows", None)   # show all rows
    pd.set_option("display.max_columns", None)  # show all columns
    pd.set_option("display.width", None)     # don't wrap lines

    print("\n=== Column names ===")
    print(list(subset.columns))

    print("\n=== Null values per column (subset) ===")
    print(subset.isnull().sum())

    print("\n=== Percentage of nulls per column (subset) ===")
    print((subset.isnull().mean() * 100).round(2))

if __name__ == "__main__":
    main()
