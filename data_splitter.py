import os
import pandas as pd
from sklearn.model_selection import train_test_split


def split_raw_data(input_file="Sets/sample_dummy.csv", output_dir="Sets"):
    print(f"Loading raw data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Shuffle data to remove any collection bias
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Stratified Split (85% Train, 15% Test)
    # If you have a 'domain' or 'category' column, use it for stratification
    stratify_col = "domain" if "domain" in df.columns else None

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df[stratify_col] if stratify_col else None
    )

    # We also split a small slice of train for validation (monitoring during training)
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.10,  # 10% of the training set
        random_state=42
    )

    # Save files
    train_df.to_csv(f"{output_dir}/train_set.csv", index=False)
    val_df.to_csv(f"{output_dir}/validation_set.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_set.csv", index=False)

    print(f"✓ Splitting Complete:")
    print(f"  - Train: {len(train_df)} rows (for SFT & DPO)")
    print(f"  - Val:   {len(val_df)} rows (for training checks)")
    print(f"  - Test:  {len(test_df)} rows (LOCKED for final eval)")


if __name__ == "__main__":
    split_raw_data()