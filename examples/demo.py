import polars as pl
import semantix

def run_test(name: str, data: list, instruction: str, expected_col: str):
    print(f"\n🧪 TEST: {name}")
    print(f"   Instruction: '{instruction}'")
    
    df = pl.DataFrame({expected_col: data})
    print(f"   Input: {data}")
    
    cleaned_df = semantix.clean(df, target_col=expected_col, instruction=instruction)
    
    print("   Output:")
    print(cleaned_df.select([expected_col, "clean_value", "clean_unit"]))
    return cleaned_df

def main():
    print("🚀 STARTING GENERIC CLEANING DEMO")

    # 1. Weights: Convert to kg (Normalization Test)
    # 500g -> 0.5 (if model follows instruction)
    run_test(
        name="Weights (Convert to kg)",
        data=["500g", "10kg", "1000g"],
        instruction="Convert to kg",
        expected_col="weight"
    )

    # 2. Currency: Convert to USD (Transformation Test)
    # 100 EUR -> 110 USD (assuming 1.1 extraction logic via prompt)
    run_test(
        name="Currency (Convert to USD. 1 EUR = 1.1 USD)",
        data=["100 EUR", "$50", "200eur"],
        instruction="Convert to USD. Assume 1 EUR = 1.1 USD.",
        expected_col="price"
    )

    # 3. Temperature: Convert to Celsius (Logic Test)
    # 32F -> 0C
    run_test(
        name="Temperature (Convert to Celsius)",
        data=["100C", "32F", "212 F"],
        instruction="Convert to Celsius",
        expected_col="temp"
    )

if __name__ == "__main__":
    main()
