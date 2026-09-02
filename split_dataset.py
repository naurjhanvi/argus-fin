import pandas as pd
import os

input_file = r"C:\Users\ranij\Downloads\HI-Medium_Trans.csv"
train_file = r"C:\Users\ranij\projects\Argus_Fin\train_normal.csv"
diag_file = r"C:\Users\ranij\projects\Argus_Fin\diagnostic_mixed.csv"

print(f"Reading {input_file} in chunks...")

normal_rows = []
fraud_rows = []

# Read in chunks to prevent memory issues with large files
for chunk in pd.read_csv(input_file, chunksize=100000):
    # Depending on exactly how the column is named, handle spacing
    laundering_col = 'Is Laundering' if 'Is Laundering' in chunk.columns else 'IsLaundering'
    
    # Collect normal rows until we have enough for training (e.g., 20,000) and some for diagnostics (10,000)
    if len(normal_rows) < 30000:
        normals = chunk[chunk[laundering_col] == 0]
        normal_rows.append(normals)
        
    # Collect all fraud rows we can find (or up to a reasonable limit)
    frauds = chunk[chunk[laundering_col] == 1]
    if not frauds.empty:
        fraud_rows.append(frauds)
        
    # Stop early if we have enough of both to build a good demo
    if len(normal_rows) >= 1 and sum(len(f) for f in fraud_rows) > 1000:
        break

print("Processing extracted rows...")

# Concatenate collected chunks
df_normal = pd.concat(normal_rows, ignore_index=True)
df_fraud = pd.concat(fraud_rows, ignore_index=True) if fraud_rows else pd.DataFrame(columns=df_normal.columns)

# 1. Build train_normal.csv (Only normal data, max 20,000 rows is plenty for the app)
train_df = df_normal.head(20000)
train_df.to_csv(train_file, index=False)
print(f"Created {train_file} with {len(train_df)} normal transactions.")

# 2. Build diagnostic_mixed.csv (Mix of normal and fraud data)
# Take 5000 normal and up to 2000 fraud
diag_normal = df_normal.iloc[20000:25000] if len(df_normal) > 25000 else df_normal.tail(5000)
diag_fraud = df_fraud.head(2000)

diagnostic_df = pd.concat([diag_normal, diag_fraud], ignore_index=True)
# Shuffle the rows so fraud is distributed
diagnostic_df = diagnostic_df.sample(frac=1, random_state=42).reset_index(drop=True)
diagnostic_df.to_csv(diag_file, index=False)
print(f"Created {diag_file} with {len(diagnostic_df)} mixed transactions (Fraud count: {len(diag_fraud)}).")

print("Done! You can now upload these to the Streamlit app.")
