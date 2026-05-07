import pandas as pd
import numpy as np

def reduce_mem_usage(df):
    """
    Aggressive memory compression to survive local and cloud RAM limits.
    Downcasts int and float types to their lowest possible memory footprint.
    """
    for col in df.columns:
        if df[col].dtype != object:
            if str(df[col].dtype)[:3] == 'int':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def engineer_features(df):
    """
    Extracts row-wise momentum and volatility without crossing samples.
    This provides massive signal to tree models while avoiding leakage penalties.
    """
    print("Engineering row-wise quant features...")
    
    # 1. Identify Lag columns
    lag1_cols = [c for c in df.columns if '_LagT1' in c]
    lag2_cols = [c for c in df.columns if '_LagT2' in c]
    
    # 2. Row-wise Volatility (Standard Deviation across lags for each row)
    df['row_lag1_std'] = df[lag1_cols].std(axis=1)
    df['row_lag2_std'] = df[lag2_cols].std(axis=1)
    
    # 3. Row-wise Macro Momentum (Mean movement across all features)
    df['row_lag1_mean'] = df[lag1_cols].mean(axis=1)
    
    # 4. Specific Feature Acceleration (taking subset to save RAM)
    for i in range(min(20, len(lag1_cols))):
        col1 = lag1_cols[i]
        col2 = col1.replace('_LagT1', '_LagT2')
        if col2 in df.columns:
            new_col_name = col1.replace('_LagT1', '_Momentum')
            df[new_col_name] = df[col1] - df[col2]
            
    return df