"""Memory management utilities for large dataset training."""

import gc
import os
from typing import Any, Optional

import pandas as pd


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce memory usage by optimizing dtypes.
    
    Args:
        df: DataFrame to optimize
        verbose: Print memory reduction stats
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > -128 and c_max < 127:
                    df[col] = df[col].astype('int8')
                elif c_min > -32768 and c_max < 32767:
                    df[col] = df[col].astype('int16')
                elif c_min > -2147483648 and c_max < 2147483647:
                    df[col] = df[col].astype('int32')
                else:
                    df[col] = df[col].astype('int64')
            else:
                if c_min > -3.4e38 and c_max < 3.4e38:
                    df[col] = df[col].astype('float32')
                else:
                    df[col] = df[col].astype('float64')
        else:
            # Convert object to category if cardinality is low
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f'Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)')
    
    return df


def clear_memory() -> None:
    """Force garbage collection to free memory."""
    gc.collect()


def load_dataset_chunked(path: str, chunksize: int = 10000, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load large CSV in chunks to reduce memory footprint.
    
    Args:
        path: Path to CSV file
        chunksize: Number of rows per chunk
        nrows: Optional limit on total rows to read
        
    Returns:
        Combined DataFrame
    """
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, nrows=nrows, low_memory=True):
        chunk = reduce_memory_usage(chunk, verbose=False)
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    clear_memory()
    return df


def get_memory_limit() -> Optional[int]:
    """Get memory limit from environment variable (in MB).
    
    Returns:
        Memory limit in MB or None if not set
    """
    limit = os.getenv("MEMORY_LIMIT_MB")
    return int(limit) if limit else None


def check_memory_available(required_mb: float) -> bool:
    """Check if enough memory is available.
    
    Args:
        required_mb: Required memory in MB
        
    Returns:
        True if enough memory is available
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / 1024**2
        return available_mb > required_mb * 1.2  # 20% buffer
    except ImportError:
        return True  # Assume OK if psutil not available


__all__ = [
    "reduce_memory_usage",
    "clear_memory",
    "load_dataset_chunked",
    "get_memory_limit",
    "check_memory_available",
]
