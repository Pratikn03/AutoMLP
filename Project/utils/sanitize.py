import re

def safe_col(name: str) -> str:
    # Remove JSON-breaking chars for LightGBM; keep alnum + _
    s = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"

def sanitize_columns(df):
    df = df.copy()
    new_cols = []
    keep_indices = []
    seen = {}

    for idx, col in enumerate(df.columns):
        cleaned = safe_col(col)
        if cleaned.lower().startswith("unnamed"):
            continue

        if cleaned in seen:
            seen[cleaned] += 1
            cleaned = f"{cleaned}_{seen[cleaned]}"
        else:
            seen[cleaned] = 0

        new_cols.append(cleaned)
        keep_indices.append(idx)

    if keep_indices:
        df = df.iloc[:, keep_indices]
        df.columns = new_cols
    else:
        df = df.iloc[:, 0:0]
    return df
