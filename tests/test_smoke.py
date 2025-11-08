import os, pandas as pd

def test_demo_data():
    # Use the project's finder to locate the demo CSV so the test works
    # regardless of whether data sits under Project/src or src/
    from Project.utils.io import find_csv
    path = find_csv()
    assert os.path.exists(path), f"Dataset not found at {path}"
    df = pd.read_csv(path)
    assert 'IsInsurable' in df.columns
    assert len(df) >= 200
