from __future__ import annotations
import io
import pandas as pd
from typing import Tuple, List

EXPECTED_SHEET = 0  # first sheet by default

def read_excel(file_stream: io.BytesIO) -> pd.DataFrame:
    df = pd.read_excel(file_stream, sheet_name=EXPECTED_SHEET, engine="openpyxl")
    # Strip columns, unify spaces
    df.columns = [str(c).strip() for c in df.columns]
    return df

def validate_columns(df: pd.DataFrame, required: set) -> Tuple[bool, List[str], List[str]]:
    found = set(df.columns)
    missing = sorted(list(required - found))
    extras = sorted(list(found - required))
    return (len(missing) == 0, missing, extras)
