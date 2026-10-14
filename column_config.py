import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / 'column_mapping.json'

# Internal names the analysis code relies on. Never rename these in code;
# users change what their CSV headers must be called on the Column Settings page.
DEFAULT_COLUMNS = [
    'Date/Time',
    'Type',
    'Trade #',
    'Price USD',
    'Position size (qty)',
    'Position size (value)',
    'Net P&L USD',
    'Net P&L %',
    'Cumulative P&L USD',
    'Cumulative P&L %',
    'Run-up USD',
    'Run-up %',
    'Drawdown USD',
    'Drawdown %',
]


def load_mapping():
    """Return {internal name: name expected in the user's CSV}."""
    mapping = {col: col for col in DEFAULT_COLUMNS}
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
        for key, value in saved.items():
            if key in mapping and str(value).strip():
                mapping[key] = str(value).strip()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return mapping


def save_mapping(mapping):
    CONFIG_PATH.write_text(json.dumps(mapping, indent=2), encoding='utf-8')


def reset_mapping():
    CONFIG_PATH.unlink(missing_ok=True)


def apply_mapping(df):
    """Rename the CSV's headers to the internal names.

    Returns (df, missing) where missing is a list of (internal, expected) pairs
    whose expected header was not found in the file.
    """
    mapping = load_mapping()
    lookup = {str(c).strip().lower(): c for c in df.columns}
    rename, missing = {}, []
    for internal, expected in mapping.items():
        actual = lookup.get(expected.lower())
        if actual is None:
            missing.append((internal, expected))
        elif actual != internal:
            rename[actual] = internal
    return df.rename(columns=rename), missing
