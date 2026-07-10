from __future__ import annotations

import pandas as pd

from scripts.audit_boltz_known_positive_calibration_v4 import quantiles


def test_quantiles_report_median_for_numeric_signal() -> None:
    result = quantiles(pd.Series([0.1, 0.2, 0.3]))

    assert result["min"] == 0.1
    assert result["median"] == 0.2
    assert result["max"] == 0.3
