import pandas as pd
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df[[
        "dist_min_ci_0_5h",
        "alignment_abs",
        "closing_speed_m_per_h",
        "dt_first_last_0_5h",
        "num_perimeters_0_5h",
        "spread_bearing_cos",
    ]].copy()

    base["inv_dist_min"] = 1 / (base["dist_min_ci_0_5h"] + 1e-6)
    base["threat_projection"] = base["alignment_abs"] * base["inv_dist_min"]
    base["time_to_collision_est"] = base["dist_min_ci_0_5h"] / (base["closing_speed_m_per_h"] + 1e-6)
    base["directed_growth"] = base["time_to_collision_est"] * base["threat_projection"]
    base["signal_confidence"] = (
        base["num_perimeters_0_5h"] * base["dt_first_last_0_5h"] *
        (1 - (base["spread_bearing_cos"] > 0.5).astype(int))
    )

    return base