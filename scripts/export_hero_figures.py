"""Export the hero figures (screen, family, regime map) to PNG for reuse in the paper."""
import pathlib

import numpy as np
import pandas as pd

from extra_boost_py.experiments import eda
from extra_boost_py.experiments.eda import PALETTE, REAL_ORDER
import plotly.graph_objects as go

OUT = pathlib.Path("reports/article_experiments/figures")
OUT.mkdir(parents=True, exist_ok=True)


def screen_fig() -> go.Figure:
    rows = [{**eda.diagnostic_verdict(n), "dataset": n} for n in REAL_ORDER]
    s = pd.DataFrame(rows)
    fig = go.Figure()
    for prom, col in [(True, PALETTE["aqua"]), (False, PALETTE["red"])]:
        d = s[s.promoted == prom]
        fig.add_scatter(x=d["separability"], y=d["extrap_gain"], mode="markers+text",
                        text=d["dataset"], textposition="top center",
                        name=("promoted" if prom else "not"), marker=dict(color=col, size=13))
    fig.add_hline(y=0.05, line=dict(color=PALETTE["muted"], dash="dash"))
    fig.update_layout(title="The pre-training screen", xaxis_title="separability index",
                      yaxis_title="extrapolation_gain")
    return eda.apply_style(fig)


def family_fig() -> go.Figure:
    fam = pd.read_csv("reports/article_experiments/hunt/family_bootstrap.csv")
    order = ["owid_childmort", "owid_maternalmort", "owid_lifeexp", "owid_gdppcap"]
    g = fam[fam.model == "gbdte_auto"].set_index("dataset").loc[order]["mean"]
    bb = fam[fam.model.isin(["xgb", "lgbm", "catboost"])]
    bb = bb.loc[bb.groupby("dataset")["mean"].idxmin()].set_index("dataset").loc[order]["mean"]
    impr = (bb - g) / bb * 100
    lab = [d.replace("owid_", "") for d in order]
    col = [PALETTE["aqua"] if v > 1 else PALETTE["muted"] for v in impr.values]
    fig = go.Figure(go.Bar(x=impr.values, y=lab, orientation="h", marker_color=col,
                           text=[f"{v:+.0f}%" for v in impr.values], textposition="outside"))
    fig.add_vline(x=0, line=dict(color=PALETTE["ink"]))
    fig.update_layout(title="GBDTE vs best constant-leaf booster (right = GBDTE wins)",
                      xaxis_title="% test-RMSE reduction vs best standard booster")
    return eda.apply_style(fig)


def main() -> None:
    for fn, fig in [("screen.png", screen_fig()), ("family.png", family_fig())]:
        fig.write_image(str(OUT / fn), width=800, height=500, scale=2)
        print("wrote", OUT / fn)


if __name__ == "__main__":
    main()
