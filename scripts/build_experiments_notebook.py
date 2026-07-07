"""Generate notebooks/experiments/all_dataset_experiments.ipynb (narrative + rich EDA)."""
import pathlib

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# GBDTE — Datasets & Findings, Explored

An interactive tour for the research team. Every dataset gets a full **card** (what it is,
its columns, records, distributions, the target over time, and the diagnostic verdict);
the narrative acts weave the claims around them. Interactive Plotly renders on **nbviewer**
and locally — hover, zoom, toggle series.""")

code("""import sys, os
_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(_root, "python")); os.chdir(_root)
import numpy as np, pandas as pd
import plotly.graph_objects as go, plotly.io as pio
from IPython.display import display, Markdown
pio.renderers.default = "notebook"
from extra_boost_py.experiments import eda
from extra_boost_py.experiments.eda import load_raw, RAW_INFO, REAL_ORDER, PALETTE""")

# reusable dataset-card renderer (lives in the notebook so eda.py stays pure)
code('''def render_card(name):
    info = RAW_INFO[name]; m = eda.overview_metrics(name); v = eda.diagnostic_verdict(name)
    flag = "🟢 promoted" if v["promoted"] else "🔴 not promoted"
    display(Markdown(
        f"### {name}\\n**{info.description}**  \\n"
        f"*source:* {info.source} · *task:* {info.task}  \\n_{info.notes}_"))
    display(Markdown(
        f"`records` **{m['records']:,}** · `time span` {m['time_span']} · "
        f"`model features` {m['features']} · `train/test` {m['train']:,}/{m['test']:,}  \\n"
        f"**pre-training screen:** separability {v['separability']}, "
        f"extrap_gain {v['extrap_gain']} → {flag} ({v['why']})"))
    raw = load_raw(name)
    display(Markdown("**a peek at the raw data**")); display(raw.head(4))
    display(Markdown("**every column** (dtype, missingness, cardinality, example, meaning)"))
    display(eda.column_table(raw, name))
    eda.target_over_time_fig(name).show()
    num = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and raw[c].nunique() > 20]
    if num:
        eda.distribution_fig(raw, num[-1]).show()''')

# ---------------- Act 0 ----------------
md("""## Act 0 — The intuition in one picture

A single group whose target rises with time. Train on the past, predict the future. A
constant-leaf tree can only repeat the last level it saw (it flattens); a linear leaf
carries the trend forward. That gap is the whole paper.""")
code('''rng = np.random.default_rng(0); t = np.sort(rng.random(400))
y = 1 + 3*t + 0.1*rng.standard_normal(400); cut = 0.6; tr = t < cut
w = np.polyfit(t[tr], y[tr], 1); const = y[tr].mean()
fig = go.Figure()
fig.add_scatter(x=t, y=y, mode="markers", name="actual",
                marker=dict(color=PALETTE["muted"], size=5))
fig.add_scatter(x=t[~tr], y=np.polyval(w, t[~tr]), mode="lines", name="linear leaf (GBDTE)",
                line=dict(color=PALETTE["gbdte"], width=3))
fig.add_scatter(x=t[~tr], y=np.full((~tr).sum(), const), mode="lines", name="constant leaf",
                line=dict(color=PALETTE["red"], width=3, dash="dash"))
fig.add_vline(x=cut, line=dict(color=PALETTE["ink"], dash="dot"))
fig.update_layout(title="Constant leaves flatten; linear leaves extrapolate",
                  xaxis_title="t (time)", yaxis_title="y (target)")
eda.apply_style(fig).show()''')

# ---------------- Act 1 ----------------
md("""## Act 1 — The controllable world (synthetic)

Synthetic benchmarks let us dial the structure. **Role purity ρ** and **drift** decide
whether linear leaves help; the regime map below is the answer (blue = GBDTE wins), and the
screen predicts it from data alone.""")
code('''res = pd.read_csv("reports/article_experiments/regime_map/results.csv")
sub = res[res.metric=="rmse"].groupby(["bench","model"])["value"].mean().unstack()
RHO=[0,.25,.5,.75,1]; DR=[0,.5,1,2]; Z=np.full((len(RHO),len(DR)), np.nan)
for i,r in enumerate(RHO):
    for j,d in enumerate(DR):
        k=f"rho{r}_drift{d}"
        if k in sub.index and {"gbdte","lgbm"}<=set(sub.columns):
            Z[i,j]=np.log2(sub.loc[k,"gbdte"]/sub.loc[k,"lgbm"])
fig=go.Figure(go.Heatmap(z=Z, x=[str(d) for d in DR], y=[str(r) for r in RHO], zmid=0,
    colorscale=[[0,PALETTE["gbdte"]],[0.5,PALETTE["mid"]],[1,PALETTE["red"]]],
    colorbar=dict(title="log2 RMSE<br>GBDTE / LGBM")))
fig.update_layout(title="Regime map: blue = GBDTE wins, red = loses",
                  xaxis_title="drift", yaxis_title="role purity ρ")
eda.apply_style(fig).show()''')

md("""### Synthetic dataset cards (ground truth known)

The three generators, with their true structure visible.""")
code('''from extra_boost_py.experiments.benchgen import BenchConfig, generate
from extra_boost_py.experiments.classical_bench import classical_bench
from extra_boost_py.experiments.poisson_bench import PoissonBenchConfig, generate_poisson

def synth_card(title, bench, note, colour_by="group"):
    df = bench.df
    display(Markdown(f"#### {title}\\n_{note}_"))
    display(Markdown(f"`records` {len(df):,} · `features` {len(bench.partition_cols)} · "
                     f"`task` {bench.task} · `split cut(t)` {bench.cut:.2f}"))
    display(df.describe().T[["mean","std","min","50%","max"]])
    fig = go.Figure()
    s = df.sample(min(3000, len(df)), random_state=0)
    fig.add_scatter(x=s["t"], y=s["y"], mode="markers",
                    marker=dict(size=4, color=s[colour_by], colorscale="Blues", showscale=True,
                                colorbar=dict(title=colour_by)))
    fig.add_vline(x=bench.cut, line=dict(color=PALETTE["red"], dash="dash"))
    fig.update_layout(title=f"{title}: target vs time (colour = {colour_by})",
                      xaxis_title="t", yaxis_title="y")
    eda.apply_style(fig).show()

synth_card("MSE family (k=8, ρ=1)",
           generate(BenchConfig(task="mse", k_groups=8, omega=8.0, n_rows=4000, seed=0)),
           "Grouped drifting trend; leaf basis (1,t,sin,cos). Claim C1.")
synth_card("LogLoss benchmark", classical_bench(n_rows=8000, seed=0),
           "Time-dependent uplift with a closed-form Bayes oracle. Claim C2.", colour_by="oracle")
synth_card("Poisson benchmark",
           generate_poisson(PoissonBenchConfig(k_groups=8, n_objects=500, seed=0)),
           "Grouped drifting event intensity. Engine fixed; out of the paper's claims for now (L4).",
           colour_by="lam_true")''')

# ---------------- Act 2 ----------------
md("""## Act 2 — The datasets (the heart)

Each dataset profiled the same way. The 🟢 four are the OWID family where GBDTE wins; the
🔴 five are the honest negatives — and the screen called every one in advance. Explore the
columns, distributions, and target trajectories.""")
for name in ["owid_childmort", "owid_maternalmort", "owid_lifeexp", "owid_gdppcap",
             "weather", "sberbank", "homecredit", "latam", "gassensor"]:
    code(f'render_card("{name}")')

# ---------------- Act 3 ----------------
md("""## Act 3 — The screen, the family, the verdict""")
code('''rows=[{**eda.diagnostic_verdict(n), "dataset":n} for n in REAL_ORDER]
S=pd.DataFrame(rows); fig=go.Figure()
for prom,col in [(True,PALETTE["aqua"]),(False,PALETTE["red"])]:
    d=S[S.promoted==prom]
    fig.add_scatter(x=d["separability"], y=d["extrap_gain"], mode="markers+text",
        text=d["dataset"], textposition="top center",
        name=("promoted" if prom else "not"), marker=dict(color=col, size=13))
fig.add_hline(y=0.05, line=dict(color=PALETTE["muted"], dash="dash"))
fig.update_layout(title="The pre-training screen — extrap_gain, not separability, decides",
                  xaxis_title="separability index", yaxis_title="extrapolation_gain")
eda.apply_style(fig).show()''')
code('''fam=pd.read_csv("reports/article_experiments/hunt/family_bootstrap.csv")
order=["owid_childmort","owid_maternalmort","owid_lifeexp","owid_gdppcap"]
g=fam[fam.model=="gbdte_auto"].set_index("dataset").loc[order]
bb=fam[fam.model.isin(["xgb","lgbm","catboost"])]
bb=bb.loc[bb.groupby("dataset")["mean"].idxmin()].set_index("dataset").loc[order]
fig=go.Figure()
fig.add_bar(x=order, y=g["mean"], error_y=dict(array=g["std"]), name="GBDTE",
            marker_color=PALETTE["gbdte"])
fig.add_bar(x=order, y=bb["mean"], error_y=dict(array=bb["std"]),
            name="best standard booster", marker_color=PALETTE["muted"])
fig.update_layout(title="GBDTE vs best constant-leaf booster (lower is better)",
                  yaxis_title="test RMSE (10-bootstrap)", barmode="group")
eda.apply_style(fig).show()''')
md("""**Verdict.** GBDTE wins on 3 of 4 family panels (12–24%) and ties on GDP; the only peer
is a manual global-detrend baseline, and it beats the naive linear tree in both mean and
stability. The screen was right on all nine datasets. Honest caveats: the GDP tie, the
detrend peer, and Poisson (out of scope — a separate tree-split issue). Wording decisions
live in `gbdte_article_2026/results_claims.md`.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path("notebooks/experiments/all_dataset_experiments.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("written", out, len(cells), "cells")
