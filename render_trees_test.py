import numpy as np
from pathlib import Path

from extra_boost_py import BoosterParams, ExtraBooster
from extra_boost_py.go_lib import build_shared

build_shared()

rows = 20
features_inter = np.zeros((rows, 3))
t = np.linspace(0, 1, rows)
features_extra = np.column_stack(
    [np.ones_like(t), t, np.sin(50 * t), np.cos(50 * t)]
)
target = 0.2 + 0.5 * t + 0.1 * np.sin(50 * t) + 0.3 * np.cos(50 * t)

params = BoosterParams(n_stages=1, learning_rate=1.0, max_depth=1, loss="mse")
booster = ExtraBooster.train(features_inter, features_extra, target, params=params)
print("trained")
Path("graphs").mkdir(exist_ok=True)
booster.render_trees(prefix="tree", figure_type="svg", directory="graphs")
print("rendered")
booster.close()
