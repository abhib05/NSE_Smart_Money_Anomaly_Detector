from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from smd.anomaly import lstm_ae
from smd.anomaly.iforest import fit_iforest, iforest_scores
from smd.anomaly.scoring import accumulation_mask
from smd.anomaly.sequences import Cube, gather, summary_vectors, valid_windows
from smd.config import IForestCfg, LstmAECfg

SMALL = LstmAECfg(hidden=16, layers=1, bottleneck=4, batch_size=16, max_epochs=3, patience=2, num_threads=1)


def _data(n=64, L=10, C=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, L, C)).astype(np.float32)


def test_output_shapes():
    m = lstm_ae.LSTMAutoencoder(n_channels=8, seq_len=30, hidden=16, layers=2, bottleneck=4)
    x = torch.zeros(5, 30, 8)
    assert m(x).shape == (5, 30, 8)
    assert m.encode(x).shape == (5, 4)
    assert lstm_ae.reconstruction_errors(m, x.numpy(), last_k=5).shape == (5,)


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    x = torch.from_numpy(_data(n=8, L=10, C=3))
    model = lstm_ae.LSTMAutoencoder(3, 10, hidden=64, layers=1, bottleneck=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(600):
        opt.zero_grad()
        loss = ((model(x) - x) ** 2).mean()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.05 * first, f"loss {loss.item():.4f} did not approach zero (start {first:.4f})"


def test_training_is_deterministic():
    x = _data()
    a = lstm_ae.train_autoencoder(x[:48], x[48:], SMALL, seed=123)
    b = lstm_ae.train_autoencoder(x[:48], x[48:], SMALL, seed=123)
    for (ka, va), (_kb, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items(), strict=True):
        assert torch.equal(va, vb), ka
    np.testing.assert_array_equal(lstm_ae.reconstruction_errors(a.model, x, 3), lstm_ae.reconstruction_errors(b.model, x, 3))


def test_anomalous_sequence_scores_higher():
    rng = np.random.default_rng(1)
    normal = (rng.normal(size=(2000, 10, 3)) * 0.3).astype(np.float32)
    cfg = LstmAECfg(hidden=32, layers=1, bottleneck=8, batch_size=64, max_epochs=8, patience=3, num_threads=1)
    res = lstm_ae.train_autoencoder(normal[:1800], normal[1800:], cfg, seed=0)
    odd = normal[:50].copy()
    odd[:, -3:, 0] += 4.0  # a burst in one channel at the end of the window
    e_norm = lstm_ae.reconstruction_errors(res.model, normal[1800:1850], last_k=5)
    e_odd = lstm_ae.reconstruction_errors(res.model, odd, last_k=5)
    assert np.median(e_odd) > 3 * np.median(e_norm)


def _cube(T=60, S=3, C=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(C, T, S)).astype(np.float32)
    return Cube(X, pd.bdate_range("2020-01-01", periods=T), pd.Index([f"S{i}" for i in range(S)]))


def test_gather_uses_only_window_ending_at_t():
    cube = _cube()
    t, s = np.array([40]), np.array([1])
    w = gather(cube, t, s, 10)
    assert w.shape == (1, 10, 2)
    np.testing.assert_array_equal(w[0, -1], cube.X[:, 40, 1])
    np.testing.assert_array_equal(w[0, 0], cube.X[:, 31, 1])
    cube.X[:, 41:, :] = 99.0  # perturb the future
    np.testing.assert_array_equal(gather(cube, t, s, 10), w)


def test_valid_windows_rules():
    cube = _cube()
    cube.X[:, 50, 0] = np.nan  # last day missing
    cube.X[:, 20:28, 2] = np.nan  # most of a window missing
    t = np.array([5, 50, 30, 30])
    s = np.array([1, 0, 2, 1])
    assert valid_windows(cube, t, s, 10).tolist() == [False, False, False, True]


def test_iforest_deterministic_and_flags_outlier():
    cube = _cube(T=200, S=5)
    t = np.repeat(np.arange(30, 200), 5)
    s = np.tile(np.arange(5), 170)
    x = summary_vectors(cube, t, s)
    a = fit_iforest(x, IForestCfg(n_estimators=50, max_samples=256), seed=1)
    b = fit_iforest(x, IForestCfg(n_estimators=50, max_samples=256), seed=1)
    np.testing.assert_array_equal(iforest_scores(a, x), iforest_scores(b, x))
    out = x[:1] + 25
    assert iforest_scores(a, out)[0] > np.quantile(iforest_scores(a, x), 0.99)


def test_accumulation_mask_direction():
    f = pd.DataFrame({
        "delivery_pct_z_60d": [2.0, 2.0, -1.0, 2.0],
        "bulk_net_buy_20d": [0.1, -0.5, 0.1, np.nan],
        "ret_5d": [0.01, 0.01, 0.01, -0.2],
        "vol_21d": [0.02, 0.02, 0.02, 0.02],
    })
    assert accumulation_mask(f).tolist() == [True, False, False, False]


def test_scores_exist_for_every_oos_date(fixture_cfg, fixture_scores):
    from smd.anomaly.scoring import anomaly_folds
    from smd.clean.panel import trading_days

    days = trading_days(fixture_cfg)
    folds = anomaly_folds(fixture_cfg, days)
    oos = days[(days >= folds[0].test_start) & (days <= folds[-1].test_end)]
    for m in ("lstm_ae", "iforest", "delivery_z_rule"):
        got = set(fixture_scores.loc[fixture_scores["model"] == m, "date"])
        assert set(oos) <= got, m
    assert fixture_scores["score"].between(0, 1).all()
    assert (fixture_scores["accumulation_score"] <= fixture_scores["score"]).all()


def test_scores_only_out_of_sample(fixture_cfg, fixture_scores):
    from smd.anomaly.scoring import anomaly_folds
    from smd.clean.panel import trading_days

    folds = {f.fold_id: f for f in anomaly_folds(fixture_cfg, trading_days(fixture_cfg))}
    for fid, g in fixture_scores.groupby("fold_id"):
        f = folds[int(fid)]
        assert g["date"].min() >= f.test_start and g["date"].max() <= f.test_end
        assert f.train_end < f.test_start


def test_planted_events_score_high(fixture_cfg, fixture_scores):
    """Sanity: planted accumulation/distribution days look anomalous to the models."""
    from tests.conftest import FIXTURES

    truth = pd.read_parquet(FIXTURES / "bronze" / "truth_events.parquet")
    panel_map = fixture_scores.merge(
        pd.read_parquet(fixture_cfg.paths.processed / "silver" / "daily_panel.parquet", columns=["date", "isin", "symbol"]),
        on=["date", "isin"],
    )
    ev = panel_map.merge(truth, on=["symbol", "date"])
    for m in ("lstm_ae", "iforest"):
        s = ev.loc[ev["model"] == m, "score"]
        if len(s) >= 5:
            assert s.mean() > 0.6, f"{m}: mean percentile on planted events {s.mean():.2f}"
