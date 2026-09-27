"""Discrete-time competing-risks survival for placements: fit, predict 6/12/18/24-month risk, evaluate.

Works on any ATS-style export mapped to one row per placement:
  duration  months from start to exit, or to the snapshot if still there (float)
  cause     0 = still there or planned contract end (censored), 1 = quit, 2 = dismissed, 3 = converted
plus feature columns. Check-ins are time-varying: day-30 answers are known from month 2 and day-90
answers from month 4, so a check-in model only predicts forward from month 4.

`python pipeline.py` simulates data and runs the recovery checks (asserts).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HORIZONS = (6, 12, 18, 24)
BAD = (1, 2)                                  # quit or dismissed: the exits that cost an agency
MAX_M = 24
MONTH_EDGES = [0, 1, 2, 3, 6, 9, 12, 18, 24]  # piecewise-constant baseline hazard
CAUSE = {"quit": 1, "dismissed": 2, "converted": 3}
DPM = 30.44


def from_ats(pl, snapshot):
    """ATS export (start_date, end_date, exit_reason) -> duration + cause."""
    out = pl.copy()
    out["duration"] = ((pl.end_date.fillna(snapshot) - pl.start_date).dt.days / DPM).clip(lower=0.01)
    out["cause"] = pl.exit_reason.map(CAUSE).fillna(0).astype(int)
    return out


def manager_history(df, window=6, k=3):
    """Smoothed share of the manager's earlier placements that ended badly within `window` months,
    counting only placements whose window had closed by this start date (no peeking)."""
    early_bad = (df.cause.isin(BAD) & (df.duration <= window)).to_numpy(float)
    known_at = (df.start_date + pd.Timedelta(days=window * DPM)).to_numpy()
    p0 = early_bad.mean()
    out = np.full(len(df), p0)
    for _, pos in df.reset_index(drop=True).groupby(df.manager_id.to_numpy()).indices.items():
        order = pos[np.argsort(known_at[pos])]
        csum = np.concatenate([[0], np.cumsum(early_bad[order])])
        n_prior = np.searchsorted(known_at[order], df.start_date.to_numpy()[pos], side="right")
        out[pos] = (csum[n_prior] + k * p0) / (n_prior + k)
    return out


def person_month(df):
    """One row per placement per month at risk (to month 24); y = cause in the exit month, else 0.
    Optional `entry` column = months already worked when the export starts covering the placement
    (delayed entry). Those months are skipped, not counted as survived: an export of "everyone
    employed since 2020" silently drops early leavers from before 2020."""
    exit_m = np.ceil(df.duration.to_numpy()).clip(1).astype(int)
    last = exit_m.clip(max=MAX_M)
    first = np.floor(df.entry.to_numpy()).astype(int) + 1 if "entry" in df else np.ones(len(df), int)
    idx = np.repeat(np.arange(len(df)), np.clip(last - first + 1, 0, None))
    pm = df.iloc[idx].reset_index(drop=True)
    pm["month"] = np.concatenate([np.arange(f, l + 1) for f, l in zip(first, last)])
    pm["y"] = np.where(pm.month.to_numpy() == exit_m[idx], df.cause.to_numpy()[idx], 0)
    return pm


def checkin_state(ci, rows):
    """Latest check-in known at the start of each row's month; zeros + a flag when not answered."""
    day = np.where(rows.month >= 4, 90, np.where(rows.month >= 2, 30, -1))
    s = pd.DataFrame({"placement_id": rows.placement_id.to_numpy(), "day": day}).merge(
        ci, on=["placement_id", "day"], how="left")
    seen = s.responded.fillna(0).to_numpy() == 1
    return pd.DataFrame({
        "ci_missing": ((day > 0) & ~seen).astype(float),         # non-response is itself a signal
        "quit_intent": np.where(seen, s.quit_intent - 3, 0),
        "thriving": np.where(seen, (s.vitality + s.learning) / 2 - 3, 0),
        "undermining": np.where(seen, s.undermining, 0),
    })


class HazardModel:
    """Multinomial logistic hazard: P(quit / dismissed / converted in month m | still there at m)."""

    def __init__(self, cats=(), nums=(), checkins=None, C=1.0, learner=None):
        """learner: any sklearn classifier with predict_proba (default: multinomial logistic regression)."""
        self.cats, self.nums, self.ci, self.C, self.cols, self.learner = list(cats), list(nums), checkins, C, None, learner

    def _X(self, rows):
        parts = [pd.get_dummies(rows[self.cats].astype(str), dtype=float),
                 rows[self.nums].astype(float),
                 pd.get_dummies(pd.cut(rows.month, MONTH_EDGES, labels=False), prefix="m", dtype=float)]
        if self.ci is not None:
            parts.append(checkin_state(self.ci, rows))
        X = pd.concat(parts, axis=1)
        self.cols = self.cols or list(X.columns)
        return X.reindex(columns=self.cols, fill_value=0.0).to_numpy()

    def fit(self, df):
        pm = person_month(df)
        X = self._X(pm)
        self.scaler = StandardScaler().fit(X)
        self.clf = (self.learner or LogisticRegression(C=self.C, max_iter=5000)).fit(self.scaler.transform(X), pm.y)
        return self

    def risk(self, df, from_month=1):
        """P(bad exit by month t | still there at the start of from_month), t = from_month..24.
        Built from monthly hazards, so it can only rise with t: horizons never contradict."""
        if self.ci is not None and from_month < 4:
            raise ValueError("check-in model predicts forward from month 4 (after the day-90 check-in)")
        months = np.arange(from_month, MAX_M + 1)
        rows = df.iloc[np.repeat(np.arange(len(df)), len(months))].reset_index(drop=True)
        rows["month"] = np.tile(months, len(df))
        p = self.clf.predict_proba(self.scaler.transform(self._X(rows))).reshape(len(df), len(months), -1)
        cls = list(self.clf.classes_)
        stay = p[..., cls.index(0)]
        bad = sum(p[..., cls.index(c)] for c in BAD if c in cls)
        alive_before = np.cumprod(np.hstack([np.ones((len(df), 1)), stay[:, :-1]]), axis=1)
        return np.cumsum(alive_before * bad, axis=1)


def censor_surv(duration, cause):
    """Kaplan-Meier of the censoring distribution: G(t) = P(still under observation at t)."""
    t = np.unique(duration[cause == 0])
    n_at = (duration[None, :] >= t[:, None]).sum(1)
    d = (duration[cause == 0][None, :] == t[:, None]).sum(1)
    G = np.concatenate([[1.0], np.cumprod(1 - d / n_at)])
    return lambda x, left=False: G[np.searchsorted(t, x, side="left" if left else "right")]


def evaluate(duration, cause, risk_t, t, min_beyond=30):
    """IPCW metrics for P(bad exit by t): censored-before-t cases get weight 0, everyone else is
    re-weighted by 1/G. Returns None when too few placements are observed past t to judge it."""
    duration, cause = np.asarray(duration), np.asarray(cause)
    if (duration > t).sum() < min_beyond:
        return None
    G = censor_surv(duration, cause)
    by_t = duration <= t
    w = np.where(by_t, 1 / np.maximum(G(duration, left=True), 0.05), 1 / max(G(t), 0.05))
    w = w * (~by_t | (cause != 0))
    y = (by_t & np.isin(cause, BAD)).astype(float)
    k = w > 0
    q = pd.qcut(risk_t, 5, labels=False, duplicates="drop")
    calib = [(round(risk_t[q == g].mean(), 3), round((w * y)[q == g].sum() / w[q == g].sum(), 3))
             for g in np.unique(q)]
    return dict(t=t, n_beyond=int((duration > t).sum()),
                auc=round(roc_auc_score(y[k], risk_t[k], sample_weight=w[k]), 3),
                brier=round(float((w * (y - risk_t) ** 2).sum() / w.sum()), 4),
                predicted=round(float(risk_t.mean()), 3), observed=round(float((w * y).sum() / w.sum()), 3),
                calib_by_quintile=calib)


def report(name, duration, cause, r, from_month=1):
    print(f"\n{name}")
    for t in HORIZONS:
        if t < from_month:
            continue
        m = evaluate(duration, cause, r[:, t - from_month], t)
        print(f"  {t:>2}m: " + ("not evaluable yet (too few placements observed this long)" if m is None else
              f"AUC {m['auc']}  Brier {m['brier']}  predicted {m['predicted']} vs observed {m['observed']}"
              f"  (n observed past t: {m['n_beyond']})"))


if __name__ == "__main__":
    import sim

    pl, ci, truth = sim.simulate()
    df = from_ats(pl, sim.SNAPSHOT)
    df["mgr_hist"] = manager_history(df)
    CATS = ["occupation", "emp_type", "source"]
    NUMS = ["tenure_norm_z", "relevance_z", "pay_uplift_pct", "screen_z", "mgr_hist"]

    # Check 1+2: fit on censored data; the curve must be monotone and recover the uncensored truth.
    perm = (df.emp_type == "perm").to_numpy()
    r = HazardModel(CATS, NUMS).fit(df).risk(df[perm])
    assert (np.diff(r, axis=1) >= -1e-12).all(), "risk must never fall as the horizon grows"
    tr, d = truth[perm], df[perm]
    print("P(bad exit by t), perm placements   truth | model | naive: drop censored | naive: censored = stayed")
    for t in HORIZONS:
        true_rate = (tr.true_cause.isin(BAD) & (tr.true_end_months <= t)).mean()
        y = d.cause.isin(BAD) & (d.duration <= t)
        known = (d.duration > t) | (d.cause != 0)
        print(f"  {t:>2}m   {true_rate:.3f} | {r[:, t - 1].mean():.3f} | {y[known].mean():.3f} | {y.mean():.3f}")
        assert abs(r[:, t - 1].mean() - true_rate) < 0.02, f"model misses the true {t}m rate"

    # Check 3: honest temporal validation (train on older placements, test on newer ones).
    cut = pd.Timestamp("2024-07-01")
    tr_df, te_df = df[df.start_date < cut], df[df.start_date >= cut]
    static = HazardModel(CATS, NUMS).fit(tr_df)
    report(f"Placement-time model, trained < {cut:%b %Y}, tested on {len(te_df)} later placements",
           te_df.duration, te_df.cause, static.risk(te_df))

    # Check 4: after the day-90 check-in, answers must sharpen the 12-month prediction.
    live = te_df[te_df.duration > 3]
    with_ci = HazardModel(CATS, NUMS, checkins=ci).fit(tr_df)
    r0, r1 = static.risk(live, from_month=4), with_ci.risk(live, from_month=4)
    a0 = evaluate(live.duration, live.cause, r0[:, 12 - 4], 12)["auc"]
    a1 = evaluate(live.duration, live.cause, r1[:, 12 - 4], 12)["auc"]
    print(f"\nStill there at month 4 -> AUC for a bad exit by 12m: placement data only {a0}, plus check-ins {a1}")
    assert a1 > a0, "check-ins should add signal"

    i = 0
    print(f"\nExample, placement {live.placement_id.iloc[i]} ({live.occupation.iloc[i]}, {live.emp_type.iloc[i]}):")
    print("  at placement           P(still there) " + " | ".join(
        f"{t}m {1 - static.risk(live.iloc[[i]])[0, t - 1]:.0%}" for t in HORIZONS))
    print("  after a day-90 check-in P(still there) " + " | ".join(f"{t}m {1 - r1[i, t - 4]:.0%}" for t in HORIZONS))
    print("\nAll checks passed.")
