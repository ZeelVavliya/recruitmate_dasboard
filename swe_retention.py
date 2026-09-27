"""Retention of placed software engineers at 6/12/18/24 months: survival model comparison on mock data.

Mock data: 5,000 permanent software-engineering placements from sim.py, calibrated to ~20% gone (quit or
dismissed) by 12 months: LinkedIn put 2017 turnover in software at 13.2%, the highest of any industry, and
first-year attrition runs above the all-employee rate. The truth is known, so every model is scored against it.

`python swe_retention.py` writes data/swe_retention.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import pipeline as pp
import sim

SWE_OCC = {"back-end": 0.0, "full-stack": 0.05, "front-end": 0.1, "mobile": 0.1, "DevOps/SRE": -0.1,
           "data engineer": -0.05, "embedded": -0.2}
SWE_P = [0.26, 0.3, 0.14, 0.08, 0.1, 0.07, 0.05]
QUIT_SHIFT = -0.45
CATS, NUMS = ["occupation", "source"], ["tenure_norm_z", "relevance_z", "pay_uplift_pct", "screen_z", "mgr_hist"]
H = list(pp.HORIZONS)


def onehot(df, cols=None):
    X = pd.get_dummies(df[CATS].astype(str), drop_first=False, dtype=float).join(df[NUMS].astype(float))
    return X if cols is None else X.reindex(columns=cols, fill_value=0.0)


class Standardise:
    def __init__(self, X):
        self.cols, self.mu, self.sd = X.columns, X.mean(), X.std().replace(0, 1)

    def __call__(self, X):
        return ((X.reindex(columns=self.cols, fill_value=0.0) - self.mu) / self.sd).to_numpy()


def fit_cox(T, E, X, iters=40, ridge=1e-4):
    """Cox proportional hazards by Newton-Raphson (Breslow ties) + Breslow baseline hazard."""
    o = np.argsort(-T)
    T, E, X = T[o], E[o], X[o]
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        w = np.exp(X @ b)
        S0 = np.cumsum(w)
        S1 = np.cumsum(w[:, None] * X, axis=0)
        S2 = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
        m = S1[E] / S0[E, None]
        g = (X[E] - m).sum(0) - ridge * b
        Hm = (S2[E] / S0[E, None, None]).sum(0) - m.T @ m + ridge * np.eye(len(b))
        step = np.linalg.solve(Hm, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    S0 = np.cumsum(np.exp(X @ b))
    ev_t, inc = T[E][::-1], (1 / S0[E])[::-1]              # ascending event times
    H0 = lambda t: np.interp(t, ev_t, np.cumsum(inc), left=0.0)
    return lambda Xn, t: 1 - np.exp(-H0(t) * np.exp(Xn @ b))


def fit_weibull(T, E, X):
    """Weibull proportional hazards by maximum likelihood: S(t|x) = exp(-(t/lam)^k exp(xb))."""
    def nll(q):
        loglam, logk, b = q[0], q[1], q[2:]
        k, eta = np.exp(logk), X @ b
        z = (T / np.exp(loglam)) ** k * np.exp(eta)
        return -np.sum(E * (logk - loglam + (k - 1) * np.log(T / np.exp(loglam)) + eta) - z)
    r = optimize.minimize(nll, np.r_[np.log(T.mean() * 3), 0.0, np.zeros(X.shape[1])], method="L-BFGS-B")
    loglam, logk, b = r.x[0], r.x[1], r.x[2:]
    return lambda Xn, t: 1 - np.exp(-(t / np.exp(loglam)) ** np.exp(logk) * np.exp(Xn @ b))


def km_risk(T, E, t):
    """1 - Kaplan-Meier at t, other exits censored."""
    times = np.unique(T[E])
    s = 1.0
    for u in times[times <= t]:
        s *= 1 - ((T == u) & E).sum() / (T >= u).sum()
    return 1 - s


def main():
    pl, ci, truth = sim.simulate(n=5000, seed=11, occ=SWE_OCC, occ_p=SWE_P, perm_share=1.0, quit_shift=QUIT_SHIFT)
    df = pp.from_ats(pl, sim.SNAPSHOT)
    df["mgr_hist"] = pp.manager_history(df)
    cut = pd.Timestamp("2024-07-01")
    tr, te = df[df.start_date < cut], df[df.start_date >= cut]
    T, E = tr.duration.to_numpy(), tr.cause.isin(pp.BAD).to_numpy()
    std = Standardise(onehot(tr))
    Xtr, Xte = std(onehot(tr)), std(onehot(te))
    risk = {}

    risk["R1 Kaplan-Meier (no covariates)"] = np.tile([km_risk(T, E, t) for t in range(1, 25)], (len(te), 1))

    # R2: one "left by month t?" logistic classifier per horizon, dropping placements not yet observed to t
    r2 = np.zeros((len(te), 24))
    for t in H:
        known = (T > t) | (tr.cause.to_numpy() != 0) & (T <= t)
        y = (E & (T <= t))[known]
        r2[:, t - 1] = LogisticRegression(max_iter=3000).fit(Xtr[known], y).predict_proba(Xte)[:, 1]
    risk["R2 Separate classifier per horizon"] = r2

    cox = fit_cox(T, E, Xtr)
    risk["R3 Cox proportional hazards"] = np.column_stack([cox(Xte, t) for t in range(1, 25)])
    wb = fit_weibull(T, E, Xtr)
    risk["R4 Weibull"] = np.column_stack([wb(Xte, t) for t in range(1, 25)])
    risk["R5 Discrete-time logistic hazard"] = pp.HazardModel(CATS, NUMS).fit(tr).risk(te)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=1)
    risk["R6 Discrete-time gradient boosting"] = pp.HazardModel(CATS, NUMS, learner=gb).fit(tr).risk(te)

    out = {"n": len(df), "n_train": len(tr), "n_test": len(te), "n_train_exits": int(E.sum()), "cut": f"{cut:%B %Y}",
           "quit_shift": QUIT_SHIFT, "models": {}}
    tt = truth.set_index("placement_id").loc[te.placement_id]
    true_bad = tt.true_cause.isin(pp.BAD).to_numpy()
    for name, r in risk.items():
        m = {}
        for t in H:
            ev = pp.evaluate(te.duration, te.cause, r[:, t - 1], t)
            m[str(t)] = None if ev is None else {k: ev[k] for k in ("auc", "brier", "predicted", "observed", "n_beyond")}
            if m[str(t)]:
                m[str(t)]["truth"] = float((true_bad & (tt.true_end_months.to_numpy() <= t)).mean())
        hz = r[:, [t - 1 for t in H]]
        out["models"][name] = {"horizons": m, "non_monotone_share": float((np.diff(hz, axis=1) < -1e-9).any(axis=1).mean()),
                               "mean_curve": [0.0] + r.mean(0).round(4).tolist()}
    out["truth_curve"] = [0.0] + [float((true_bad & (tt.true_end_months.to_numpy() <= t)).mean()) for t in range(1, 25)]

    # R7: R5 plus the day-30 and day-90 check-ins, predicting forward from month 4
    live = te[te.duration > 3]
    base = pp.HazardModel(CATS, NUMS).fit(tr).risk(live, from_month=4)
    withci = pp.HazardModel(CATS, NUMS, checkins=ci).fit(tr).risk(live, from_month=4)
    out["checkin_lift"] = {str(t): [pp.evaluate(live.duration, live.cause, r[:, t - 4], t)["auc"] for r in (base, withci)]
                           for t in H if pp.evaluate(live.duration, live.cause, base[:, t - 4], t)}
    Path(pp.__file__).parent.joinpath("data", "swe_retention.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    out = main()
    print(f"train {out['n_train']}, test {out['n_test']}")
    for name, m in out["models"].items():
        row = " | ".join(f"{t}m AUC {v['auc']:.3f} Br {v['brier']:.3f} pred {v['predicted']:.3f} obs {v['observed']:.3f}"
                         if v else f"{t}m n/a" for t, v in m["horizons"].items())
        print(f"{name:<38} non-monotone {m['non_monotone_share']:.1%} | {row}")
    print("truth (test cohort):", {t: round(out['truth_curve'][t], 3) for t in H})
    print("check-in lift:", out["checkin_lift"])
