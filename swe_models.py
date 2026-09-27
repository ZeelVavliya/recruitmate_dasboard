"""Thriving and retention models for software engineers, on the Stack Overflow Developer Survey 2019.

Data: 52,695 respondents who code professionally, work full time and hold a software-engineering role.
Licence: Open Database License (ODbL) - commercial use allowed with attribution.

Thriving  = satisfied with the job (slightly or very) AND not actively looking for another one.
Retention = time with the current employer, estimated from current durations (Keiding et al. 2002):
            in a steady-state workforce the elapsed tenure of people in post has density S(t)/mean(T),
            so the survival curve of completed tenures is recoverable from a cross-section.

`python swe_models.py` fits everything and writes data/swe_results.json + data/swe_figure.png.
"""
import json
import urllib.request
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score

DATA = Path(__file__).parent / "data"
SO_URL = "https://media.githubusercontent.com/media/StackExchange/Survey/main/packages/archive/2019/results.csv"
SWE_ROLES = ["Developer, back-end", "Developer, front-end", "Developer, full-stack", "Developer, mobile",
             "Developer, desktop or enterprise applications", "Developer, embedded applications or devices",
             "DevOps specialist", "Engineer, site reliability"]
EUROPE = {"Germany", "France", "Netherlands", "Spain", "Italy", "Poland", "Sweden", "Switzerland", "Belgium", "Austria",
          "Denmark", "Norway", "Finland", "Portugal", "Czech Republic", "Romania", "Ukraine", "Russian Federation", "Greece",
          "Hungary", "Bulgaria", "Serbia", "Croatia", "Slovakia", "Slovenia", "Lithuania", "Latvia", "Estonia", "Belarus"}
# (predictor, reference level) - protected attributes (age, gender, dependents) are deliberately excluded
PREDICTORS = {"mgr": "not confident", "remote": "rarely", "org": "<20", "exp": "<2 yrs", "tenure": "<1 yr",
              "plan": "no spec", "review": "none", "tests": "in process", "hours": "<=40", "pay": "Q1 (lowest)",
              "competence": "average", "region": "AU/NZ", "role": "back-end"}
HORIZONS_YRS = [0.5, 1.0, 1.5, 2.0]


def load():
    path = DATA / "public" / "so2019.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SO_URL, path)
    cols = ["MainBranch", "Employment", "Country", "OrgSize", "DevType", "YearsCodePro", "JobSat", "MgrIdiot", "JobSeek",
            "LastHireDate", "ConvertedComp", "WorkWeekHrs", "WorkPlan", "WorkRemote", "ImpSyn", "CodeRev", "UnitTests",
            "Gender", "Age"]
    d = pd.read_csv(path, usecols=cols, low_memory=False)
    d = d[(d.MainBranch == "I am a developer by profession") & (d.Employment == "Employed full-time")]
    return d[d.DevType.fillna("").apply(lambda s: any(r in s for r in SWE_ROLES))].copy()


def prepare(d):
    m = lambda col, mapping: d[col].map(mapping)
    yrs = pd.to_numeric(d.YearsCodePro.replace({"Less than 1 year": 0.5, "More than 50 years": 51}), errors="coerce")
    hrs = d.WorkWeekHrs.where(d.WorkWeekHrs.between(10, 100))
    pay_rank = d.groupby("Country").ConvertedComp.rank(pct=True).where(d.groupby("Country").ConvertedComp.transform("count") >= 20)
    role = d.DevType.fillna("")
    x = pd.DataFrame({
        "mgr": m("MgrIdiot", {"Very confident": "very confident", "Somewhat confident": "somewhat confident",
                              "Not at all confident": "not confident", "I don't have a manager": "no manager"}).fillna("not answered"),
        "remote": m("WorkRemote", {"Less than once per month / Never": "rarely", "A few days each month": "few days a month",
                                   "Less than half the time, but at least one day each week": "weekly", "About half the time": "weekly",
                                   "More than half, but not all, the time": "mostly", "All or almost all the time (I'm full-time remote)": "mostly",
                                   "It's complicated": "complicated"}),
        "org": m("OrgSize", {"2-9 employees": "<20", "10 to 19 employees": "<20", "20 to 99 employees": "20-99",
                             "100 to 499 employees": "100-999", "500 to 999 employees": "100-999", "1,000 to 4,999 employees": "1,000-9,999",
                             "5,000 to 9,999 employees": "1,000-9,999", "10,000 or more employees": "10,000+"}),
        "exp": pd.cut(yrs, [0, 2, 5, 10, 99], right=False, labels=["<2 yrs", "2-4 yrs", "5-9 yrs", "10+ yrs"]).astype(object),
        "tenure": m("LastHireDate", {"Less than a year ago": "<1 yr", "1-2 years ago": "1-2 yrs", "3-4 years ago": "3-4 yrs",
                                     "More than 4 years ago": ">4 yrs"}),
        "plan": m("WorkPlan", {"There's no schedule or spec; I work on what seems most important or urgent": "no spec",
                               "There is a schedule and/or spec (made by me or by a colleague), and my work somewhat aligns": "spec, loosely",
                               "There is a schedule and/or spec (made by me or by a colleague), and I follow it very closely": "spec, closely"}),
        "review": m("CodeRev", {"Yes, because I see value in code review": "valued", "Yes, because I was told to do so": "mandated", "No": "none"}),
        "tests": m("UnitTests", {"Yes, it's part of our process": "in process", "Yes, it's not part of our process but the developers do it on their own": "devs choose",
                                 "No, but I think we should": "missing, wanted", "No, and I'm glad we don't": "missing, fine"}),
        "hours": pd.cut(hrs, [0, 40.5, 45.5, 50.5, 101], labels=["<=40", "41-45", "46-50", ">50"]).astype(object),
        "pay": pd.cut(pay_rank, [0, .25, .5, .75, 1.0001], labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]).astype(object),
        "competence": m("ImpSyn", {"Far below average": "below", "A little below average": "below", "Average": "average",
                                   "A little above average": "above", "Far above average": "far above"}),
        "region": np.select([d.Country.isin(["Australia", "New Zealand"]), d.Country.isin(["United States", "Canada"]),
                             d.Country.isin(["United Kingdom", "Ireland"]), d.Country.isin(EUROPE), d.Country == "India"],
                            ["AU/NZ", "US/Canada", "UK/Ireland", "Europe (other)", "India"], "Rest of world"),
        "role": np.select([role.str.contains("DevOps|site reliability"), role.str.contains("mobile"), role.str.contains("embedded"),
                           role.str.contains("full-stack"), role.str.contains("back-end"), role.str.contains("front-end")],
                          ["DevOps/SRE", "mobile", "embedded", "full-stack", "back-end", "front-end"], "desktop/enterprise"),
    }, index=d.index)
    x["hours"] = x.hours.fillna("not given")
    x["pay"] = x.pay.fillna("not given")
    x["jobsat"] = m("JobSat", {"Very dissatisfied": 1, "Slightly dissatisfied": 2, "Neither satisfied nor dissatisfied": 3,
                               "Slightly satisfied": 4, "Very satisfied": 5})
    x["thriving"] = ((x.jobsat >= 4) & (d.JobSeek != "I am actively looking for a job")).astype(float)
    x["gender"] = np.select([d.Gender == "Man", d.Gender == "Woman"], ["man", "woman"], "other/not given")
    x["age_band"] = pd.cut(d.Age, [0, 30, 40, 120], right=False, labels=["<30", "30-39", "40+"]).astype(object)
    return x.dropna(subset=list(PREDICTORS) + ["jobsat"])


# ---------------------------------------------------------------- design matrices
def dummies(x, var):
    levels = [l for l in sorted(x[var].unique()) if l != PREDICTORS[var]]
    return pd.DataFrame({f"{var}[{l}]": (x[var] == l).astype(float) for l in levels}, index=x.index)


def design(x, mains, pairs=()):
    blocks = {v: dummies(x, v) for v in mains}
    parts = [pd.Series(1.0, index=x.index, name="const")] + list(blocks.values())
    for a, b in pairs:
        A, B = blocks.get(a, dummies(x, a)), blocks.get(b, dummies(x, b))
        parts.append(pd.DataFrame({f"{ca}:{cb}": A[ca] * B[cb] for ca in A for cb in B}, index=x.index))
    X = pd.concat(parts, axis=1)
    return X.loc[:, X.std() > 0].assign(const=1.0)


class Fit:
    """Logistic regression by Newton-Raphson (statsmodels' Logit does not import with SciPy >= 1.16)."""

    def __init__(self, y, X, start=None, ridge=1e-6):
        A, y = X.to_numpy(float), np.asarray(y, float)
        b = np.zeros(A.shape[1]) if start is None else np.asarray(start, float).copy()
        for _ in range(60):
            p = special.expit(A @ b)
            H = A.T @ (A * (p * (1 - p))[:, None]) + ridge * np.eye(len(b))
            step = np.linalg.solve(H, A.T @ (y - p) - ridge * b)
            b += step
            if np.abs(step).max() < 1e-9:
                break
        p = np.clip(special.expit(A @ b), 1e-12, 1 - 1e-12)
        self.params = pd.Series(b, X.columns)
        self.llf = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        self.df_model = len(b) - 1
        self.aic = 2 * len(b) - 2 * self.llf
        self.bse = pd.Series(np.sqrt(np.diag(np.linalg.inv(H))), X.columns)
        self.pvalues = pd.Series(2 * stats.norm.sf(np.abs(b / self.bse)), X.columns)

    def conf_int(self):
        return pd.DataFrame({0: self.params - 1.96 * self.bse, 1: self.params + 1.96 * self.bse})

    def predict(self, X):
        return pd.Series(special.expit(X[self.params.index].to_numpy(float) @ self.params.to_numpy()), X.index)


def fit_logit(y, X, start=None):
    return Fit(y, X, start)


def fit_ordinal(y, X):
    """Proportional-odds logit: P(y <= j) = expit(theta_j - Xb). Returns (aic, predict function)."""
    A, y = X.to_numpy(float), np.asarray(y, int) - 1
    J = y.max() + 1

    def unpack(q):
        return np.cumsum(np.r_[q[0], np.exp(q[1:J - 1])]), q[J - 1:]

    def nll(q):
        th, b = unpack(q)
        cdf = np.c_[np.zeros(len(y)), special.expit(th[None, :] - (A @ b)[:, None]), np.ones(len(y))]
        return -np.sum(np.log(np.maximum(cdf[np.arange(len(y)), y + 1] - cdf[np.arange(len(y)), y], 1e-12)))

    q0 = np.r_[-2.0, np.zeros(J - 2), np.zeros(A.shape[1])]
    r = optimize.minimize(nll, q0, method="L-BFGS-B", options=dict(maxiter=2000))
    th, b = unpack(r.x)

    def predict_top(Xn, top=2):  # P(y in the top `top` categories)
        return 1 - special.expit(th[J - 1 - top] - Xn[X.columns].to_numpy(float) @ b)
    return 2 * len(r.x) + 2 * r.fun, predict_top


def forward_interactions(y, x, mains, candidates, max_steps=25):
    """stepAIC-style forward search over two-way interactions."""
    chosen, res = [], fit_logit(y, design(x, mains))
    for _ in range(max_steps):
        best = None
        for p in candidates:
            if p in chosen:
                continue
            X = design(x, mains, chosen + [p])
            start = res.params.reindex(X.columns).fillna(0).to_numpy()
            r = fit_logit(y, X, start)
            if best is None or r.aic < best[1].aic:
                best = (p, r)
        if best is None or best[1].aic > res.aic - 2:  # stop when the next term buys < 2 AIC points
            break
        chosen.append(best[0]); res = best[1]
    return chosen, res


def backward_mains(y, x, mains, alpha=0.05):
    """Drop main effects that fail a likelihood-ratio test, one at a time."""
    mains = list(mains)
    while True:
        full = fit_logit(y, design(x, mains))
        pvals = {}
        for v in mains:
            red = fit_logit(y, design(x, [m for m in mains if m != v]))
            pvals[v] = stats.chi2.sf(2 * (full.llf - red.llf), full.df_model - red.df_model)
        worst = max(pvals, key=pvals.get)
        if pvals[worst] < alpha:
            return mains, full, pvals
        mains.remove(worst)


# ---------------------------------------------------------------- current-duration survival
BINS = [(0, 1), (1, 3), (3, 5), (5, np.inf)]  # "<1 yr", "1-2 yrs", "3-4 yrs", ">4 yrs" read as whole years
FAMILIES = {  # survival functions S(t; a, b) on the positive half-line; a, b unconstrained via exp()
    "Exponential": (1, lambda t, p: np.exp(-t / np.exp(p[0]))),
    "Weibull": (2, lambda t, p: np.exp(-(t / np.exp(p[0])) ** np.exp(p[1]))),
    "Log-logistic": (2, lambda t, p: 1 / (1 + (t / np.exp(p[0])) ** np.exp(p[1]))),
    "Log-normal": (2, lambda t, p: stats.norm.sf((np.log(np.maximum(t, 1e-12)) - p[0]) / np.exp(p[1]))),
}


def current_duration_fit(counts, family, bins=BINS):
    """ML fit of completed-tenure survival S(t) to binned elapsed tenures: P(bin) = integral of S over bin / mean(T).
    Tenures are capped at 60 years so heavy-tailed families keep a finite mean."""
    k, S = FAMILIES[family]
    grid = np.linspace(0, 60, 24001)

    def probs(p):
        cum = np.concatenate([[0], integrate.cumulative_trapezoid(S(grid, p), grid)])
        area = [np.interp(min(b, 60), grid, cum) - np.interp(a, grid, cum) for a, b in bins]
        return np.array(area) / cum[-1], cum[-1]

    full = lambda q: np.r_[q, 0.0][:2]  # the exponential has one free parameter
    nll = lambda q: -np.sum(counts * np.log(np.maximum(probs(full(q))[0], 1e-12)))
    starts = [[0.5, 0.0], [1.0, -0.5], [1.5, 0.3], [0.0, 0.5]] if k == 2 else [[0.5], [1.0], [1.5]]
    best = min((optimize.minimize(nll, s0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000))
                for s0 in starts), key=lambda r: r.fun)
    p = full(best.x)
    fitted, mean_t = probs(p)
    surv = {str(h): float(S(np.array([h]), p)[0]) for h in HORIZONS_YRS}
    median = float(optimize.brentq(lambda t: S(np.array([t]), p)[0] - 0.5, 1e-4, 60))
    return dict(family=family, aic=float(2 * k + 2 * best.fun), params=p.tolist(), fitted=fitted.tolist(),
                observed=(counts / counts.sum()).tolist(), surv=surv, median_yrs=median, mean_yrs=float(mean_t),
                curve=[float(v) for v in S(np.linspace(0, 5, 61), p)])


def tenure_counts(x):
    order = ["<1 yr", "1-2 yrs", "3-4 yrs", ">4 yrs"]
    return x.tenure.value_counts().reindex(order).to_numpy(float)


# ---------------------------------------------------------------- main
def main():
    x = prepare(load())
    rng = np.random.default_rng(2402)
    test_mask = rng.random(len(x)) < 0.3
    tr, te = x[~test_mask], x[test_mask]
    y_tr, y_te = tr.thriving, te.thriving
    mains = list(PREDICTORS)
    out = {"n": len(x), "n_train": len(tr), "n_test": len(te), "thriving_rate": float(x.thriving.mean()),
           "n_aunz": int((x.region == "AU/NZ").sum())}
    out["summary"] = {v: x[v].value_counts().to_dict() for v in mains}
    out["jobsat_dist"] = x.jobsat.value_counts().sort_index().to_dict()

    def score(p):
        return dict(auc=float(roc_auc_score(y_te, p)), brier=float(brier_score_loss(y_te, p)))

    models = {}
    # M1: all main effects
    r1 = fit_logit(y_tr, design(tr, mains))
    models["M1"] = dict(desc="Logistic, all 13 main effects", k=int(r1.df_model + 1), aic=float(r1.aic),
                        **score(r1.predict(design(te, mains).reindex(columns=r1.params.index, fill_value=0))))
    # M2: + two-way interactions, forward stepwise AIC
    pairs, r2 = forward_interactions(y_tr, tr, mains, list(combinations(mains, 2)))
    X2 = design(te, mains, pairs).reindex(columns=r2.params.index, fill_value=0)
    models["M2"] = dict(desc=f"M1 + {len(pairs)} interactions (stepwise AIC)", k=int(r2.df_model + 1), aic=float(r2.aic),
                        pairs=[list(p) for p in pairs], **score(r2.predict(X2)))
    # M3: significant main effects only
    keep, r3, pvals = backward_mains(y_tr, tr, mains)
    models["M3"] = dict(desc=f"Significant main effects only ({len(keep)} of 13)", k=int(r3.df_model + 1), aic=float(r3.aic),
                        dropped=[v for v in mains if v not in keep], pvals={k_: float(v) for k_, v in pvals.items()},
                        **score(r3.predict(design(te, keep).reindex(columns=r3.params.index, fill_value=0))))
    # M4: M3 + the two interactions that bought the most AIC in M2, if both terms survived
    top = [p for p in pairs if p[0] in keep and p[1] in keep][:2]
    r4 = fit_logit(y_tr, design(tr, keep, top))
    models["M4"] = dict(desc="M3 + " + ", ".join(f"{a} x {b}" for a, b in top), k=int(r4.df_model + 1), aic=float(r4.aic),
                        pairs=[list(p) for p in top],
                        **score(r4.predict(design(te, keep, top).reindex(columns=r4.params.index, fill_value=0))))
    # M5: proportional-odds model on the 5-level satisfaction scale (uses more of the response)
    Xo_tr = design(tr, keep).drop(columns="const")
    aic5, top2 = fit_ordinal(tr.jobsat, Xo_tr)
    p_sat = top2(design(te, keep).reindex(columns=Xo_tr.columns, fill_value=0))
    models["M5"] = dict(desc="Proportional-odds (5-level satisfaction), M3 terms", k=int(Xo_tr.shape[1] + 4), aic=None,
                        aic_ordinal=float(aic5), note="scores P(satisfied), not thriving", **score(p_sat))
    # M6: gradient boosting, all predictors, no manual interactions
    Xg_tr, Xg_te = pd.get_dummies(tr[mains]), pd.get_dummies(te[mains]).reindex(columns=pd.get_dummies(tr[mains]).columns, fill_value=0)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=1).fit(Xg_tr, y_tr)
    models["M6"] = dict(desc="Gradient boosting, all 13 predictors", k=None, aic=None, **score(gb.predict_proba(Xg_te)[:, 1]))
    out["models"] = models

    # Final model: M3 refitted on all data, odds ratios with 95% CI
    final = fit_logit(x.thriving, design(x, keep))
    ci = final.conf_int()
    out["final"] = {c: dict(coef=float(final.params[c]), or_=float(np.exp(final.params[c])), lo=float(np.exp(ci.loc[c, 0])),
                            hi=float(np.exp(ci.loc[c, 1])), p=float(final.pvalues[c])) for c in final.params.index}
    # Calibration and fairness of M3 on the test set
    p3 = r3.predict(design(te, keep).reindex(columns=r3.params.index, fill_value=0))
    q = pd.qcut(p3, 10, labels=False, duplicates="drop")
    out["calibration"] = [(float(p3[q == g].mean()), float(y_te[q == g].mean())) for g in range(q.max() + 1)]
    out["subgroups"] = {f"{col}={g}": dict(n=int((te[col] == g).sum()), auc=float(roc_auc_score(y_te[te[col] == g], p3[te[col] == g])),
                                         predicted=float(p3[te[col] == g].mean()), observed=float(y_te[te[col] == g].mean()))
                        for col in ["gender", "age_band", "region"] for g in sorted(te[col].dropna().unique())
                        if (te[col] == g).sum() >= 100 and y_te[te[col] == g].nunique() == 2}

    # Retention: current-duration survival, four families, all SWE and by segment
    out["tenure_counts"] = tenure_counts(x).tolist()
    fits = [current_duration_fit(tenure_counts(x), f) for f in FAMILIES]
    out["survival"] = fits
    best = min(fits, key=lambda f: f["aic"])["family"]
    out["survival_best"] = best
    out["survival_segments"] = {f"{col}: {g}": dict(n=int((x[col] == g).sum()), **current_duration_fit(tenure_counts(x[x[col] == g]), best))
                                for col, groups in [("region", ["AU/NZ", "US/Canada", "UK/Ireland", "India"]),
                                                    ("org", ["<20", "100-999", "10,000+"]),
                                                    ("mgr", ["very confident", "not confident"])] for g in groups}
    alt = [(0, 1), (1, 3), (3, 4), (4, np.inf)]
    out["survival_sensitivity"] = current_duration_fit(tenure_counts(x), best, alt)["surv"]
    (DATA / "swe_results.json").write_text(json.dumps(out, indent=1, default=float))
    return out, (tr, te, y_te, p3, r3, keep)


if __name__ == "__main__":
    import time
    t0 = time.time()
    out, _ = main()
    for k, v in out["models"].items():
        print(k, v["desc"], "| k", v["k"], "| AIC", None if v["aic"] is None else round(v["aic"], 1),
              "| AUC", round(v["auc"], 4), "| Brier", round(v["brier"], 4))
    for f in out["survival"]:
        print(f["family"], round(f["aic"], 1), {h: round(s, 3) for h, s in f["surv"].items()}, "median", round(f["median_yrs"], 2))
    print(f"{time.time() - t0:.0f}s")
