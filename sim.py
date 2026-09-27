"""Mock agency placements with a known data-generating process.

Each month a placement stays, quits, is dismissed, or (temps only) converts to perm: a monthly
discrete-time competing-risks hazard (Beyersmann et al. 2009, in discrete time). A planned contract
end is administrative censoring. Calibration targets for perm placements (quit + dismissed):
~12% gone by 3 months, ~27% by 12, ~38% by 24. Sources in DATA_AND_CHECKINS.md.

Writes mock/data/{placements,checkins,truth}.csv. truth.csv holds the hidden and uncensored
values; the model must never see it. It exists only for the recovery checks in pipeline.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-09-30")
FIRST_START = pd.Timestamp("2021-10-01")
DAYS_PER_MONTH = 30.44
OCC = {"electrician": -0.25, "plumber": -0.2, "carpenter": 0.0, "plant_operator": 0.05,
       "truck_driver": 0.15, "labourer": 0.35, "site_admin": -0.1}  # quit log-odds offsets
OCC_P = [0.14, 0.08, 0.14, 0.1, 0.12, 0.32, 0.1]
SOURCES, SOURCE_P = ["job_board", "database", "referral", "headhunt"], [0.45, 0.3, 0.15, 0.1]

# True effects on monthly log-odds. Pre-hire effects sit in the ranges the research reports.
A_QUIT, G_QUIT = -3.2, -0.45     # baseline quit hazard, falling with tenure (log-month slope)
A_DISM, G_DISM = -4.6, -0.9      # dismissals concentrate in probation
A_CONV = -3.3                    # temp-to-perm conversion, from month 3
B_QUIT = dict(tenure_norm_z=np.log(0.89), relevance_z=np.log(0.92), referral=np.log(0.75),
              pay_uplift_pct=-0.02, screen_z=-0.05, temp=0.35, u=0.55, mgr=0.6)
B_DISM = dict(relevance_z=-0.3, screen_z=-0.45, u=0.4, mgr=0.3)
UNDERMINED_OR = 2.5              # supervisor undermining roughly triples quit hazard (Kammeyer-Mueller 2013)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _likert(z):
    """Latent score -> 1..5 answer."""
    return np.digitize(z, [-1.3, -0.45, 0.45, 1.3]) + 1


def simulate(n=5000, seed=7, snapshot=SNAPSHOT, quit_shift=0.0, undermined_or=UNDERMINED_OR, fit_effect=B_QUIT["u"],
             occ=OCC, occ_p=OCC_P, perm_share=0.45):
    """quit_shift moves the baseline quit log-odds; fit_effect is how strongly hidden fit drives quits;
    occ / occ_p set the occupation mix (quit log-odds offsets); perm_share is the share of permanent placements."""
    rng = np.random.default_rng(seed)
    n_mgr = 450
    mgr_client = rng.integers(0, 150, n_mgr)
    mgr_eff = rng.normal(0, 0.5, n_mgr)                       # hidden: how good each site manager is
    mgr = rng.integers(0, n_mgr, n)
    span = (snapshot - FIRST_START).days
    pl = pd.DataFrame(dict(
        placement_id=np.arange(1, n + 1),
        client_id=mgr_client[mgr],
        manager_id=mgr,
        occupation=rng.choice(list(occ), n, p=occ_p),
        emp_type=np.where(rng.random(n) < 1 - perm_share, "temp", "perm"),
        source=rng.choice(SOURCES, n, p=SOURCE_P),
        start_date=FIRST_START + pd.to_timedelta(np.sort(rng.integers(0, span, n)), "D"),
        tenure_norm_z=rng.normal(size=n).round(2),                # prior tenure vs occupation norm (from CV)
        relevance_z=rng.normal(size=n).round(2),                  # relevance of prior roles (from CV)
        pay_uplift_pct=rng.normal(5, 8, n).round(1),
        screen_z=rng.normal(size=n).round(2),                     # structured voice-screen score
    ))
    pl["referral"] = (pl.source == "referral").astype(int)
    temp = (pl.emp_type == "temp").to_numpy()
    u = rng.normal(size=n)                                        # hidden: how well the candidate fits
    m_eff = mgr_eff[mgr]
    undermined = rng.random(n) < _sig(-2.2 + 1.5 * m_eff)
    planned = np.where(temp, rng.choice([3, 6, 12, 24, np.inf], n, p=[.2, .3, .25, .1, .15]), np.inf)

    x = pl.assign(temp=temp.astype(int), u=u, mgr=m_eff)
    lin_q = pl.occupation.map(occ).to_numpy() + sum(b * x[k].to_numpy() for k, b in {**B_QUIT, "u": fit_effect}.items())
    lin_d = sum(b * x[k].to_numpy() for k, b in B_DISM.items())

    exit_m, cause = np.full(n, np.inf), np.zeros(n, int)
    alive = np.ones(n, bool)
    for m in range(1, 73):
        eq = A_QUIT + quit_shift + G_QUIT * np.log(m) + lin_q + np.log(undermined_or) * (undermined & (m >= 2))
        ed = A_DISM + G_DISM * np.log(m) + lin_d
        ec = np.where(temp & (m >= 3), A_CONV - 0.5 * u, -np.inf)
        ex = np.exp(np.stack([eq, ed, ec], 1))
        p = ex / (1 + ex.sum(1, keepdims=True))
        r = rng.random(n)[:, None]
        hit = (r < p.cumsum(1)).argmax(1) + 1
        hit[r[:, 0] >= p.sum(1)] = 0
        new = alive & (hit > 0) & (m <= planned)
        exit_m[new], cause[new] = m - 1 + rng.random(new.sum()), hit[new]
        alive &= ~new
    true_end = np.minimum(exit_m, planned)                        # months, continuous
    true_cause = np.where(exit_m <= planned, cause, 0)            # 0 = planned contract end / still there

    follow = ((snapshot - pl.start_date).dt.days / DAYS_PER_MONTH).to_numpy()
    observed = true_end <= follow
    pl["end_date"] = pd.to_datetime(np.where(observed, pl.start_date + pd.to_timedelta(
        np.where(observed, true_end, 0) * DAYS_PER_MONTH, "D"), pd.NaT)).floor("D")
    reason = pd.Series(np.select([true_cause == 1, true_cause == 2, true_cause == 3],
                                 ["quit", "dismissed", "converted"], "contract_end"))
    pl["exit_reason"] = reason.where(observed)
    pl["days_guaranteed"] = np.where(temp, 0, 90)

    rows = []
    for day in (30, 90):
        due = (true_end > day / DAYS_PER_MONTH) & (follow > day / DAYS_PER_MONTH)
        k = due.sum()
        resp = rng.random(k) < _sig(1.6 - 0.6 * u[due])          # people who are struggling answer less
        uu, mm, und = u[due], m_eff[due], undermined[due]
        rows.append(pd.DataFrame(dict(
            placement_id=pl.placement_id[due].to_numpy(), day=day, responded=resp.astype(int),
            quit_intent=_likert(0.9 * uu + 0.6 * mm + 0.8 * und + rng.normal(0, .7, k)),
            vitality=_likert(-0.7 * uu - 0.4 * mm + rng.normal(0, .8, k)),
            learning=_likert(-0.5 * uu + 0.3 * pl.relevance_z[due].to_numpy() + rng.normal(0, .8, k)),
            role_clarity=_likert(-0.5 * uu + 0.3 * pl.screen_z[due].to_numpy() + rng.normal(0, .8, k)),
            undermining=(und & (rng.random(k) < 0.8)).astype(int),
            client_rating=_likert(0.5 * pl.screen_z[due].to_numpy() - 0.4 * uu + rng.normal(0, .7, k)),
        )))
    ci = pd.concat(rows, ignore_index=True)
    answers = ["quit_intent", "vitality", "learning", "role_clarity", "undermining"]
    ci[answers] = ci[answers].astype(float).where(ci.responded == 1)
    ci["client_rating"] = ci.client_rating.astype(float).where(rng.random(len(ci)) < 0.7)  # client answers separately

    truth = pd.DataFrame(dict(placement_id=pl.placement_id, u=u.round(3), manager_effect=m_eff.round(3),
                              undermined=undermined.astype(int), true_end_months=true_end.round(3),
                              true_cause=true_cause))
    return pl, ci, truth


if __name__ == "__main__":
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    pl, ci, truth = simulate()
    for name, df in (("placements", pl), ("checkins", ci), ("truth", truth)):
        df.to_csv(out / f"{name}.csv", index=False)
    perm = truth[pl.emp_type == "perm"]
    bad = perm.true_cause.isin([1, 2])
    print({t: round(float((bad & (perm.true_end_months <= t)).mean()), 3) for t in (3, 6, 12, 18, 24)},
          "<- true P(bad exit by t), perm")
    print(pl.exit_reason.value_counts(dropna=False).to_dict())
