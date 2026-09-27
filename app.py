"""Placement Survival Lab: Streamlit version of the mock-data dashboard.

Unlike the static dashboard.html, this one re-simulates live from the sidebar assumptions.
Run locally:  streamlit run app.py        Deploy: see README.md
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pipeline as pp
import replay
import sim

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]  # validated, fixed order
INK, MUTED = "#0f1720", "#7d8591"
H = list(pp.HORIZONS)
NICE = {"perm": "Permanent", "temp": "Temp / labour hire", "job_board": "Job board", "database": "Agency database",
        "referral": "Referral", "headhunt": "Headhunt", "plant_operator": "Plant operator", "truck_driver": "Truck driver",
        "site_admin": "Site admin", "contract_end": "Contract ended", "converted": "Converted to perm"}
label = lambda k: NICE.get(k, k[:1].upper() + k[1:])

st.set_page_config(page_title="Placement Survival Lab", layout="wide")


# ---------------------------------------------------------------- computation (cached)
@st.cache_data(show_spinner="Simulating placements and fitting the models…", max_entries=8)
def run(n, seed, quit_shift, undermined_or, fit_effect):
    pl, ci, truth = sim.simulate(n=n, seed=seed, quit_shift=quit_shift, undermined_or=undermined_or, fit_effect=fit_effect)
    df = pp.from_ats(pl, sim.SNAPSHOT)
    df["mgr_hist"] = pp.manager_history(df)
    cats = ["occupation", "emp_type", "source"]
    nums = ["tenure_norm_z", "relevance_z", "pay_uplift_pct", "screen_z", "mgr_hist"]
    months = np.arange(1, pp.MAX_M + 1)

    # Recovery: model on censored data vs the uncensored truth (permanent placements)
    perm = (df.emp_type == "perm").to_numpy()
    tr, d = truth[perm], df[perm]
    bad_true = tr.true_cause.isin(pp.BAD).to_numpy()
    full = pp.HazardModel(cats, nums).fit(df)
    r_perm = full.risk(d)
    rec = {"Truth": [], "Survival model": r_perm.mean(0), "Drop recent placements": [], "Count recent as stayed": []}
    for t in months:
        y = (d.cause.isin(pp.BAD) & (d.duration <= t)).to_numpy()
        known = ((d.duration > t) | (d.cause != 0)).to_numpy()
        rec["Truth"].append((bad_true & (tr.true_end_months <= t)).mean())
        rec["Drop recent placements"].append(y[known].mean())
        rec["Count recent as stayed"].append(y.mean())
    rec = pd.DataFrame(rec, index=months)

    still = 1 - full.risk(df)
    segments = {dim: {lvl: (int((df[dim] == lvl).sum()), np.r_[1.0, still[(df[dim] == lvl).to_numpy()].mean(0)])
                      for lvl in sorted(df[dim].unique())} for dim in cats}

    # Temporal validation
    cut = pd.Timestamp("2024-07-01")
    train, test = df[df.start_date < cut], df[df.start_date >= cut]
    static = pp.HazardModel(cats, nums).fit(train)
    r_test = static.risk(test)
    val = {t: pp.evaluate(test.duration, test.cause, r_test[:, t - 1], t) for t in H}

    # Check-ins: raw signal and model lift from month 4
    c30 = ci[ci.day == 30].merge(df[["placement_id", "duration", "cause"]], on="placement_id")
    c30 = c30[(c30.duration > 6) | (c30.cause != 0)].copy()
    c30["gone"] = c30.cause.isin(pp.BAD) & (c30.duration <= 6)
    c30["answer"] = np.where(c30.responded == 1, c30.quit_intent.fillna(0).astype(int).astype(str), "No reply")
    by_answer = c30.groupby("answer").gone.agg(n="size", share="mean").reindex(["1", "2", "3", "4", "5", "No reply"])
    undermine = c30[c30.responded == 1].groupby("undermining").gone.mean()
    live = test[test.duration > 3]
    with_ci = pp.HazardModel(cats, nums, checkins=ci).fit(train)
    r0, r1 = static.risk(live, from_month=4), with_ci.risk(live, from_month=4)
    lift = {}
    for t in H:
        a, b = (pp.evaluate(live.duration, live.cause, r[:, t - 4], t) for r in (r0, r1))
        if a and b:
            lift[t] = (a["auc"], b["auc"])

    # Explorer: 150 placements still there at month 4
    idx = np.sort(np.random.default_rng(3).choice(len(live), min(150, len(live)), replace=False))
    pick = live.iloc[idx].reset_index(drop=True)
    at_start = 1 - static.risk(pick)
    after = 1 - r1[idx]
    return dict(meta=dict(n=n, n_perm=int(perm.sum()), n_temp=int((~perm).sum()), n_clients=pl.client_id.nunique(),
                          n_mgr=pl.manager_id.nunique(), n_train=len(train), n_test=len(test), cut=f"{cut:%b %Y}"),
                rec=rec, segments=segments, val=val, by_answer=by_answer, undermine=undermine, lift=lift,
                pick=pick, at_start=at_start, after=after, ci=ci[ci.placement_id.isin(pick.placement_id)])


@st.cache_data(show_spinner="Replaying public data…")
def public():
    bab = replay.load_babushkin()
    km = {replay.CHANNELS[ch]: (len(g), [replay.km(g.duration, g.cause, t) for t in range(25)]) for ch, g in bab.groupby("traffic")}
    test = bab.sample(frac=0.3, random_state=1)
    rb = pp.HazardModel(replay.BAB_CATS, replay.BAB_NUMS).fit(bab.drop(test.index)).risk(test)
    hold = {t: pp.evaluate(test.duration, test.cause, rb[:, t - 1], t) for t in H}

    hue, w = replay.load_huebner()
    cut = pd.Timestamp("2013-01-01")
    tr, te = hue[hue.start_date < cut], hue[hue.start_date >= cut]
    naive = pp.HazardModel(replay.HUE_CATS, replay.HUE_NUMS).fit(tr.drop(columns="entry")).risk(te)
    clean = pp.HazardModel(replay.HUE_CATS, replay.HUE_NUMS).fit(tr).risk(te)
    leaky = pp.HazardModel(replay.HUE_CATS, replay.HUE_NUMS + replay.HUE_LEAKY).fit(tr).risk(te)
    ev = lambda r, t: pp.evaluate(te.duration, te.cause, r[:, t - 1], t)
    hue_tab = pd.DataFrame({t: {"Ignoring it": ev(naive, t)["predicted"], "Corrected": ev(clean, t)["predicted"],
                                "Observed": ev(clean, t)["observed"]} for t in H}).T
    return dict(n_bab=len(bab), km=km, hold=hold, hue_tab=hue_tab, n_hue=len(hue), n_hue_test=len(te),
                first_hire=f"{w['first_hire']:%Y}", window=f"{w['window']:%b %Y}",
                leak=(ev(clean, 12)["auc"], ev(leaky, 12)["auc"]))


# ---------------------------------------------------------------- chart helpers
def curves(series, ytitle, yrange=(0, 1), height=380):
    """series: list of (name, x, y, color, dash). One y-axis, % ticks, unified hover."""
    fig = go.Figure()
    for name, x, y, color, dash in series:
        fig.add_scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2, dash=dash),
                        hovertemplate="%{y:.1%}")
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      xaxis=dict(title="Months since start", tickvals=[0, 3, 6, 12, 18, 24], range=[0, 24]),
                      yaxis=dict(title=ytitle, tickformat=".0%", range=list(yrange)))
    return fig


def as_text(df, formats):
    """Format columns as strings with a dash for missing values (Styler na_rep is not honoured by st.dataframe 1.43)."""
    out = df.copy().astype(object)
    for col, f in formats.items():
        out[col] = [("–" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)) for v in df[col]]
    return out


def metrics_table(m):
    rows = {f"{t} months": ({"AUC": f"{x['auc']:.2f}", "Brier": f"{x['brier']:.3f}", "Predicted": f"{x['predicted']:.1%}",
                             "Observed": f"{x['observed']:.1%}", "Observed that long": f"{x['n_beyond']:,}"}
                            if x else {"AUC": "not evaluable yet"}) for t, x in m.items()}
    return pd.DataFrame(rows).T.fillna("")


# ---------------------------------------------------------------- sidebar
with st.sidebar.form("assumptions"):
    st.subheader("Simulation assumptions")
    n = st.slider("Placements", 1000, 10000, 5000, step=500)
    quit_shift = st.slider("Baseline quit rate (shift)", -0.6, 0.6, 0.0, 0.1,
                           help="Moves the monthly quit log-odds. 0 is calibrated to about 12% gone by 3 months and 27% by 12.")
    undermined_or = st.slider("Supervisor undermining multiplies quit odds by", 1.0, 5.0, float(sim.UNDERMINED_OR), 0.5,
                              help="Research puts it near 3 (Kammeyer-Mueller et al. 2013).")
    fit_effect = st.slider("Weight of hidden candidate fit", 0.0, 1.2, float(sim.B_QUIT["u"]), 0.05,
                           help="How much fit that no one can see at placement drives quits. Higher: placement data explains "
                                "less and check-ins matter more.")
    seed = st.number_input("Random seed", 0, 9999, 7)
    st.form_submit_button("Re-simulate", type="primary", use_container_width=True)
st.sidebar.caption("Every other number on the page is estimated by the model from the simulated data.")

R = run(n, int(seed), quit_shift, undermined_or, fit_effect)
P = public()
m, rec = R["meta"], R["rec"]

# ---------------------------------------------------------------- header
st.title("Placement Survival Lab")

err = abs(rec.loc[24, "Survival model"] - rec.loc[24, "Truth"])
err_drop = abs(rec.loc[24, "Drop recent placements"] - rec.loc[24, "Truth"])
l12 = R["lift"].get(12)

k = st.columns(4)
k[0].metric("True share gone by 12 months", f"{rec.loc[12, 'Truth']:.1%}", help="Permanent placements, quit or dismissed")
k[1].metric("Error at 24 months: model vs dropping recent", f"{err * 100:.1f} vs {err_drop * 100:.1f} pts",
            help="Distance from the true 24-month rate, for the survival model and for the drop-recent-placements shortcut")
k[2].metric("12-month AUC, placement data", f"{l12[0]:.2f}" if l12 else "–")
k[3].metric("12-month AUC with check-ins", f"{l12[1]:.2f}" if l12 else "–",
            f"{l12[1] - l12[0]:+.2f}" if l12 else None)

tabs = st.tabs(["Recovery", "Segments", "Validation", "Check-ins", "Explorer", "Replay", "SWE: thriving", "SWE: retention"])

# ---------------------------------------------------------------- tabs
with tabs[0]:
    st.subheader("The survival model recovers the truth")
    st.write("Share of permanent placements that quit or were dismissed by each month. The model is fitted on the censored "
             "data an agency would hold at the snapshot. The dashed lines are the two shortcuts a plain classifier forces.")
    x = [0] + rec.index.tolist()
    st.plotly_chart(curves([(c, x, [0] + rec[c].tolist(), col, dash) for c, col, dash in
                            zip(rec.columns, [INK, S[0], S[1], S[2]], ["solid", "solid", "dash", "dash"])],
                           "P(quit or dismissed by month)", (0, max(0.5, rec.values.max() * 1.1))), use_container_width=True)
    st.dataframe(rec.loc[H].rename(index=lambda t: f"{t} months").style.format("{:.1%}"), use_container_width=True)

with tabs[1]:
    st.subheader("Who stays, by segment")
    dim = st.radio("Segment by", ["occupation", "source", "emp_type"], horizontal=True,
                   format_func={"occupation": "Occupation", "source": "Source", "emp_type": "Placement type"}.get)
    seg = R["segments"][dim]
    lo = min(c.min() for _, c in seg.values())
    st.plotly_chart(curves([(f"{label(k)} (n={n_:,})", list(range(25)), c, S[i], "solid")
                            for i, (k, (n_, c)) in enumerate(seg.items())], "P(still there)", (max(0, lo - 0.05), 1)),
                    use_container_width=True)
    st.caption("Model estimate: no quit or dismissal. Planned contract ends and temp-to-perm conversions are not bad exits.")

with tabs[2]:
    st.subheader("Tested on placements it has never seen")
    st.write(f"Trained on the {m['n_train']:,} placements that started before {m['cut']}, tested on the {m['n_test']:,} "
             "that started after. Newer placements have not been around long, so the 24-month check rests on few of them.")
    c1, c2 = st.columns(2)
    c1.dataframe(metrics_table(R["val"]), use_container_width=True)
    c1.caption("AUC ranks who leaves first (0.5 = chance). Brier scores the probability itself (lower is better). "
               "Both are weighted for censoring (IPCW).")
    h = c2.radio("Horizon", H, index=1, horizontal=True, format_func=lambda t: f"{t} months")
    v = R["val"][h]
    if v:
        p, o = zip(*v["calib_by_quintile"])
        hi = min(1.0, max(p + o) * 1.15)
        fig = go.Figure([go.Scatter(x=[0, hi], y=[0, hi], mode="lines", name="Perfect calibration",
                                    line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip"),
                         go.Scatter(x=p, y=o, mode="lines+markers", name=f"Model, {h} months", line=dict(color=S[0], width=2),
                                    marker=dict(size=9), hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<extra></extra>")])
        fig.update_layout(height=340, margin=dict(l=8, r=8, t=36, b=8), legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                          xaxis=dict(title="Predicted", tickformat=".0%", range=[0, hi]),
                          yaxis=dict(title="Observed", tickformat=".0%", range=[0, hi]))
        c2.plotly_chart(fig, use_container_width=True)
    else:
        c2.info("Not enough placements have been observed this long to judge this horizon yet.")

with tabs[3]:
    st.subheader("What the check-ins add")
    st.write('"How likely are you to be looking for a different job in the next 3 months?" asked at day 30, and what '
             "happened by month 6. Then the model, for placements still there at month 4, with and without the answers.")
    c1, c2 = st.columns(2)
    ba = R["by_answer"]
    fig = go.Figure(go.Bar(x=[{"1": "1 · very unlikely", "5": "5 · very likely"}.get(a, a) for a in ba.index], y=ba.share,
                           marker_color=[S[1] if a == "No reply" else S[0] for a in ba.index], customdata=ba.n,
                           hovertemplate="%{y:.1%} gone by 6 months<br>n=%{customdata:,}<extra></extra>"))
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=36, b=8), title=dict(text="Gone within 6 months, by day-30 answer",
                      font=dict(size=14)), yaxis=dict(tickformat=".0%"), bargap=0.35)
    c1.plotly_chart(fig, use_container_width=True)
    lift = R["lift"]
    fig = go.Figure([go.Bar(name=nm, x=[f"{t} months" for t in lift], y=[v[i] - 0.5 for v in lift.values()], base=0.5,
                            marker_color=S[i], customdata=[v[i] for v in lift.values()],
                            hovertemplate=nm + ": %{customdata:.3f}<extra></extra>")
                     for i, nm in enumerate(["Placement data only", "Plus check-in answers"])])
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=36, b=8), barmode="group", bargap=0.35,
                      title=dict(text="Ranking accuracy (AUC) from month 4", font=dict(size=14)),
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), yaxis=dict(title="AUC (0.50 = chance)", range=[0.5, 0.85]))
    c2.plotly_chart(fig, use_container_width=True)
    u = R["undermine"]
    s1, s2 = st.columns(2)
    s1.metric("Gone by 6 months after reporting supervisor undermining", f"{u.get(1.0, np.nan):.0%}",
              f"{u.get(0.0, np.nan):.0%} when not reported", delta_color="off")
    s2.metric("Gone by 6 months when the day-30 check-in went unanswered", f"{ba.loc['No reply', 'share']:.0%}")

with tabs[4]:
    st.subheader("What a recruiter would see")
    pick, ci = R["pick"], R["ci"].set_index(["placement_id", "day"])
    flagged = lambda pid: (pid, 90) in ci.index and ((ci.loc[(pid, 90), "quit_intent"] or 0) >= 4
                                                     or ci.loc[(pid, 90), "undermining"] == 1 or not ci.loc[(pid, 90), "responded"])
    only = st.toggle("Only placements flagged at day 90", value=True)
    rows = [i for i, pid in enumerate(pick.placement_id) if flagged(pid) or not only] or list(range(len(pick)))
    opts = {f"#{pick.placement_id[j]} · {label(pick.occupation[j])} · {label(pick.emp_type[j])}": j for j in rows}
    i = opts[st.selectbox("Placement", list(opts))]
    p = pick.iloc[i]
    c1, c2 = st.columns([2, 3])
    z = lambda v: f"{v:+.2f} SD"
    c1.dataframe(pd.Series({"Source": label(p.source), "Prior tenure vs norm": z(p.tenure_norm_z),
                            "Relevance of past roles": z(p.relevance_z), "Pay change": f"{p.pay_uplift_pct:+.1f}%",
                            "Voice-screen score": z(p.screen_z), "Manager's early-exit rate": f"{p.mgr_hist:.0%}"},
                           name="At placement"), use_container_width=True)
    items = {"Looking to leave (1–5)": "quit_intent", "Energy (1–5)": "vitality", "Learning (1–5)": "learning",
             "Knows what's expected (1–5)": "role_clarity", "Supervisor undermining": "undermining", "Client rating (1–5)": "client_rating"}

    def answer(day, col):
        if (p.placement_id, day) not in ci.index:
            return "–"
        r = ci.loc[(p.placement_id, day)]
        if col != "client_rating" and not r.responded:
            return "no reply"
        v = r[col]
        return "–" if pd.isna(v) else ("yes" if v else "no") if col == "undermining" else f"{v:.0f}"

    c1.dataframe(pd.DataFrame({f"Day {d}": {k: answer(d, c) for k, c in items.items()} for d in (30, 90)}), use_container_width=True)
    reason = p.exit_reason if isinstance(p.exit_reason, str) else "still there"
    fig = curves([("At placement", list(range(25)), np.r_[1.0, R["at_start"][i]], S[0], "solid"),
                  ("After day-90 check-in (given still there)", list(range(3, 25)), np.r_[1.0, R["after"][i]], S[1], "solid")],
                 "P(still there)", (0, 1), height=400)
    if p.duration <= 24:
        fig.add_vline(x=p.duration, line=dict(color=MUTED, dash="dot", width=1),
                      annotation_text=f"{label(reason)} · {p.duration:.1f} mo", annotation_position="top")
    c2.plotly_chart(fig, use_container_width=True)
    c2.caption(f"Outcome: **{label(reason)}** at {p.duration:.1f} months" + (" (snapshot)" if reason == "still there" else ""))

with tabs[5]:
    st.subheader("Replaying real history")
    st.write(f"Babushkin: {P['n_bab']:,} real employees with tenure and the channel they were hired through, including "
             f"recruitment agencies. Huebner: {P['n_hue']} synthetic employees shaped like an ATS export. Both are licensed "
             "for non-commercial use only.")
    order = list(replay.CHANNELS.values())
    chosen = st.multiselect("Hiring channels", order, default=order[:4], format_func=label)
    st.plotly_chart(curves([(f"{label(c)} (n={P['km'][c][0]})", list(range(25)), P["km"][c][1], S[order.index(c)], "solid")
                            for c in chosen], "P(still employed)", (0.5, 1)).update_traces(line_shape="hv"),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Babushkin holdout (random 30%)**")
    c1.dataframe(metrics_table(P["hold"]), use_container_width=True)
    c1.caption("No start dates in this dataset, so the split is random. Gender, age and supervisor gender are left out.")
    c2.markdown("**Huebner: the delayed-entry trap**")
    c2.write(f"Hires go back to {P['first_hire']}, but the first exit in the file is {P['window']}: early leavers from "
             f"before then are missing. Starting each placement's clock at {P['window']} instead of counting those years as survived:")
    c2.dataframe(P["hue_tab"].rename(index=lambda t: f"{t} months").style.format("{:.1%}"), use_container_width=True)
    c2.caption(f"Leakage check: engagement, satisfaction and performance fields move 12-month AUC from {P['leak'][0]:.2f} to "
               f"{P['leak'][1]:.2f}. They are recorded after the outcome, so they stay out. On {P['n_hue_test']} test hires "
               "this synthetic set carries no real signal.")

# ---------------------------------------------------------------- software engineers (precomputed by swe_models.py / swe_retention.py)
SWE_VARS = {"mgr": "Confidence in manager", "plan": "Planning", "tests": "Unit tests", "review": "Code review",
            "pay": "Pay (quartile in country)", "hours": "Hours per week", "remote": "Remote work", "org": "Organisation size",
            "exp": "Professional experience", "tenure": "Time in current job", "competence": "Self-rated skill",
            "region": "Region", "role": "Role"}
TH = json.loads((Path(__file__).parent / "data" / "swe_results.json").read_text())
RT = json.loads((Path(__file__).parent / "data" / "swe_retention.json").read_text())
F = TH["final"]

with tabs[6]:
    st.subheader("Software engineers: who thrives")
    st.write(f"Real data: {TH['n']:,} full-time professional software engineers from the 2019 Stack Overflow Developer Survey "
             f"({TH['n_aunz']:,} in Australia and New Zealand; Open Database License). Thriving means satisfied with the job and "
             f"not actively looking for another: {TH['thriving_rate']:.1%} of engineers. Age and gender are not used by the model.")
    mt = pd.DataFrame(TH["models"]).T[["desc", "k", "aic", "auc", "brier"]].astype({"k": float, "aic": float})
    mt.columns = ["Model", "Parameters", "AIC (train)", "AUC (test)", "Brier (test)"]
    st.dataframe(as_text(mt, {"Parameters": "{:.0f}", "AIC (train)": "{:,.0f}", "AUC (test)": "{:.3f}", "Brier (test)": "{:.4f}"}),
                 use_container_width=True)
    st.caption("M3 (the 13 main effects) is selected: interactions, an ordinal model and gradient boosting do no better on "
               "held-out data.")

    st.markdown("#### Thriving calculator")
    st.write("Describe an engineer's situation. The selected model gives the probability that they thrive. It starts on the "
             "most common answer for each question.")
    levels = {v: sorted(TH["summary"][v], key=lambda lv: -TH["summary"][v][lv]) for v in SWE_VARS}
    cols = st.columns(4)
    chosen = {v: cols[i % 4].selectbox(lab, levels[v], key=f"swe_{v}") for i, (v, lab) in enumerate(SWE_VARS.items())}
    effects = {v: F.get(f"{v}[{lv}]", {"coef": 0.0})["coef"] for v, lv in chosen.items()}
    p_thrive = 1 / (1 + np.exp(-(F["const"]["coef"] + sum(effects.values()))))
    c1, c2 = st.columns([1, 2])
    c1.metric("Probability of thriving", f"{p_thrive:.0%}", f"{(p_thrive - TH['thriving_rate']) * 100:+.0f} pts vs all engineers")
    moves = sorted(((abs(e), f"{SWE_VARS[v]}: {chosen[v]} (odds × {np.exp(e):.2f})") for v, e in effects.items() if e), reverse=True)
    c2.write("Biggest influences on this profile, against each question's reference answer: " +
             ("; ".join(m for _, m in moves[:3]) if moves else "none, every answer is the reference level") + ".")

    c1, c2 = st.columns([3, 2])
    terms = {k: v for k, v in F.items() if k != "const" and "not answered" not in k and "not given" not in k}
    order = sorted(terms, key=lambda k: terms[k]["or_"])
    fig = go.Figure(go.Scatter(
        x=[terms[k]["or_"] for k in order], y=[f"{SWE_VARS[k.split('[')[0]]}: {k.split('[')[1][:-1]}" for k in order],
        mode="markers", marker=dict(color=S[0], size=8),
        error_x=dict(type="data", symmetric=False, array=[terms[k]["hi"] - terms[k]["or_"] for k in order],
                     arrayminus=[terms[k]["or_"] - terms[k]["lo"] for k in order], thickness=1.5, width=0),
        customdata=[[terms[k]["lo"], terms[k]["hi"]] for k in order],
        hovertemplate="%{y}<br>odds ratio %{x:.2f} (95% CI %{customdata[0]:.2f}–%{customdata[1]:.2f})<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=MUTED, dash="dash", width=1))
    fig.update_layout(height=820, margin=dict(l=8, r=8, t=36, b=8), title=dict(text="Odds ratios in the selected model (M3)",
                      font=dict(size=14)), xaxis=dict(type="log", title="Odds ratio vs reference level (log scale)",
                                                      tickvals=[0.4, 0.7, 1, 1.5, 3, 8]))
    c1.plotly_chart(fig, use_container_width=True)
    p_cal, o_cal = zip(*TH["calibration"])
    fig = go.Figure([go.Scatter(x=[0.2, 0.95], y=[0.2, 0.95], mode="lines", name="Perfect calibration",
                                line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip"),
                     go.Scatter(x=p_cal, y=o_cal, mode="lines+markers", name="M3, by decile", line=dict(color=S[0], width=2),
                                hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<extra></extra>")])
    fig.update_layout(height=360, margin=dict(l=8, r=8, t=36, b=8), title=dict(text="Calibration on the test set", font=dict(size=14)),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
                      xaxis=dict(title="Predicted", tickformat=".0%"), yaxis=dict(title="Observed", tickformat=".0%"))
    c2.plotly_chart(fig, use_container_width=True)
    sub = pd.DataFrame(TH["subgroups"]).T[["n", "auc", "predicted", "observed"]].astype(float)
    sub.index = [i.replace("age_band=", "Age ").replace("gender=", "").replace("region=", "") for i in sub.index]
    c2.markdown("**Fairness and transfer checks (test set)**")
    c2.dataframe(sub.style.format({"n": "{:,.0f}", "auc": "{:.2f}", "predicted": "{:.1%}", "observed": "{:.1%}"}),
                 use_container_width=True)
    with st.expander("Why retention is not estimated from the survey"):
        st.write("The survey only records a band for time with the current employer (<1, 1–2, 3–4, >4 years). Four "
                 "current-duration models (Keiding et al. 2002) fitted to those bands disagree badly:")
        st.dataframe(pd.DataFrame({f["family"]: {**{f"S({float(h) * 12:.0f}m)": s for h, s in f["surv"].items()}, "AIC": f["aic"]}
                                   for f in TH["survival"]}).T.style.format("{:.1%}", subset=[c for c in ["S(6m)", "S(12m)", "S(18m)", "S(24m)"]])
                     .format("{:,.0f}", subset=["AIC"]), use_container_width=True)

with tabs[7]:
    st.subheader("Software engineers: retention at 6, 12, 18 and 24 months")
    st.write(f"Mock data: {RT['n']:,} permanent software-engineering placements, calibrated to about 20% quitting or dismissed "
             f"within 12 months (software had the highest turnover of any industry in LinkedIn's 2017 data, 13.2%). Trained on "
             f"placements that started before {RT['cut']} ({RT['n_train']:,}), tested on later ones ({RT['n_test']:,}).")
    rows = {name: {**{f"AUC {t}m": m["horizons"][t]["auc"] for t in ("6", "12", "18", "24")},
                   "12m predicted": m["horizons"]["12"]["predicted"], "12m observed": m["horizons"]["12"]["observed"],
                   "Brier 12m": m["horizons"]["12"]["brier"], "Contradictory": m["non_monotone_share"]}
            for name, m in RT["models"].items()}
    rows["R7 R5 + day-30/90 check-ins"] = {f"AUC {t}m": v[1] for t, v in RT["checkin_lift"].items()}
    st.dataframe(as_text(pd.DataFrame(rows).T, {**{f"AUC {t}m": "{:.3f}" for t in ("6", "12", "18", "24")},
                                                "12m predicted": "{:.1%}", "12m observed": "{:.1%}", "Brier 12m": "{:.3f}",
                                                "Contradictory": "{:.1%}"}), use_container_width=True)
    st.caption("Contradictory = share of placements given a higher risk at one horizon than at a later one. R7 is scored on "
               "placements still there at month 4, predicting forward.")
    c1, c2 = st.columns(2)
    shown = ["R3 Cox proportional hazards", "R4 Weibull", "R5 Discrete-time logistic hazard", "R6 Discrete-time gradient boosting"]
    c1.plotly_chart(curves([("Truth (test cohort)", list(range(25)), RT["truth_curve"], INK, "solid")] +
                           [(n_[3:], list(range(25)), RT["models"][n_]["mean_curve"], S[i], "dash" if i == 3 else "solid")
                            for i, n_ in enumerate(shown)], "P(quit or dismissed by month)", (0, 0.35)),
                    use_container_width=True)
    fig = go.Figure()
    for i, n_ in enumerate(["R2 Separate classifier per horizon"] + shown[:1] + shown[2:]):
        fig.add_scatter(x=[6, 12, 18, 24], y=[RT["models"][n_]["horizons"][t]["auc"] for t in ("6", "12", "18", "24")],
                        mode="lines+markers", name=n_[3:], line=dict(color=[S[4], S[0], S[2], S[3]][i], width=2))
    fig.add_scatter(x=[int(t) for t in RT["checkin_lift"]], y=[v[1] for v in RT["checkin_lift"].values()], mode="lines+markers",
                    name="Discrete-time hazard + check-ins", line=dict(color=S[6], width=2), marker=dict(symbol="square"))
    fig.add_hline(y=0.5, line=dict(color=MUTED, dash="dash", width=1))
    fig.update_layout(height=380, margin=dict(l=8, r=8, t=36, b=8), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      xaxis=dict(title="Horizon (months)", tickvals=[6, 12, 18, 24]), yaxis=dict(title="AUC (0.5 = chance)", range=[0.45, 0.75]))
    c2.plotly_chart(fig, use_container_width=True)
    st.info("Recommended: the discrete-time logistic hazard. It ties Cox on accuracy, takes check-in answers month by month and "
            "extends to separate exit types while keeping the four horizons consistent. The check-ins add more than any choice of model.",
            icon=":material/lightbulb:")
