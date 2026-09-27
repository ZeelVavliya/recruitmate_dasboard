"""Replay the pipeline on public turnover data, standing in for an agency's ATS export.

1. Babushkin "Employee Turnover": 1,129 real employees, tenure in months, hiring channel including
   recruitment agency ("KA"). Licence CC BY-NC-SA 4.0.
2. Huebner & Patalano "Human Resources Data Set" v13: 310 synthetic employees with hire and
   termination dates, exit reasons and recruitment source. Licence CC BY-NC-ND 4.0 (Kaggle page).
Both are non-commercial: fine for prototyping and discussion, never for training a product model.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.survfunc import SurvfuncRight

from pipeline import DPM, HazardModel, evaluate, report

DATA = Path(__file__).parent / "data" / "public"
URLS = {  # canonical pages: kaggle.com/datasets/davinwijaya/employee-turnover, github.com/RHuebner1972/HRDatasets
    "turnover.csv": "https://raw.githubusercontent.com/shaunmcgregor/Basic_Stats_in_Python_Student_Workbooks/main/data/turnover.csv",
    "HRDataset_v13.csv": "https://raw.githubusercontent.com/RHuebner1972/HRDatasets/master/HRDataset_v13.csv",
}


def fetch(name):
    path = DATA / name
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URLS[name], path)
    return path


def km(duration, event, t):
    sf = SurvfuncRight(np.asarray(duration), np.asarray(event))
    past = sf.surv_times <= t
    return float(sf.surv_prob[past][-1]) if past.any() else 1.0


BAB_CATS = ["traffic", "coach", "way", "greywage", "industry", "profession"]
BAB_NUMS = ["extraversion", "independ", "selfcontrol", "anxiety", "novator"]
CHANNELS = {"KA": "recruitment agency", "referal": "employee referral", "youjs": "applied on job site",
            "empjs": "employer found CV on job site", "friends": "knew employer", "rabrecNErab":
            "employer approached via contact", "recNErab": "non-employee recommendation", "advert": "advert"}
HUE_CATS, HUE_NUMS = ["source", "dept"], ["pay"]
HUE_LEAKY = ["engagement", "satisfaction", "perf", "days_late", "projects"]


def load_babushkin():
    b = pd.read_csv(fetch("turnover.csv"), encoding="ISO-8859-1")
    # gender, age and supervisor gender are deliberately left out: protected attributes.
    return b[BAB_CATS + BAB_NUMS].assign(duration=b.stag, cause=b.event.astype(int))


def load_huebner():
    h = pd.read_csv(fetch("HRDataset_v13.csv"))
    h = h[h.EmploymentStatus != "Future Start"].reset_index(drop=True)
    start = pd.to_datetime(h.DateofHire, format="%m/%d/%Y")
    end = pd.to_datetime(h.DateofTermination, format="mixed")  # mixes 2- and 4-digit years
    snapshot = max(start.max(), end.max())
    df = pd.DataFrame(dict(
        start_date=start,
        duration=((end.fillna(snapshot) - start).dt.days / DPM).clip(lower=0.01),
        cause=h.EmploymentStatus.map({"Voluntarily Terminated": 1, "Terminated for Cause": 2}).fillna(0).astype(int),
        source=h.RecruitmentSource, dept=h.Department.str.strip(), pay=h.PayRate,
        # Measured on the snapshot date, i.e. after the outcome: including them is leakage.
        engagement=h.EngagementSurvey, satisfaction=h.EmpSatisfaction, perf=h.PerfScoreID,
        days_late=h.DaysLateLast30, projects=h.SpecialProjectsCount))
    # Hires go back to 2006 but the first exit is in 2012: the file only holds people employed from
    # 2012 on, so early leavers from 2006-2011 are missing. Enter those placements at their 2012 tenure.
    window = end.min()
    df["entry"] = ((window - start).dt.days / DPM).clip(lower=0)
    return df, dict(first_hire=start.min(), last_hire=start.max(), window=window, snapshot=snapshot)


def babushkin():
    df = load_babushkin()
    print("=" * 88 + "\nBabushkin (real): Kaplan-Meier P(still there) by hiring channel")
    for ch, g in df.groupby("traffic"):
        print(f"  {CHANNELS[ch]:<32} n={len(g):>4}   " + "   ".join(f"{t}m {km(g.duration, g.cause, t):.2f}"
                                                                  for t in (6, 12, 24)))
    test = df.sample(frac=0.3, random_state=1)
    train = df.drop(test.index)
    r = HazardModel(BAB_CATS, BAB_NUMS).fit(train).risk(test)
    report("Babushkin holdout (random 30%: no start dates, so no temporal split)", test.duration, test.cause, r)


def huebner():
    df, w = load_huebner()
    print("=" * 88 + f"\nHuebner (synthetic, ATS-shaped): hires {w['first_hire']:%Y}-{w['last_hire']:%Y}, exits only "
          f"from {w['window']:%b %Y}; {(df.cause == 1).sum()} quits, {(df.cause == 2).sum()} dismissals, "
          f"{(df.cause == 0).sum()} still there at {w['snapshot']:%b %Y}")

    cut = pd.Timestamp("2013-01-01")
    train, test = df[df.start_date < cut], df[df.start_date >= cut]
    naive = HazardModel(HUE_CATS, HUE_NUMS).fit(train.drop(columns="entry"))
    clean = HazardModel(HUE_CATS, HUE_NUMS).fit(train)
    for name, m in (("ignoring delayed entry", naive), ("with delayed entry", clean)):
        report(f"Huebner, trained on hires before {cut:%Y} ({name}), tested on {len(test)} later hires",
               test.duration, test.cause, m.risk(test))

    leaky = HazardModel(HUE_CATS, HUE_NUMS + HUE_LEAKY).fit(train)
    a_clean = evaluate(test.duration, test.cause, clean.risk(test)[:, 11], 12)["auc"]
    a_leaky = evaluate(test.duration, test.cause, leaky.risk(test)[:, 11], 12)["auc"]
    print(f"\n  Leakage check, 12m AUC: hire-time fields {a_clean} vs + survey/performance fields {a_leaky}. "
          "Those fields are recorded at the snapshot, after the outcome, so they stay out whatever the AUC says.")


if __name__ == "__main__":
    babushkin()
    huebner()
