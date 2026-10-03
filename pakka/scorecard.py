"""
RTO risk models for Valmo Pakka.

  Rules v0    : hand-set points (the first prototype). Baseline.
  Scorecard   : bins -> Weight of Evidence -> logistic regression -> 0-100 risk points. CHAMPION.
  GBM         : gradient-boosted trees on raw features. CHALLENGER, used only to
                measure how much accuracy the interpretable scorecard gives up.

WoE convention (as in the brief):  WoE = ln(%non-RTO in bin / %RTO in bin)
  -> a risky bin has negative WoE, so the logistic coefficients come out negative.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier

RTO_COST = 170.0
NO_HIST = "No history (first order)"

# ---------------------------------------------------------------- binning spec
# Each entry: label shown in the app, raw column, how to bin it.
# Edges are business-readable on purpose (ops can say them out loud).
FEATURES = {
    "prior_rto_rate":   dict(label="Customer's past RTO rate", col="prior_rto_rate", kind="num_hist",
                             edges=[-0.001, 0.0, 0.25, 0.5, 1.0], names=["0%", "1-25%", "26-50%", "Over 50%"]),
    "prior_cod_refusals": dict(label="Past COD refusals at the door", col="prior_cod_refusals", kind="count_hist",
                               edges=[-1, 0, 1, 99], names=["0", "1", "2 or more"]),
    "prior_orders":     dict(label="Past orders", col="prior_orders", kind="num",
                             edges=[-1, 0, 1, 4, 9, 9999], names=["0 (first order)", "1", "2-4", "5-9", "10+"]),
    "days_since_last":  dict(label="Days since last order", col="days_since_last_order", kind="num_hist",
                             edges=[-1, 7, 30, 90, 9999], names=["0-7", "8-30", "31-90", "Over 90"]),
    "pincode_rto_rate": dict(label="Pincode's past RTO rate", col="pincode_rto_rate", kind="num",
                             edges=[-1, 0.12, 0.15, 0.18, 0.21, 0.25, 1], names=["Under 12%", "12-15%", "15-18%", "18-21%", "21-25%", "Over 25%"]),
    "payment_mode":     dict(label="Payment mode", col="payment_mode", kind="cat"),
    "is_new_address":   dict(label="Returning customer at a new address", col="is_new_address", kind="flag"),
    "map_pin_dropped":  dict(label="Map pin dropped", col="map_pin_dropped", kind="flag"),
    "landmark_only":    dict(label="Landmark-only address", col="landmark_only_address", kind="flag"),
    "distance":         dict(label="Distance to hub", col="distance_to_hub_km", kind="num",
                             edges=[-1, 5, 10, 20, 9999], names=["Up to 5 km", "5-10 km", "10-20 km", "Over 20 km"]),
    "value_ratio":      dict(label="Order value vs pincode average", col="value_to_pincode_aov", kind="num",
                             edges=[-1, 1, 2, 3, 9999], names=["Up to 1x", "1-2x", "2-3x", "Over 3x"]),
    "category":         dict(label="Category", col="category", kind="cat"),
    "multi_sizes":      dict(label="Several sizes of one item in cart", col="multiple_sizes_in_cart", kind="flag"),
    "promised_days":    dict(label="Promised delivery time", col="promised_delivery_days", kind="num",
                             edges=[0, 3, 5, 7, 99], names=["2-3 days", "4-5 days", "6-7 days", "8+ days"]),
    "late_night":       dict(label="Ordered between midnight and 5 am", col="order_hour", kind="hour"),
    "festival":         dict(label="Festival sale period", col="festival_sale", kind="flag"),
    "pincode_tier":     dict(label="Pincode tier", col="pincode_tier", kind="cat"),
}


RETURNING_ONLY = {"prior_rto_rate", "prior_cod_refusals", "days_since_last", "is_new_address"}


def bin_column(df: pd.DataFrame, key: str) -> pd.Series:
    s = FEATURES[key]
    x = df[s["col"]]
    k = s["kind"]
    if k == "cat":
        return x.astype(str)
    if k == "flag":
        out = pd.Series(np.where(x.astype(int) == 1, "Yes", "No"), index=df.index)
        if key == "is_new_address":
            out = out.where(df["prior_orders"] > 0, "Yes (first order)")
        return out
    if k == "hour":
        return pd.Series(np.where(x < 5, "Yes", "No"), index=df.index)
    b = pd.cut(x, s["edges"], labels=s["names"]).astype(object)
    if k in ("num_hist", "count_hist"):
        first = df["prior_orders"] == 0
        b = b.where(~first, NO_HIST)
    return pd.Series(b, index=df.index).fillna(NO_HIST).astype(str)


def bin_frame(df: pd.DataFrame, keys=None) -> pd.DataFrame:
    keys = keys or list(FEATURES)
    return pd.DataFrame({k: bin_column(df, k) for k in keys}, index=df.index)


def bin_order(key: str) -> list[str]:
    s = FEATURES[key]
    if key == "is_new_address":
        return ["Yes (first order)", "No", "Yes"]
    if s["kind"] == "flag" or s["kind"] == "hour":
        return ["No", "Yes"]
    if "names" in s:
        return ([NO_HIST] if s["kind"] in ("num_hist", "count_hist") else []) + list(s["names"])
    return []


# ---------------------------------------------------------------- scorecard
@dataclass
class Scorecard:
    woe: dict = field(default_factory=dict)        # key -> DataFrame(bin, n, rto, rate, woe, iv)
    iv: dict = field(default_factory=dict)
    kept: list = field(default_factory=list)
    dropped: dict = field(default_factory=dict)    # key -> reason
    coef: dict = field(default_factory=dict)
    intercept: float = 0.0
    # risk points = A + B * logit(P);  0 points <-> P_LO, 100 points <-> P_HI
    P_LO: float = 0.03
    P_HI: float = 0.75
    base_rate: float = 0.0

    @property
    def B(self):
        return 100.0 / (np.log(self.P_HI / (1 - self.P_HI)) - np.log(self.P_LO / (1 - self.P_LO)))

    @property
    def A(self):
        return -self.B * np.log(self.P_LO / (1 - self.P_LO))

    # ---- fit
    def fit(self, df: pd.DataFrame, y: pd.Series, iv_min=0.02, corr_max=0.7, C=1.0):
        bins = bin_frame(df)
        y = y.values
        G, Bd = (y == 0).sum(), (y == 1).sum()
        self.base_rate = y.mean()
        for k in FEATURES:
            t = pd.DataFrame({"bin": bins[k].values, "y": y}).groupby("bin").y.agg(rto="sum", n="count")
            t["good"] = t.n - t.rto
            pg = (t.good + 0.5) / (G + 0.5 * len(t))
            pb = (t.rto + 0.5) / (Bd + 0.5 * len(t))
            t["woe"] = np.log(pg / pb)
            # History features describe RETURNING customers only. First orders get WoE 0 here,
            # because "is this a first order" is carried once, by the Past orders feature.
            if k in RETURNING_ONLY:
                for lab in (NO_HIST, "Yes (first order)"):
                    if lab in t.index:
                        t.loc[lab, "woe"] = 0.0
            t["iv"] = (pg - pb) * t.woe
            t["rate"] = t.rto / t.n
            order = bin_order(k)
            if order:
                t = t.reindex([o for o in order if o in t.index] + [i for i in t.index if i not in order])
            self.woe[k] = t.reset_index().rename(columns={"index": "bin"})
            self.iv[k] = float(t.iv.sum())

        cand = sorted([k for k in FEATURES if self.iv[k] >= iv_min], key=lambda k: -self.iv[k])
        for k in FEATURES:
            if self.iv[k] < iv_min:
                self.dropped[k] = f"IV {self.iv[k]:.3f} is below {iv_min}"
        W = self._woe_matrix(bins, cand)
        returning = (df["prior_orders"] > 0).values

        def corr(a, b):
            # History features are neutral on first orders, so compare them on returning customers only;
            # otherwise the first-order/returning split alone makes every pair look correlated.
            m = returning if (a in RETURNING_ONLY or b in RETURNING_ONLY) else slice(None)
            x, z = W[a].values[m], W[b].values[m]
            if x.std() == 0 or z.std() == 0:
                return 0.0
            return float(np.corrcoef(x, z)[0, 1])

        kept = []
        self.corr = {}
        for k in cand:
            clash = None
            for j in kept:
                r = corr(k, j)
                self.corr[(k, j)] = r
                if abs(r) > corr_max:
                    clash = (j, r)
                    break
            if clash:
                self.dropped[k] = f"moves with '{FEATURES[clash[0]]['label']}' (corr {clash[1]:.2f}); kept the stronger one"
            else:
                kept.append(k)
        # refit until every coefficient has the expected (negative) sign
        while True:
            lr = LogisticRegression(C=C, max_iter=2000).fit(W[kept].values, y)
            bad = [k for k, b in zip(kept, lr.coef_[0]) if b >= 0]
            if not bad:
                break
            for k in bad:
                kept.remove(k)
                self.dropped[k] = "sign flipped once other features were in (adds nothing on its own)"
        self.kept = kept
        self.coef = {k: float(b) for k, b in zip(kept, lr.coef_[0])}
        self.intercept = float(lr.intercept_[0])
        return self

    def _woe_matrix(self, bins: pd.DataFrame, keys) -> pd.DataFrame:
        cols = {}
        for k in keys:
            m = dict(zip(self.woe[k].bin, self.woe[k].woe))
            cols[k] = bins[k].map(m).fillna(0.0).astype(float)
        return pd.DataFrame(cols, index=bins.index)

    # ---- predict
    def logit(self, df: pd.DataFrame) -> np.ndarray:
        W = self._woe_matrix(bin_frame(df, self.kept), self.kept)
        return self.intercept + W.values @ np.array([self.coef[k] for k in self.kept])

    def predict_proba(self, df):
        return 1 / (1 + np.exp(-self.logit(df)))

    def points(self, df):
        return np.clip(self.A + self.B * self.logit(df), 0, 100)

    def base_points(self):
        return self.A + self.B * self.intercept

    def bin_points(self, key, bin_label):
        t = self.woe[key]
        w = float(t.loc[t.bin == bin_label, "woe"].iloc[0]) if (t.bin == bin_label).any() else 0.0
        return self.B * self.coef[key] * w

    def explain(self, row: pd.DataFrame) -> pd.DataFrame:
        bins = bin_frame(row, self.kept).iloc[0]
        rows = [{"feature": FEATURES[k]["label"], "key": k, "bin": bins[k], "points": self.bin_points(k, bins[k])} for k in self.kept]
        return pd.DataFrame(rows).sort_values("points", ascending=False)

    def points_table(self) -> pd.DataFrame:
        out = []
        for k in self.kept:
            t = self.woe[k]
            for _, r in t.iterrows():
                out.append({"feature": FEATURES[k]["label"], "bin": r.bin, "orders": int(r.n), "rto_rate": r.rate,
                            "woe": r.woe, "points": self.B * self.coef[k] * r.woe})
        return pd.DataFrame(out)

    def p_at(self, pts):
        lg = (pts - self.A) / self.B
        return 1 / (1 + np.exp(-lg))


# ---------------------------------------------------------------- rules v0
RULES_V0 = [
    ("Brand-new address", 3, lambda d: d.is_new_address == 1),
    ("No map pin", 2, lambda d: d.map_pin_dropped == 0),
    ("Landmark-only address", 3, lambda d: d.landmark_only_address == 1),
    ("Pincode in top RTO decile", 3, lambda d: d.pincode_rto_rate >= d.pincode_rto_rate.quantile(0.9)),
    ("2+ prior COD refusals", 4, lambda d: d.prior_cod_refusals >= 2),
    ("First-time buyer", 2, lambda d: d.prior_orders == 0),
    ("COD", 3, lambda d: d.payment_mode == "COD"),
    ("Size-variant item", 2, lambda d: d.size_variant_item == 1),
    ("Several sizes in cart", 2, lambda d: d.multiple_sizes_in_cart == 1),
    ("Value over 3x pincode AOV", 2, lambda d: d.value_to_pincode_aov > 3),
    ("Over 10 km from hub", 2, lambda d: d.distance_to_hub_km > 10),
]


def rules_score(df):
    s = np.zeros(len(df))
    for _, w, f in RULES_V0:
        s += w * f(df).values.astype(float)
    return s


# ---------------------------------------------------------------- challenger
GBM_NUM = ["prior_rto_rate", "prior_cod_refusals", "prior_orders", "days_since_last_order", "customer_tenure_days",
           "pincode_rto_rate", "is_new_address", "map_pin_dropped", "landmark_only_address", "distance_to_hub_km",
           "value_to_pincode_aov", "size_variant_item", "multiple_sizes_in_cart", "promised_delivery_days", "order_hour",
           "festival_sale", "order_value"]
GBM_CAT = ["payment_mode", "category", "pincode_tier"]


def gbm_matrix(df):
    X = df[GBM_NUM].astype(float).copy()
    for c in GBM_CAT:
        X[c] = df[c].astype("category").cat.codes
    return X


def fit_gbm(df, y, seed=0):
    X = gbm_matrix(df)
    cat_mask = [c in GBM_CAT for c in X.columns]
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                       l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                       categorical_features=cat_mask, random_state=seed)
    return m.fit(X, y)


# ---------------------------------------------------------------- evaluation
def evaluate(y, score, n_bins=10):
    y = np.asarray(y)
    score = np.asarray(score, float)
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(len(y)), -score))   # ties broken at random
    ys = y[order]
    n = len(y)
    cum_bad = np.cumsum(ys) / ys.sum()
    cum_good = np.cumsum(1 - ys) / (1 - ys).sum()
    auc = roc_auc_score(y, score)
    dec = []
    for i in range(n_bins):
        a, b = i * n // n_bins, (i + 1) * n // n_bins
        dec.append({"decile": i + 1, "orders": b - a, "rto_rate": ys[a:b].mean(), "share_of_rto": ys[a:b].sum() / ys.sum()})
    gains = pd.DataFrame({"coverage": np.arange(0, 101) / 100,
                          "captured": [0.0] + [cum_bad[max(0, k * n // 100 - 1)] for k in range(1, 101)]})
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(cum_bad - cum_good)), "base": y.mean(),
            "top10_rate": dec[0]["rto_rate"], "top10_capture": float(cum_bad[n // 10 - 1]),
            "lift10": dec[0]["rto_rate"] / y.mean(), "deciles": pd.DataFrame(dec), "gains": gains}


def calibration(y, p, n_bins=10):
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(predicted=("p", "mean"), actual=("y", "mean"), orders=("y", "size"))
    return t.reset_index(drop=True)


def psi(expected, actual, edges):
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# ---------------------------------------------------------------- one-call pipeline
SPLIT_DATE = "2026-07-01"


def train_all(df: pd.DataFrame):
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    tr, te = df[df.order_date < SPLIT_DATE], df[df.order_date >= SPLIT_DATE]
    sc = Scorecard().fit(tr, tr.rto)
    gbm = fit_gbm(tr, tr.rto)
    res = {"train": tr, "test": te, "scorecard": sc, "gbm": gbm}
    p_sc_te, p_sc_tr = sc.predict_proba(te), sc.predict_proba(tr)
    p_gbm_te = gbm.predict_proba(gbm_matrix(te))[:, 1]
    r_te = rules_score(te)
    res["eval"] = {
        "Rules v0": evaluate(te.rto, r_te),
        "Scorecard": evaluate(te.rto, p_sc_te),
        "Gradient boosting": evaluate(te.rto, p_gbm_te),
        "Truth (hidden)": evaluate(te.rto, te["_true_p_rto"]),
    }
    res["brier"] = {"Scorecard": brier_score_loss(te.rto, p_sc_te), "Gradient boosting": brier_score_loss(te.rto, p_gbm_te)}
    res["train_auc"] = {"Scorecard": roc_auc_score(tr.rto, p_sc_tr)}
    res["calibration"] = calibration(te.rto.values, p_sc_te)
    pts_tr, pts_te = sc.points(tr), sc.points(te)
    res["psi"] = psi(pts_tr, pts_te, np.linspace(0, 100, 11))
    res["points_test"] = pts_te
    res["p_test"] = p_sc_te
    return res
