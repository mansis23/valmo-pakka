"""
Valmo Pakka · RTO risk scorecard prototype
Team Prod Gods, IIT Kanpur · Meesho DICE S3

Run locally:   streamlit run app.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pakka import flow
from pakka.data_gen import ASSUMPTIONS, TARGETS, CATEGORIES, generate
from pakka.scorecard import FEATURES, NO_HIST, RULES_V0, Scorecard, bin_frame, train_all, SPLIT_DATE

st.set_page_config(page_title="Valmo Pakka · RTO scorecard", page_icon="📦", layout="wide")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "valmo_synthetic_orders.csv.gz"
ACCENT, RED, AMBER, GREEN, GREY = "#1F5F8B", "#C0392B", "#C27C0E", "#2F7D4A", "#8A929C"
MODEL_COLORS = {"Rules v0": AMBER, "Scorecard": ACCENT, "Gradient boosting": "#7B4FA0", "Truth (hidden)": GREY}
RTO_LOSS = 170


# ------------------------------------------------------------------ data + models
@st.cache_data(show_spinner="Generating 2 lakh synthetic orders (seed 42)…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA) if DATA.exists() else generate()
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@st.cache_resource(show_spinner="Fitting scorecard and challenger model…")
def load_models():
    return train_all(load_data())


df = load_data()
R = load_models()
SC: Scorecard = R["scorecard"]
TR, TE = R["train"], R["test"]
PTS_TR = SC.points(TR)
PTS_TE = R["points_test"]

# default tier cut-offs: top 10% of orders -> pre-confirmation, next 20% -> soft reminder (the deck's pilot design)
_recent = SC.points(TR[TR.order_date >= TR.order_date.max() - pd.Timedelta(days=60)])   # cut-offs set on the latest 2 months
DEF_HI = int(np.round(np.quantile(_recent, 0.90)))
DEF_LO = int(np.round(np.quantile(_recent, 0.70)))
st.session_state.setdefault("cut_hi", DEF_HI)
st.session_state.setdefault("cut_lo", DEF_LO)


def tier_of(points: float) -> str:
    return "A" if points >= st.session_state.cut_hi else "B" if points >= st.session_state.cut_lo else "C"


TIER_NAME = {"A": "Pre-confirmation", "B": "Soft reminder", "C": "No contact"}
TIER_COLOR = {"A": RED, "B": AMBER, "C": GREEN}


def badge(t):
    return f"<span style='background:{TIER_COLOR[t]};color:white;padding:4px 10px;border-radius:6px;font-weight:600'>{TIER_NAME[t]}</span>"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 📦 Valmo Pakka")
    st.caption("*Pehle Pakka, Phir Package.* Score every order, confirm the risky ones before pickup, "
               "nudge them to prepaid, and make cancelling cheap and early.")
    st.markdown("### Tier cut-offs")
    st.slider("Pre-confirmation from (points)", 0, 100, key="cut_hi")
    st.slider("Soft reminder from (points)", 0, 100, key="cut_lo")
    if st.session_state.cut_lo > st.session_state.cut_hi:
        st.warning("The soft-reminder cut-off should be below the pre-confirmation cut-off.")
    a = (PTS_TE >= st.session_state.cut_hi).mean()
    b = ((PTS_TE >= st.session_state.cut_lo) & (PTS_TE < st.session_state.cut_hi)).mean()
    st.markdown(f"- **Pre-confirmation:** {a:.0%} of orders · P(RTO) ≥ {SC.p_at(st.session_state.cut_hi):.0%}\n"
                f"- **Soft reminder:** {b:.0%} · P(RTO) ≥ {SC.p_at(st.session_state.cut_lo):.0%}\n"
                f"- **No contact:** {1 - a - b:.0%}")
    st.caption(f"Defaults ({DEF_HI} / {DEF_LO}) are set on the latest two training months so the riskiest ~10% get pre-confirmation "
               "and the next ~20% a reminder, as in the pilot plan. Shares above are on the held-out test months.")
    if st.button("Reset cut-offs"):
        st.session_state.cut_hi, st.session_state.cut_lo = DEF_HI, DEF_LO
        st.rerun()
    st.divider()
    st.caption("All orders here are **synthetic**, calibrated to the case-pack figures (17% RTO, 20% COD, 5% prepaid). "
               "No Valmo data is used. The *Real-data check* tab uses a public Amazon.in seller report.")

st.title("Valmo Pakka · RTO risk scorecard")
tabs = st.tabs(["① Score an order", "② How the scorecard works", "③ Validation & tiers", "④ Customer flow",
                "⑤ Real-data check", "⑥ The dataset"])

# ================================================================== TAB 1 · SCORE
PRESETS = {
    "Habitual refuser, COD": dict(prior_orders=6, prior_rto=4, prior_ref=3, days_since=12, new_addr=False, pin=False, landmark=False,
                                  tier="Tier-2", pin_rate=19, dist=8.0, prom=5, cod=True, cat="Apparel", value=899, aov=430, multi=True, hour=23, fest=False),
    "First-time rural COD, landmark address": dict(prior_orders=0, prior_rto=0, prior_ref=0, days_since=0, new_addr=True, pin=False, landmark=True,
                                                   tier="Tier-3", pin_rate=26, dist=22.0, prom=8, cod=True, cat="Apparel", value=449, aov=380, multi=False, hour=21, fest=False),
    "Loyal metro buyer, prepaid": dict(prior_orders=14, prior_rto=0, prior_ref=0, days_since=9, new_addr=False, pin=True, landmark=False,
                                       tier="Metro", pin_rate=11, dist=3.5, prom=3, cod=False, cat="Beauty", value=299, aov=520, multi=False, hour=14, fest=False),
    "Returning customer at a new address": dict(prior_orders=3, prior_rto=0, prior_ref=0, days_since=45, new_addr=True, pin=False, landmark=True,
                                                tier="Tier-2", pin_rate=17, dist=11.0, prom=5, cod=True, cat="Home & kitchen", value=549, aov=430, multi=False, hour=19, fest=True),
}


def apply_preset():
    p = PRESETS[st.session_state.preset]
    for k, v in p.items():
        st.session_state["in_" + k] = v


if "in_prior_orders" not in st.session_state:
    st.session_state.preset = "First-time rural COD, landmark address"
    apply_preset()

with tabs[0]:
    st.selectbox("Load an example order", list(PRESETS), key="preset", on_change=apply_preset)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Customer history** · known from past orders on this phone number")
            c1, c2 = st.columns(2)
            po = c1.number_input("Past orders", 0, 60, key="in_prior_orders")
            if po > 0:
                st.session_state.in_prior_rto = min(st.session_state.in_prior_rto, po)
                pr = c2.number_input("Past RTOs", 0, int(po), key="in_prior_rto")
                st.session_state.in_prior_ref = min(st.session_state.in_prior_ref, pr)
                c3, c4 = st.columns(2)
                prf = c3.number_input("Past COD refusals at the door", 0, int(pr), key="in_prior_ref")
                ds = c4.number_input("Days since last order", 0, 365, key="in_days_since")
                na = st.checkbox("Ordering to an address they haven't used before", key="in_new_addr")
            else:
                c2.caption("First order: no history yet. The scorecard treats this as its own risk group.")
                pr, prf, ds, na = 0, 0, np.nan, True
        with st.container(border=True):
            st.markdown("**Address & location**")
            c1, c2 = st.columns(2)
            mp = c1.checkbox("Map pin dropped", key="in_pin")
            lm = c2.checkbox("Landmark-only address", key="in_landmark", help="No house or flat number, e.g. 'Mandir ke paas'")
            c1, c2 = st.columns(2)
            tier = c1.selectbox("Pincode tier", ["Metro", "Tier-2", "Tier-3"], key="in_tier")
            pinr = c2.slider("Pincode's past RTO rate (%)", 5, 40, key="in_pin_rate")
            c1, c2 = st.columns(2)
            dist = c1.slider("Distance to hub (km)", 0.5, 60.0, step=0.5, key="in_dist")
            prom = c2.slider("Promised delivery (days)", 2, 10, key="in_prom")
        with st.container(border=True):
            st.markdown("**This order**")
            c1, c2 = st.columns(2)
            cod = c1.toggle("Cash on delivery", key="in_cod")
            cat = c2.selectbox("Category", CATEGORIES, key="in_cat")
            c1, c2 = st.columns(2)
            val = c1.number_input("Order value (₹)", 50, 20000, step=50, key="in_value")
            aov = c2.number_input("Pincode average order (₹)", 100, 3000, step=10, key="in_aov")
            c1, c2, c3 = st.columns(3)
            multi = c1.checkbox("Several sizes in cart", key="in_multi")
            hour = c2.number_input("Order hour (0-23)", 0, 23, key="in_hour")
            fest = c3.checkbox("Festival sale", key="in_fest")

    row = pd.DataFrame([{
        "prior_orders": po, "prior_rto": pr if po else 0, "prior_cod_refusals": prf if po else 0,
        "prior_rto_rate": (pr / po) if po else np.nan, "days_since_last_order": ds if po else np.nan,
        "pincode_rto_rate": pinr / 100, "payment_mode": "COD" if cod else "Prepaid",
        "is_new_address": int(na), "map_pin_dropped": int(mp), "landmark_only_address": int(lm),
        "distance_to_hub_km": dist, "value_to_pincode_aov": val / aov, "category": cat,
        "multiple_sizes_in_cart": int(multi), "promised_delivery_days": prom, "order_hour": hour,
        "festival_sale": int(fest), "pincode_tier": tier, "size_variant_item": int(cat in ("Apparel", "Footwear")),
    }])
    p = float(SC.predict_proba(row)[0])
    pts = float(SC.points(row)[0])
    t = tier_of(pts)
    alt = row.copy(); alt["payment_mode"] = "Prepaid" if cod else "COD"
    p_alt = float(SC.predict_proba(alt)[0])
    p_cod, p_pre = (p, p_alt) if cod else (p_alt, p)
    cap = max(0.0, (p_cod - p_pre) * RTO_LOSS)
    cashback = 15 if cap >= 15 else 10 if cap >= 10 else 0
    token = (30 if val < 400 else 50) if t == "A" else 0
    pct = (PTS_TE < pts).mean()

    with right:
        with st.container(border=True):
            m1, m2, m3 = st.columns([1, 1, 1.3])
            m1.metric("Risk score", f"{pts:.0f} / 100")
            m2.metric("P(RTO)", f"{p:.1%}", f"{(p - SC.base_rate) * 100:+.0f} pts vs avg", delta_color="inverse")
            m3.markdown(f"<div style='margin-top:6px'>{badge(t)}</div><div style='font-size:13px;opacity:.75;margin-top:8px'>Riskier than {pct:.0%} of orders</div>",
                        unsafe_allow_html=True)
            if t == "A":
                st.markdown(f"""**What happens:** WhatsApp within 15 minutes, before the seller packs: *Confirm / Fix address / Cancel*.
On confirm, {'offer ₹%d back for paying by UPI, or a ₹%d token now with the rest on delivery' % (cashback, token) if cod and cashback else 'offer a ₹%d UPI token' % token if cod else 'no payment nudge (already prepaid)'}.
No reply in 6 hours → IVR call. Still nothing → ship flagged *Unconfirmed*, and the rider calls first.""")
            elif t == "B":
                st.markdown(f"**What happens:** one WhatsApp reminder with a one-tap cancel link"
                            f"{' and ₹%d back for paying by UPI' % cashback if cod and cashback else ''}. No reply needed; it ships as normal.")
            else:
                st.markdown("**What happens:** nothing. It ships as normal. Messaging low-risk orders adds friction without moving RTO.")

        expl = SC.explain(row)
        neutral = expl[expl.points.abs() < 0.05]
        expl = expl[expl.points.abs() >= 0.05]
        base = SC.base_points()
        cum = base + np.concatenate([[0], np.cumsum(expl.points.values)])
        fig = go.Figure(go.Waterfall(
            orientation="h", measure=["absolute"] + ["relative"] * len(expl) + ["total"],
            y=["Starting points"] + [f"{r.feature}: {r.bin}" for r in expl.itertuples()] + ["Risk score"],
            x=[base] + list(expl.points) + [0],
            text=[f"{base:.0f}"] + [f"{v:+.1f}" for v in expl.points] + [f"{pts:.0f}"], textposition="outside",
            increasing=dict(marker_color=RED), decreasing=dict(marker_color=GREEN), totals=dict(marker_color=ACCENT),
            connector=dict(line=dict(color="rgba(128,128,128,.4)")),
        ))
        fig.update_layout(title="Why this score: points from each feature", height=110 + 34 * (len(expl) + 2),
                          yaxis=dict(autorange="reversed"), margin=dict(l=10, r=40, t=40, b=10), showlegend=False,
                          xaxis=dict(title="Risk points (red adds risk, green removes it)", range=[max(-5, cum.min() - 12), min(115, cum.max() + 12)]))
        st.plotly_chart(fig, width="stretch")
        if len(neutral):
            st.caption("No effect for this order: " + ", ".join(neutral.feature) + " (first order, so history features are neutral).")

        with st.container(border=True):
            st.markdown("**Prepaid nudge: how much can Valmo afford?**")
            c1, c2, c3 = st.columns(3)
            c1.metric("P(RTO) as COD", f"{p_cod:.1%}")
            c2.metric("P(RTO) if prepaid", f"{p_pre:.1%}")
            c3.metric("Break-even incentive", f"₹{cap:.0f}")
            st.caption(f"Switching this order to prepaid lowers its RTO chance by {(p_cod - p_pre) * 100:.1f} points, worth ₹{cap:.0f} "
                       f"at ₹170 per RTO. That's the most a cashback can cost before Valmo loses money. Riskier orders justify bigger "
                       f"nudges; a flat ₹50-100 discount doesn't pay. Caveat: customers who accept are probably safer than average.")

    st.session_state.order_ctx = dict(
        id="#MSH-48213", name="Priya", item=("Cotton kurta, size M" + (" + L" if multi else "")) if cat == "Apparel" else cat,
        value=int(val), cod=bool(cod), addr=("Hanuman Mandir ke paas, Kalyanpur, Kanpur" if lm else "H.No. 117/42, Gali 3, Kalyanpur, Kanpur"),
        p=p, p_prepaid=p_pre, points=pts, tier=t, tier_label=TIER_NAME[t], cashback=cashback if cod else 0, token=token if cod else 0)

# ================================================================== TAB 2 · METHOD
with tabs[1]:
    st.markdown("Rules → **statistical scorecard** → ML. The scorecard is the middle step: the data sets how much each "
                "factor matters, but every point can still be read out to an ops manager.")

    st.subheader("Step 1 · Historical orders with a known outcome")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training orders", f"{len(TR):,}", "Oct 2025 – Jun 2026", delta_color="off")
    c2.metric("Test orders (held out)", f"{len(TE):,}", "Jul – Sep 2026", delta_color="off")
    c3.metric("RTO rate (train)", f"{TR.rto.mean():.1%}")
    c4.metric("First orders (no history)", f"{(TR.prior_orders == 0).mean():.0%}")
    st.caption("Target: **rto = 1** if the parcel came back for any reason. Every history feature is point-in-time: an order only sees "
               "orders placed before it, so nothing from the future leaks in. We train on the first 9 months and test on the last 3, "
               "the way a model is checked before it goes live.")

    st.subheader("Steps 2-3 · Bin every feature and measure its Weight of Evidence")
    st.latex(r"WoE_i=\ln\left(\frac{\%\ \text{non-RTO orders in bin } i}{\%\ \text{RTO orders in bin } i}\right)\qquad "
             r"IV=\sum_i(\%\text{non-RTO}_i-\%\text{RTO}_i)\cdot WoE_i")
    keys = list(FEATURES)
    pick = st.selectbox("Feature", keys, format_func=lambda k: f"{FEATURES[k]['label']}  (IV {SC.iv[k]:.3f})",
                        index=keys.index("prior_cod_refusals"))
    t = SC.woe[pick].copy()
    t["points"] = [SC.B * SC.coef[pick] * w for w in t.woe] if pick in SC.kept else np.nan
    c1, c2 = st.columns([1.1, 1])
    with c1:
        fig = go.Figure(go.Bar(x=t.bin, y=t.rate, marker_color=[RED if w < 0 else GREEN for w in t.woe],
                               text=[f"{v:.1%}" for v in t.rate], textposition="outside",
                               hovertemplate="%{x}<br>RTO rate %{y:.1%}<extra></extra>"))
        fig.add_hline(y=SC.base_rate, line_dash="dash", line_color=GREY, annotation_text=f"average {SC.base_rate:.1%}")
        fig.update_layout(title=f"RTO rate by bin · {FEATURES[pick]['label']}", yaxis_tickformat=".0%", height=340,
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.dataframe(t[["bin", "n", "rto", "rate", "woe", "iv", "points"]].rename(columns={
            "bin": "Bin", "n": "Orders", "rto": "RTOs", "rate": "RTO rate", "woe": "WoE", "iv": "IV part", "points": "Points"}),
            hide_index=True, width="stretch",
            column_config={"RTO rate": st.column_config.NumberColumn(format="percent"),
                           "WoE": st.column_config.NumberColumn(format="%.3f"), "IV part": st.column_config.NumberColumn(format="%.4f"),
                           "Points": st.column_config.NumberColumn(format="%+.1f")})
        if pick in ("prior_rto_rate", "prior_cod_refusals", "days_since_last", "is_new_address"):
            st.caption("First orders get WoE 0 here on purpose. 'Is this a first order' is counted once, in *Past orders*, "
                       "so this feature only separates returning customers.")
        if pick not in SC.kept:
            st.warning(f"Not in the final model: {SC.dropped.get(pick, '')}.")

    st.subheader("Feature selection")
    iv = pd.DataFrame([{"Feature": FEATURES[k]["label"], "IV": SC.iv[k],
                        "Status": "In model" if k in SC.kept else "Dropped",
                        "Why": "" if k in SC.kept else SC.dropped.get(k, "")} for k in FEATURES]).sort_values("IV")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        fig = go.Figure(go.Bar(y=iv.Feature, x=iv.IV, orientation="h",
                               marker_color=[ACCENT if s == "In model" else GREY for s in iv.Status],
                               text=[f"{v:.3f}" for v in iv.IV], textposition="outside"))
        fig.add_vline(x=0.02, line_dash="dash", line_color=RED, annotation_text="0.02 cut-off", annotation_position="bottom right")
        fig.update_layout(title="Information Value (blue = kept)", height=520, margin=dict(l=10, r=30, t=40, b=10), xaxis_title="IV")
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("Three filters, in order:\n1. **IV ≥ 0.02.** Below that, a feature barely separates RTO from delivered.\n"
                    "2. **No near-duplicates.** If two features move together (correlation > 0.7), keep the one with the higher IV.\n"
                    "3. **Sensible sign.** Every coefficient must point the same way as its WoE. A feature that flips once the others are in adds nothing.")
        st.dataframe(iv.sort_values("IV", ascending=False), hide_index=True, width="stretch",
                     column_config={"IV": st.column_config.NumberColumn(format="%.3f")})

    st.subheader("Step 4 · Logistic regression on the WoE values")
    terms = " ".join(f"{'-' if SC.coef[k] < 0 else '+'} {abs(SC.coef[k]):.3f}\\,WoE_{{\\text{{{FEATURES[k]['label'].split()[0] + ('' if len(FEATURES[k]['label'].split()) == 1 else ' ' + FEATURES[k]['label'].split()[1])}}}}}" for k in SC.kept)
    st.latex(r"\text{logit}\,P(RTO) = " + f"{SC.intercept:.3f}" + r"\ " + terms)
    st.caption("Coefficients are negative because WoE is ln(non-RTO ÷ RTO): a risky bin has negative WoE, which pushes P(RTO) up.")

    st.subheader("Step 5 · Turn the probability into 0-100 risk points")
    st.latex(r"\text{Risk points} = A + B\cdot\text{logit}\,P \qquad 0\text{ points}=P\ %d\%%,\quad 100\text{ points}=P\ %d\%%" % (SC.P_LO * 100, SC.P_HI * 100))
    st.markdown(f"Because the scale is linear in log-odds, **each feature's points simply add up**: start at {SC.base_points():.0f}, then add or subtract the points for "
                "each bin the order falls in. That's what the waterfall on tab ① shows. The table below *is* the scorecard: printable, explainable, and "
                "editable by ops without retraining.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        ptab = SC.points_table()
        st.dataframe(ptab.rename(columns={"feature": "Feature", "bin": "Bin", "orders": "Orders", "rto_rate": "RTO rate", "woe": "WoE", "points": "Points"}),
                     hide_index=True, width="stretch", height=520,
                     column_config={"RTO rate": st.column_config.NumberColumn(format="percent"),
                                    "WoE": st.column_config.NumberColumn(format="%.3f"),
                                    "Points": st.column_config.NumberColumn(format="%+.1f")})
        st.download_button("Download the scorecard (CSV)", ptab.to_csv(index=False).encode(), "valmo_pakka_scorecard.csv", "text/csv")
    with c2:
        xs = np.arange(0, 101)
        fig = go.Figure()
        for lo, hi, colr, name in ((0, st.session_state.cut_lo, GREEN, "No contact"), (st.session_state.cut_lo, st.session_state.cut_hi, AMBER, "Soft reminder"),
                                   (st.session_state.cut_hi, 100, RED, "Pre-confirmation")):
            fig.add_vrect(x0=lo, x1=hi, fillcolor=colr, opacity=0.12, line_width=0, annotation_text=name, annotation_position="top left")
        fig.add_trace(go.Scatter(x=xs, y=SC.p_at(xs), line=dict(color=ACCENT, width=2.5), hovertemplate="%{x} points → P %{y:.1%}<extra></extra>"))
        fig.update_layout(title="Points → probability of RTO", xaxis_title="Risk points", yaxis_tickformat=".0%", height=360,
                          margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Why not fixed 0-30 / 30-60 / 60-100 tiers? With a 17% base rate, about {(PTS_TE >= 30).mean():.0%} of orders score above 30. "
                   "So the cut-offs are set by capacity instead: who can be messaged without annoying good customers. Adjust them in the sidebar.")

# ================================================================== TAB 3 · VALIDATION
with tabs[2]:
    E = R["eval"]
    st.subheader("Which model? Rules vs scorecard vs gradient boosting")
    comp = pd.DataFrame([{"Model": m, "AUC": e["auc"], "Gini": e["gini"], "KS": e["ks"], "Top-10% RTO rate": e["top10_rate"],
                          "Lift (top 10%)": e["lift10"], "RTOs caught in top 10%": e["top10_capture"]} for m, e in E.items()])
    comp["Explainable to ops?"] = ["Yes, but weights are guesses", "Yes, every point is traceable", "Needs SHAP; hard to audit", "n/a (the hidden truth)"]
    st.dataframe(comp, hide_index=True, width="stretch",
                 column_config={"AUC": st.column_config.NumberColumn(format="%.3f"), "Gini": st.column_config.NumberColumn(format="%.3f"),
                                "KS": st.column_config.NumberColumn(format="%.3f"), "Top-10% RTO rate": st.column_config.NumberColumn(format="percent"),
                                "Lift (top 10%)": st.column_config.NumberColumn(format="%.2f×"), "RTOs caught in top 10%": st.column_config.NumberColumn(format="percent")})
    gap = E["Gradient boosting"]["auc"] - E["Scorecard"]["auc"]
    st.info(f"**Our pick: the scorecard.** It beats the hand-set rules by {E['Scorecard']['auc'] - E['Rules v0']['auc']:.3f} AUC "
            f"and catches {E['Scorecard']['top10_capture']:.0%} of RTOs in the top 10% (rules: {E['Rules v0']['top10_capture']:.0%}). "
            f"Gradient boosting adds only {gap:.3f} AUC on top. That isn't worth losing the ability to tell an ops manager, a seller or a "
            f"customer *why* an order was flagged. It also isn't worth the harder monitoring, when the first pilot has to earn trust. "
            f"Revisit ML once Valmo has a year of labelled confirmations and new signals (rider notes, address-graph confidence) that trees exploit better.")
    st.caption("'Truth (hidden)' scores orders with the true probabilities the data was generated from. No model can beat it. "
               "The gap to it is information that isn't in the features at all, such as a customer's mood on the day.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m, e in E.items():
            g = e["gains"]
            fig.add_trace(go.Scatter(x=g.coverage, y=g.captured, name=m, line=dict(color=MODEL_COLORS[m], width=2.5 if m == "Scorecard" else 1.6,
                                                                                     dash="dot" if m == "Truth (hidden)" else None),
                                     hovertemplate="Contact top %{x:.0%} → catch %{y:.1%} of RTOs<extra>" + m + "</extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, width=1, dash="dash"), hoverinfo="skip"))
        fig.update_layout(title="RTOs caught vs orders contacted (test months)", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="Orders contacted, riskiest first", yaxis_title="RTOs caught", height=420, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with c2:
        cal = R["calibration"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, cal.predicted.max() * 1.1], y=[0, cal.predicted.max() * 1.1], name="Perfect", line=dict(color=GREY, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cal.predicted, y=cal.actual, mode="markers+lines", name="Scorecard", marker=dict(size=9, color=ACCENT),
                                 hovertemplate="Predicted %{x:.1%}<br>Actual %{y:.1%}<extra></extra>"))
        fig.update_layout(title="Calibration: is a 30% prediction really 30%?", xaxis_title="Predicted P(RTO), by decile",
                          yaxis_title="Actual RTO rate", xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=420,
                          margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scorecard AUC, train → test", f"{R['train_auc']['Scorecard']:.3f} → {E['Scorecard']['auc']:.3f}")
    c2.metric("Brier score (lower is better)", f"{R['brier']['Scorecard']:.4f}", f"GBM {R['brier']['Gradient boosting']:.4f}", delta_color="off")
    c3.metric("Score drift (PSI)", f"{R['psi']:.3f}", "under 0.1 stable · 0.1-0.25 watch", delta_color="off")
    hab = TE[TE.prior_orders > 0]
    hab_pts = SC.points(hab)
    in_a = (hab_pts[hab._habitual_refuser.values == 1] >= st.session_state.cut_hi).mean()
    c4.metric("Hidden habitual refusers sent to pre-confirmation", f"{in_a:.0%}",
              f"vs {(hab_pts[hab._habitual_refuser.values == 0] >= st.session_state.cut_hi).mean():.0%} of other returning customers", delta_color="off")
    st.caption("The last number is a recovery test. The data generator secretly marked 7% of customers as habitual refusers. "
               "The model never sees that flag, only their past behaviour, so this shows how well it finds them.")

    st.subheader("What the tiers would do")
    pts = PTS_TE
    y = TE.rto.values
    cA, cB = pts >= st.session_state.cut_hi, (pts >= st.session_state.cut_lo) & (pts < st.session_state.cut_hi)
    cC = ~(cA | cB)
    tiers = pd.DataFrame([{"Tier": TIER_NAME[k], "Share of orders": m.mean(), "RTO rate": y[m].mean() if m.any() else 0,
                           "Share of all RTOs": y[m].sum() / y.sum()} for k, m in (("A", cA), ("B", cB), ("C", cC))])
    st.dataframe(tiers, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in ["Share of orders", "RTO rate", "Share of all RTOs"]})

    st.markdown("**Economics.** Change the assumptions; nothing here is measured yet. The pilot measures it.")
    e1, e2, e3, e4 = st.columns(4)
    qA = e1.slider("RTOs prevented in pre-confirmation tier", 0, 60, 30, format="%d%%") / 100
    qB = e2.slider("RTOs prevented in soft tier", 0, 30, 10, format="%d%%") / 100
    val_saved = e3.select_slider("Value of one prevented RTO", [120, 170], value=120, format_func=lambda v: f"₹{v}")
    ships = e4.number_input("Shipments a year (crore)", 10, 300, 160)
    n = len(y)
    prevented = qA * y[cA].sum() + qB * y[cB].sum()
    msg_cost = cA.sum() * (0.13 + 0.35 * 0.25) + cB.sum() * 0.13
    new_rto = (y.sum() - prevented) / n
    per_order = (prevented * val_saved - msg_cost) / n
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("RTO rate", f"{new_rto:.1%}", f"{(new_rto - y.mean()) * 100:+.1f} pts", delta_color="inverse")
    k2.metric("RTOs prevented", f"{prevented / y.sum():.1%} of all")
    k3.metric("Messaging cost per order", f"₹{msg_cost / n:.3f}")
    k4.metric("Net saving a year", f"₹{per_order * ships * 1e7 / 1e7:,.0f} cr")
    st.caption(f"₹120 is the deck's conservative value (₹170 avoided minus ₹50 forward trip re-spent on a saved delivery). ₹170 applies to pre-pickup cancels. "
               f"Messaging costs ₹0.13 per WhatsApp, plus an IVR call at ₹0.25 for the ~35% who don't reply. That's so cheap the cut-off is a "
               f"customer-experience choice, not a cost one.")

# ================================================================== TAB 4 · FLOW
with tabs[3]:
    ctx = st.session_state.order_ctx
    flow.render(ctx, ctx["tier"])

# ================================================================== TAB 5 · REAL DATA
with tabs[4]:
    J = json.loads((ROOT / "data" / "amazon_scorecard_results.json").read_text())
    te = J["test_eval"]
    st.markdown(f"The same WoE + logistic method, run on **{J['rows_labelled']:,} real Indian e-commerce orders**: a public Amazon.in seller report "
                f"(Mar-Jun 2022), trained on {J['train']['period']}, tested on {J['test']['period']}. "
                "It checks that the method holds up on messy real data. It can't validate the Valmo model, because this file has no customer IDs "
                "and no payment mode.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test AUC", f"{te['auc']:.3f}", f"95% range {te['auc_ci'][0]:.2f}-{te['auc_ci'][1]:.2f}", delta_color="off")
    c2.metric("KS", f"{te['ks']:.3f}")
    c3.metric("Top-decile RTO rate", f"{te['dec'][0]['rto']:.1%}", f"vs {te['base']:.1%} overall", delta_color="off")
    c4.metric("RTOs caught in top 10%", f"{te['top10_capture']:.1%}")
    c1, c2 = st.columns(2)
    with c1:
        fv = pd.DataFrame(J["features"]).sort_values("iv")
        fig = go.Figure(go.Bar(y=fv.label, x=fv.iv, orientation="h", marker_color=[ACCENT if k else GREY for k in fv.kept],
                               text=[f"{v:.3f}" for v in fv.iv], textposition="outside"))
        fig.add_vline(x=0.02, line_dash="dash", line_color=RED)
        fig.update_layout(title="Information Value on real data (blue = kept)", height=360, margin=dict(l=10, r=30, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        dd = pd.DataFrame(te["dec"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dd.d, y=dd.rto, name="Actual", marker_color=ACCENT))
        fig.add_trace(go.Scatter(x=dd.d, y=dd.pred, name="Predicted", mode="markers", marker=dict(size=10, color="#333")))
        fig.add_hline(y=te["base"], line_dash="dash", line_color=GREY)
        fig.update_layout(title="June 2022: actual vs predicted RTO by decile", xaxis_title="Decile (1 = riskiest)", yaxis_tickformat=".0%",
                          height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    st.markdown("**What it shows:** only location carried signal (area RTO history and metro vs non-metro). Price, size, category and dates didn't. "
                "The features that matter most in the synthetic Valmo model (past refusals, past RTO rate, COD) aren't in any public dataset. "
                "**The method works on real orders; the signal it needs is in Valmo's order logs.**")
    st.caption("Caveats: the file appears to be one apparel seller's account. Its RTO rate (6.8%) is far below Valmo's 17%. 'Returned to seller' may include "
               "some post-delivery returns. Reproduce with scripts/amazon_scorecard.py.")

# ================================================================== TAB 6 · DATASET
with tabs[5]:
    st.markdown(f"**{len(df):,} synthetic orders from {df.customer_id.nunique():,} customers in 1,200 pincodes, Oct 2025 – Sep 2026.** "
                "The published numbers are hit exactly. Everything else is a stated assumption, listed below.")
    rto_by = df.groupby("payment_mode").rto.mean()
    cs = df[df.rto == 1].rto_cause.value_counts(normalize=True)
    chk = pd.DataFrame([
        ("Overall RTO rate", "17%", f"{df.rto.mean():.1%}"), ("COD share of orders", "80%", f"{(df.payment_mode == 'COD').mean():.1%}"),
        ("COD RTO rate", "20%", f"{rto_by['COD']:.1%}"), ("Prepaid RTO rate", "5%", f"{rto_by['Prepaid']:.1%}"),
        ("Share: refused at door", "40%", f"{cs['refusal']:.1%}"), ("Share: not available", "28%", f"{cs['unavailable']:.1%}"),
        ("Share: wrong address", "18%", f"{cs['address']:.1%}"), ("Share: fake order", "9%", f"{cs['fake']:.1%}"),
        ("Share: carrier issue", "5%", f"{cs['carrier']:.1%}")], columns=["Target", "Case pack / deck", "Synthetic data"])
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown("**Calibration check**")
        st.dataframe(chk, hide_index=True, width="stretch")
    with c2:
        st.markdown("**Assumptions (challenge any of these)**")
        st.dataframe(pd.DataFrame(ASSUMPTIONS, columns=["Area", "Assumption"]), hide_index=True, width="stretch", height=360)
    st.markdown("**Data dictionary**")
    dic = [
        ("order_id / order_date / order_hour", "Order identifiers and timing"),
        ("customer_id", "Phone-number-level customer key"),
        ("pincode / pincode_tier", "Delivery pincode and Metro / Tier-2 / Tier-3"),
        ("distance_to_hub_km / promised_delivery_days", "Last-mile distance and promised delivery time"),
        ("payment_mode", "COD or Prepaid"),
        ("category / size_variant_item / multiple_sizes_in_cart", "Cart contents"),
        ("order_value / pincode_avg_order_value / value_to_pincode_aov", "Order value and how unusual it is for the area"),
        ("is_new_address / map_pin_dropped / landmark_only_address", "Address quality at checkout"),
        ("festival_sale", "Order placed during a sale peak"),
        ("prior_orders / prior_rto / prior_delivered / prior_cod_refusals / prior_rto_rate", "Customer history, point-in-time"),
        ("days_since_last_order / customer_tenure_days", "Customer recency and tenure"),
        ("pincode_rto_rate", "Pincode's past RTO rate, point-in-time, smoothed toward 17%"),
        ("rto / rto_cause", "Outcome (target) and cause"),
        ("_true_p_rto / _habitual_refuser", "Hidden truth for the recovery test. Never used as features."),
    ]
    st.dataframe(pd.DataFrame(dic, columns=["Column(s)", "Meaning"]), hide_index=True, width="stretch")
    st.markdown("**Preview**")
    st.dataframe(df.head(300), hide_index=True, width="stretch", height=320)
    @st.cache_data(show_spinner=False)
    def _csv_gz() -> bytes:
        import gzip
        return gzip.compress(load_data().to_csv(index=False).encode())

    st.download_button("Download the full dataset (.csv.gz)", _csv_gz(), "valmo_synthetic_orders.csv.gz", "application/gzip")
