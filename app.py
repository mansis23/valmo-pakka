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
from pakka.data_gen import ASSUMPTIONS, CATEGORIES, generate
from pakka.scorecard import FEATURES, Scorecard, train_all

st.set_page_config(page_title="Valmo Pakka · RTO scorecard", page_icon="📦", layout="wide")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "valmo_synthetic_orders.csv.gz"
INK, ACCENT, RED, AMBER, GREEN, GREY = "#14283A", "#1F5F8B", "#C0392B", "#C27C0E", "#2F7D4A", "#8A929C"
MODEL_COLORS = {"Rules v0": AMBER, "Scorecard": ACCENT, "Gradient boosting": "#7B4FA0", "Truth (hidden)": GREY}
RTO_LOSS = 170

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'DM Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', 'DM Sans', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 3.6rem; max-width: 1280px; }
.hero { background: #14283A; color: #EEF3F7; border-radius: 16px; padding: 22px 28px; display: flex; flex-wrap: wrap;
        justify-content: space-between; align-items: flex-end; gap: 14px; margin-bottom: 14px; }
.hero .eyebrow { font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: #9FB6C9; font-weight: 700; }
.hero h1 { color: #FFFFFF !important; font-size: 40px; margin: 2px 0 4px; padding: 0; }
.hero h1 span { color: #F2B04B !important; }
.hero p { margin: 0; color: #C9D6E1; max-width: 62ch; }
.hero .chips { display: flex; gap: 8px; flex-wrap: wrap; }
.hero .chips span { border: 1px solid #2F4A61; background: #1B3550; color: #DCE6EE; border-radius: 999px; padding: 4px 12px; font-size: 12.5px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #DCE0D8; }
.stTabs [data-baseweb="tab"] { padding: 10px 16px; border-radius: 10px 10px 0 0; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #FFFFFF; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E1E4DC; border-radius: 12px; padding: 12px 16px; }
[data-testid="stMetricValue"] { font-family: 'Sora', sans-serif; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px !important; }
.sectionlabel { font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: #5B6572; font-weight: 700; margin: 4px 0 6px; }
.scorebox { display: flex; gap: 22px; align-items: center; background: #FFFFFF; border: 1px solid #E1E4DC; border-radius: 16px; padding: 18px 22px; }
.scorebox .big { font-family: 'Sora', sans-serif; font-size: 46px; font-weight: 700; line-height: 1; color: #14283A; }
.scorebox .big small { font-size: 18px; color: #8A929C; font-weight: 600; }
.scorebox .lbl { font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: #5B6572; font-weight: 700; }
.scorebox .p { margin-top: 6px; color: #2B3540; }
.pill { display: inline-block; color: #FFFFFF; font-weight: 700; padding: 5px 12px; border-radius: 999px; font-size: 13px; margin-top: 10px; }
.muted { color: #5B6572; font-size: 13px; margin-left: 8px; }
.next { background: #FFFFFF; border: 1px solid #E1E4DC; border-left: 5px solid var(--c); border-radius: 12px; padding: 14px 18px; margin-top: 12px; }
.next h4 { margin: 0 0 6px; font-size: 15px; }
.next ol { margin: 0; padding-left: 20px; } .next li { margin: 3px 0; }
.models { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.mcard { background: #FFFFFF; border: 1px solid #E1E4DC; border-radius: 14px; padding: 14px 16px; }
.mcard.pick { border: 2px solid #1F5F8B; box-shadow: 0 4px 14px rgba(31, 95, 139, .12); }
.mcard .tag { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; color: #5B6572; }
.mcard.pick .tag { color: #1F5F8B; }
.mcard .name { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 16px; margin: 2px 0 8px; color: #14283A; }
.mcard .auc { font-family: 'Sora', sans-serif; font-size: 30px; font-weight: 700; color: #14283A; }
.mcard .sub { font-size: 13px; color: #4A5562; }
@media (max-width: 900px) { .models { grid-template-columns: repeat(2, minmax(0, 1fr)); } .scorebox { flex-direction: column; align-items: flex-start; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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
PTS_TE = R["points_test"]

# default cut-offs: riskiest ~10% -> pre-confirmation, next ~20% -> soft reminder (the pilot design)
_recent = SC.points(TR[TR.order_date >= TR.order_date.max() - pd.Timedelta(days=60)])
DEF_HI = int(np.round(np.quantile(_recent, 0.90)))
DEF_LO = int(np.round(np.quantile(_recent, 0.70)))
st.session_state.setdefault("cut_hi", DEF_HI)
st.session_state.setdefault("cut_lo", DEF_LO)

TIER_NAME = {"A": "Pre-confirmation", "B": "Soft reminder", "C": "No contact"}
TIER_COLOR = {"A": RED, "B": AMBER, "C": GREEN}


def tier_of(points: float) -> str:
    return "A" if points >= st.session_state.cut_hi else "B" if points >= st.session_state.cut_lo else "C"


def ring(points: float, color: str) -> str:
    r, c = 42, 2 * np.pi * 42
    return (f"<svg width='112' height='112' viewBox='0 0 112 112'><circle cx='56' cy='56' r='{r}' fill='none' stroke='#ECEFE8' stroke-width='12'/>"
            f"<circle cx='56' cy='56' r='{r}' fill='none' stroke='{color}' stroke-width='12' stroke-linecap='round' "
            f"stroke-dasharray='{c * points / 100:.1f} {c:.1f}' transform='rotate(-90 56 56)'/>"
            f"<text x='56' y='63' text-anchor='middle' font-family='Sora, sans-serif' font-size='24' font-weight='700' fill='#14283A'>{points:.0f}</text></svg>")


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Tier cut-offs")
    st.slider("Pre-confirmation from (points)", 0, 100, key="cut_hi")
    st.slider("Soft reminder from (points)", 0, 100, key="cut_lo")
    if st.session_state.cut_lo > st.session_state.cut_hi:
        st.warning("The soft-reminder cut-off should be below the pre-confirmation cut-off.")
    a = (PTS_TE >= st.session_state.cut_hi).mean()
    b = ((PTS_TE >= st.session_state.cut_lo) & (PTS_TE < st.session_state.cut_hi)).mean()
    st.markdown(f"- **Pre-confirmation:** {a:.0%} of orders\n- **Soft reminder:** {b:.0%}\n- **No contact:** {1 - a - b:.0%}")
    st.caption(f"Defaults ({DEF_HI} / {DEF_LO}) send the riskiest ~10% to pre-confirmation and the next ~20% a reminder.")
    if st.button("Reset cut-offs"):
        st.session_state.cut_hi, st.session_state.cut_lo = DEF_HI, DEF_LO
        st.rerun()
    st.divider()
    st.caption("All orders are synthetic, calibrated to the case-pack figures (17% RTO, 20% COD, 5% prepaid). No Valmo data is used.")

st.markdown("""<div class="hero"><div><div class="eyebrow">Meesho DICE S3 · Team Prod Gods, IIT Kanpur</div>
<h1>Valmo <span>Pakka</span></h1><p>Pehle Pakka, Phir Package. Score every order at checkout, confirm the risky ones on WhatsApp
before pickup, and make cancelling cheap and early.</p></div>
<div class="chips"><span>WoE + logistic scorecard</span><span>2.1 lakh synthetic orders</span><span>Live WhatsApp flow mock</span></div></div>""",
            unsafe_allow_html=True)

tabs = st.tabs(["① Score an order", "② Model & validation", "③ Customer flow", "④ Real-data check", "⑤ The dataset"])

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
    for k, v in PRESETS[st.session_state.preset].items():
        st.session_state["in_" + k] = v


if "in_prior_orders" not in st.session_state:
    st.session_state.preset = "First-time rural COD, landmark address"
    apply_preset()

with tabs[0]:
    st.selectbox("Load an example order", list(PRESETS), key="preset", on_change=apply_preset)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="sectionlabel">Customer history</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="sectionlabel">Address & location</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="sectionlabel">This order</div>', unsafe_allow_html=True)
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
        col = TIER_COLOR[t]
        st.markdown(f"""<div class="scorebox">{ring(pts, col)}<div>
<div class="lbl">Risk score</div><div class="big">{pts:.0f}<small> / 100</small></div>
<div class="p">Chance of RTO <b>{p:.1%}</b> · {p / SC.base_rate:.1f}× the average order</div>
<span class="pill" style="background:{col}">{TIER_NAME[t]}</span><span class="muted">Riskier than {pct:.0%} of orders</span>
</div></div>""", unsafe_allow_html=True)
        if t == "A":
            pay = (f"Offer ₹{cashback} back for paying by UPI, or a ₹{token} token now with the rest on delivery." if cod and cashback
                   else f"Offer a ₹{token} UPI token, rest on delivery." if cod else "Already prepaid, so no payment nudge.")
            steps = ["WhatsApp within 15 minutes, before the seller packs: <b>Confirm / Fix address / Cancel</b>.", pay,
                     "No reply in 6 hours: IVR call. Still nothing: ship flagged <b>Unconfirmed</b>, and the rider calls first."]
        elif t == "B":
            steps = ["One WhatsApp reminder with a one-tap cancel link.",
                     f"Offer ₹{cashback} back for paying by UPI." if cod and cashback else "No payment nudge.",
                     "No reply needed. It ships as normal."]
        else:
            steps = ["No message. It ships as normal.", "Messaging low-risk orders adds friction without moving RTO."]
        st.markdown(f'<div class="next" style="--c:{col}"><h4>What happens next</h4><ol>' + "".join(f"<li>{s}</li>" for s in steps) + "</ol></div>",
                    unsafe_allow_html=True)

        expl = SC.explain(row)
        expl = expl[expl.points.abs() >= 0.05]
        base = SC.base_points()
        cum = base + np.concatenate([[0], np.cumsum(expl.points.values)])
        fig = go.Figure(go.Waterfall(
            orientation="h", measure=["absolute"] + ["relative"] * len(expl) + ["total"],
            y=["Average order"] + [f"{r.feature}: {r.bin}" for r in expl.itertuples()] + ["This order"],
            x=[base] + list(expl.points) + [0],
            text=[f"{base:.0f}"] + [f"{v:+.1f}" for v in expl.points] + [f"{pts:.0f}"], textposition="outside",
            increasing=dict(marker_color=RED), decreasing=dict(marker_color=GREEN), totals=dict(marker_color=INK),
            connector=dict(line=dict(color="rgba(128,128,128,.35)")),
        ))
        fig.update_layout(title=dict(text="Why this score", font=dict(family="Sora", size=16)), height=120 + 34 * (len(expl) + 2),
                          yaxis=dict(autorange="reversed"), margin=dict(l=10, r=40, t=46, b=10), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="Risk points (red adds risk, green removes it)", range=[max(-5, cum.min() - 12), min(115, cum.max() + 12)],
                                     gridcolor="#E6E8E1"))
        st.plotly_chart(fig, width="stretch")

    st.session_state.order_ctx = dict(
        id="#MSH-48213", name="Priya", item=("Cotton kurta, size M" + (" + L" if multi else "")) if cat == "Apparel" else cat,
        value=int(val), cod=bool(cod), addr=("Hanuman Mandir ke paas, Kalyanpur, Kanpur" if lm else "H.No. 117/42, Gali 3, Kalyanpur, Kanpur"),
        p=p, p_prepaid=p_pre, points=pts, tier=t, tier_label=TIER_NAME[t], cashback=cashback if cod else 0, token=token if cod else 0)

# ================================================================== TAB 2 · MODEL & VALIDATION
with tabs[1]:
    E = R["eval"]
    st.markdown('<div class="sectionlabel">Rules → scorecard → ML, tested on 3 held-out months</div>', unsafe_allow_html=True)
    meta = [("Rules v0", "Baseline", "Hand-set points"), ("Scorecard", "Our pick", "WoE scorecard"),
            ("Gradient boosting", "Challenger", "Gradient boosting"), ("Truth (hidden)", "Ceiling", "Hidden truth")]
    st.markdown('<div class="models">' + "".join(
        f'<div class="mcard{" pick" if m == "Scorecard" else ""}"><div class="tag">{tag}</div><div class="name">{name}</div>'
        f'<div class="auc">{E[m]["auc"]:.3f}</div><div class="sub">AUC</div>'
        f'<div class="sub" style="margin-top:8px"><b>{E[m]["top10_capture"]:.0%}</b> of RTOs caught by contacting the riskiest 10%</div></div>'
        for m, tag, name in meta) + "</div>", unsafe_allow_html=True)
    gap = E["Gradient boosting"]["auc"] - E["Scorecard"]["auc"]
    st.info(f"**Why the scorecard:** it beats hand-set rules by {E['Scorecard']['auc'] - E['Rules v0']['auc']:.3f} AUC, and ML adds only {gap:.3f} more. "
            "That isn't worth losing the ability to explain every flag to ops, sellers and customers.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m, e in E.items():
            g = e["gains"]
            fig.add_trace(go.Scatter(x=g.coverage, y=g.captured, name=m, line=dict(color=MODEL_COLORS[m], width=3 if m == "Scorecard" else 1.6,
                                                                                     dash="dot" if m == "Truth (hidden)" else None),
                                     hovertemplate="Contact top %{x:.0%} → catch %{y:.1%} of RTOs<extra>" + m + "</extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(color=GREY, width=1, dash="dash"), hoverinfo="skip"))
        fig.update_layout(title=dict(text="RTOs caught vs orders contacted", font=dict(family="Sora", size=16)), xaxis_tickformat=".0%",
                          yaxis_tickformat=".0%", xaxis_title="Orders contacted, riskiest first", yaxis_title="RTOs caught", height=420,
                          margin=dict(l=10, r=10, t=46, b=10), legend=dict(orientation="h", y=-0.25), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        cal = R["calibration"]
        mx = cal.predicted.max() * 1.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], name="Perfect", line=dict(color=GREY, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cal.predicted, y=cal.actual, mode="markers+lines", name="Scorecard", line=dict(color=ACCENT, width=2.5),
                                 marker=dict(size=9, color=ACCENT), hovertemplate="Predicted %{x:.1%}<br>Actual %{y:.1%}<extra></extra>"))
        fig.update_layout(title=dict(text="Calibration: is a 30% prediction really 30%?", font=dict(family="Sora", size=16)),
                          xaxis_title="Predicted P(RTO), by decile", yaxis_title="Actual RTO rate", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          height=420, margin=dict(l=10, r=10, t=46, b=10), legend=dict(orientation="h", y=-0.25), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    hab = TE[TE.prior_orders > 0]
    hab_pts = SC.points(hab)
    is_hab = hab._habitual_refuser.values == 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC, train → test", f"{R['train_auc']['Scorecard']:.3f} → {E['Scorecard']['auc']:.3f}", "no overfitting", delta_color="off")
    c2.metric("Brier score", f"{R['brier']['Scorecard']:.4f}", f"GBM {R['brier']['Gradient boosting']:.4f}", delta_color="off")
    c3.metric("Score drift (PSI)", f"{R['psi']:.3f}", "under 0.1 stable · 0.1-0.25 watch", delta_color="off")
    c4.metric("Hidden habitual refusers flagged", f"{(hab_pts[is_hab] >= st.session_state.cut_hi).mean():.0%}",
              f"vs {(hab_pts[~is_hab] >= st.session_state.cut_hi).mean():.0%} of other returning customers", delta_color="off")

    st.markdown('<div class="sectionlabel" style="margin-top:18px">What the tiers would do</div>', unsafe_allow_html=True)
    y = TE.rto.values
    cA, cB = PTS_TE >= st.session_state.cut_hi, (PTS_TE >= st.session_state.cut_lo) & (PTS_TE < st.session_state.cut_hi)
    cC = ~(cA | cB)
    tiers = pd.DataFrame([{"Tier": TIER_NAME[k], "Share of orders": m.mean(), "RTO rate": y[m].mean() if m.any() else 0,
                           "Share of all RTOs": y[m].sum() / y.sum()} for k, m in (("A", cA), ("B", cB), ("C", cC))])
    st.dataframe(tiers, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent") for c in ["Share of orders", "RTO rate", "Share of all RTOs"]})

    with st.container(border=True):
        st.markdown("**Economics.** These are assumptions; the pilot measures them.")
        e1, e2, e3, e4 = st.columns(4)
        qA = e1.slider("RTOs prevented, pre-confirmation tier", 0, 60, 30, format="%d%%") / 100
        qB = e2.slider("RTOs prevented, soft tier", 0, 30, 10, format="%d%%") / 100
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
        k4.metric("Net saving a year", f"₹{per_order * ships:,.0f} cr")

    with st.expander("See the scorecard itself (features, bins and points)"):
        iv = pd.DataFrame([{"Feature": FEATURES[k]["label"], "IV": SC.iv[k], "In model": k in SC.kept} for k in FEATURES]).sort_values("IV")
        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig = go.Figure(go.Bar(y=iv.Feature, x=iv.IV, orientation="h", marker_color=[ACCENT if s else GREY for s in iv["In model"]],
                                   text=[f"{v:.3f}" for v in iv.IV], textposition="outside"))
            fig.add_vline(x=0.02, line_dash="dash", line_color=RED)
            fig.update_layout(title=dict(text="Information Value (blue = in model)", font=dict(family="Sora", size=15)), height=500,
                              margin=dict(l=10, r=30, t=46, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
        with c2:
            ptab = SC.points_table()
            st.dataframe(ptab.rename(columns={"feature": "Feature", "bin": "Bin", "orders": "Orders", "rto_rate": "RTO rate", "woe": "WoE", "points": "Points"}),
                         hide_index=True, width="stretch", height=460,
                         column_config={"RTO rate": st.column_config.NumberColumn(format="percent"),
                                        "WoE": st.column_config.NumberColumn(format="%.3f"),
                                        "Points": st.column_config.NumberColumn(format="%+.1f")})
            st.download_button("Download the scorecard (CSV)", ptab.to_csv(index=False).encode(), "valmo_pakka_scorecard.csv", "text/csv")

# ================================================================== TAB 3 · FLOW
with tabs[2]:
    ctx = st.session_state.order_ctx
    flow.render(ctx, ctx["tier"])

# ================================================================== TAB 4 · REAL DATA
with tabs[3]:
    J = json.loads((ROOT / "data" / "amazon_scorecard_results.json").read_text())
    te = J["test_eval"]
    st.markdown(f"The same WoE + logistic method on **{J['rows_labelled']:,} real Indian e-commerce orders** from a public Amazon.in seller report "
                f"(trained {J['train']['period']}, tested {J['test']['period']}). It shows the method holds up on messy real data.")
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
        fig.update_layout(title=dict(text="Information Value on real data (blue = kept)", font=dict(family="Sora", size=15)), height=360,
                          margin=dict(l=10, r=30, t=46, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        dd = pd.DataFrame(te["dec"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dd.d, y=dd.rto, name="Actual", marker_color=ACCENT))
        fig.add_trace(go.Scatter(x=dd.d, y=dd.pred, name="Predicted", mode="markers", marker=dict(size=10, color=INK)))
        fig.add_hline(y=te["base"], line_dash="dash", line_color=GREY)
        fig.update_layout(title=dict(text="June 2022: actual vs predicted RTO by decile", font=dict(family="Sora", size=15)),
                          xaxis_title="Decile (1 = riskiest)", yaxis_tickformat=".0%", height=360, margin=dict(l=10, r=10, t=46, b=10),
                          legend=dict(orientation="h", y=-0.25), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**What it shows:** only location carried signal here. This file has no customer history or payment mode, which are the strongest "
                "features in the Valmo model. **The method works on real orders; the signal it needs is in Valmo's order logs.**")
    st.caption("Caveats: one apparel seller's account, 6.8% RTO (far below Valmo's 17%), and 'Returned to seller' may include some post-delivery returns.")

# ================================================================== TAB 5 · DATASET
with tabs[4]:
    st.markdown(f"**{len(df):,} synthetic orders from {df.customer_id.nunique():,} customers in 1,200 pincodes, Oct 2025 – Sep 2026.** "
                "The published numbers are hit exactly. Everything else is a stated assumption.")
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
        st.markdown('<div class="sectionlabel">Calibration check</div>', unsafe_allow_html=True)
        st.dataframe(chk, hide_index=True, width="stretch")
    with c2:
        st.markdown('<div class="sectionlabel">Assumptions</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(ASSUMPTIONS, columns=["Area", "Assumption"]), hide_index=True, width="stretch", height=360)
    st.markdown('<div class="sectionlabel">Preview</div>', unsafe_allow_html=True)
    st.dataframe(df.head(300), hide_index=True, width="stretch", height=320)
    st.caption("Columns starting with _ are the hidden truth used only for the recovery test. They are never model features.")

    @st.cache_data(show_spinner=False)
    def _csv_gz() -> bytes:
        import gzip
        return gzip.compress(load_data().to_csv(index=False).encode())

    st.download_button("Download the full dataset (.csv.gz)", _csv_gz(), "valmo_synthetic_orders.csv.gz", "application/gzip")
