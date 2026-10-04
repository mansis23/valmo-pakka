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
from pakka.data_gen import CATEGORIES, generate
from pakka.scorecard import Scorecard, train_all

st.set_page_config(page_title="Valmo Pakka · RTO scorecard", page_icon="📦", layout="wide", initial_sidebar_state="collapsed")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "valmo_synthetic_orders.csv.gz"
INK, ACCENT, RED, AMBER, GREEN, GREY = "#14283A", "#1F5F8B", "#C0392B", "#C27C0E", "#2F7D4A", "#8A929C"
RTO_LOSS = 170

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'DM Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'Sora', 'DM Sans', sans-serif !important; letter-spacing: -0.01em; }
.block-container { padding-top: 3.6rem; max-width: 1280px; }
[data-testid="stSidebarCollapsedControl"] { display: none; }
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
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px !important; }
.sectionlabel { font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: #5B6572; font-weight: 700; margin: 4px 0 6px; }
.intro { color: #3B4652; margin: 2px 0 10px; }
.scorebox { display: flex; gap: 22px; align-items: center; background: #FFFFFF; border: 1px solid #E1E4DC; border-radius: 16px; padding: 18px 22px; }
.scorebox .big { font-family: 'Sora', sans-serif; font-size: 46px; font-weight: 700; line-height: 1; color: #14283A; }
.scorebox .big small { font-size: 18px; color: #8A929C; font-weight: 600; }
.scorebox .lbl { font-size: 12px; letter-spacing: .09em; text-transform: uppercase; color: #5B6572; font-weight: 700; }
.scorebox .p { margin-top: 6px; color: #2B3540; }
.pill { display: inline-block; color: #FFFFFF; font-weight: 700; padding: 5px 12px; border-radius: 999px; font-size: 13px; margin-top: 10px; }
.scale { position: relative; margin: 14px 2px 26px; height: 12px; display: flex; border-radius: 6px; overflow: visible; }
.scale div { height: 12px; } .scale .mk { position: absolute; top: -6px; width: 4px; height: 24px; background: #14283A; border-radius: 2px; }
.scale .t { position: absolute; top: 16px; font-size: 11.5px; color: #5B6572; white-space: nowrap; }
.next { background: #FFFFFF; border: 1px solid #E1E4DC; border-left: 5px solid var(--c); border-radius: 12px; padding: 14px 18px; margin-top: 12px; }
.next h4 { margin: 0 0 6px; font-size: 15px; }
.next ol { margin: 0; padding-left: 20px; } .next li { margin: 3px 0; }
.why { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
.why .col { background: #FFFFFF; border: 1px solid #E1E4DC; border-radius: 12px; padding: 12px 14px; }
.why h5 { margin: 0 0 8px; font-size: 13px; letter-spacing: .06em; text-transform: uppercase; }
.reason { display: flex; justify-content: space-between; gap: 8px; padding: 5px 0; border-bottom: 1px dashed #E6E8E1; font-size: 14px; }
.reason:last-child { border-bottom: 0; }
.reason b { font-family: 'Sora', sans-serif; white-space: nowrap; }
.stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 8px; }
.stat { background: #FFFFFF; border: 1px solid #E1E4DC; border-radius: 14px; padding: 16px 18px; }
.stat .v { font-family: 'Sora', sans-serif; font-size: 32px; font-weight: 700; color: #14283A; line-height: 1.1; }
.stat .k { color: #3B4652; font-size: 14px; margin-top: 4px; }
@media (max-width: 900px) { .stats, .why { grid-template-columns: 1fr; } .scorebox { flex-direction: column; align-items: flex-start; } }
.models { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 6px 0 10px; }
.mcard { background: #FFFFFF; border: 1px solid #E1E4DC; border-radius: 14px; padding: 14px 16px; }
.mcard.pick { border: 2px solid #C0392B; box-shadow: 0 4px 14px rgba(192, 57, 43, .12); }
.mcard .tag { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; font-weight: 700; color: #5B6572; }
.mcard.pick .tag { color: #C0392B; }
.mcard .name { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 16px; margin: 2px 0 8px; color: #14283A; }
.mcard .auc { font-family: 'Sora', sans-serif; font-size: 30px; font-weight: 700; color: #14283A; }
.mcard .sub { font-size: 13px; color: #4A5562; }
@media (max-width: 900px) { .models { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data + models
@st.cache_data(show_spinner="Generating 2 lakh synthetic orders (seed 42)…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA) if DATA.exists() else generate()
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@st.cache_resource(show_spinner="Fitting the scorecard…")
def load_models():
    return train_all(load_data())


df = load_data()
R = load_models()
SC: Scorecard = R["scorecard"]
TR, TE = R["train"], R["test"]
PTS_TE = R["points_test"]

# Tier cut-offs: riskiest ~10% -> pre-confirmation, next ~20% -> soft reminder (the pilot design)
_recent = SC.points(TR[TR.order_date >= TR.order_date.max() - pd.Timedelta(days=60)])
CUT_HI = int(np.round(np.quantile(_recent, 0.90)))
CUT_LO = int(np.round(np.quantile(_recent, 0.70)))

TIER_NAME = {"A": "Confirm on WhatsApp", "B": "Send a reminder", "C": "Ship as normal"}
TIER_COLOR = {"A": RED, "B": AMBER, "C": GREEN}


def tier_of(points: float) -> str:
    return "A" if points >= CUT_HI else "B" if points >= CUT_LO else "C"


def ring(points: float, color: str) -> str:
    r, c = 42, 2 * np.pi * 42
    return (f"<svg width='112' height='112' viewBox='0 0 112 112'><circle cx='56' cy='56' r='{r}' fill='none' stroke='#ECEFE8' stroke-width='12'/>"
            f"<circle cx='56' cy='56' r='{r}' fill='none' stroke='{color}' stroke-width='12' stroke-linecap='round' "
            f"stroke-dasharray='{c * points / 100:.1f} {c:.1f}' transform='rotate(-90 56 56)'/>"
            f"<text x='56' y='63' text-anchor='middle' font-family='Sora, sans-serif' font-size='24' font-weight='700' fill='#14283A'>{points:.0f}</text></svg>")


def plain(key: str, b: str) -> str:
    """Turn a scorecard bin into a sentence a non-analyst can read."""
    if key == "prior_orders":
        return "First order, no history" if b.startswith("0") else f"{b} past orders"
    if key == "prior_cod_refusals":
        return "Never refused a COD parcel" if b == "0" else f"Refused {b} COD parcel{'s' if b != '1' else ''} before"
    if key == "prior_rto_rate":
        return "No past RTOs" if b == "0%" else f"{b} of past orders came back"
    if key == "days_since_last":
        return f"Last ordered {b} days ago"
    if key == "pincode_rto_rate":
        return f"Pincode's past RTO rate {b.lower()}"
    if key == "payment_mode":
        return "Cash on delivery" if b == "COD" else "Paid online"
    if key == "is_new_address":
        return "New address for this customer" if b == "Yes" else "Address used before"
    if key == "map_pin_dropped":
        return "Map pin dropped" if b == "Yes" else "No map pin"
    if key == "landmark_only":
        return "Landmark-only address" if b == "Yes" else "Full address with house number"
    if key == "promised_days":
        return f"Promised delivery in {b}"
    if key == "pincode_tier":
        return f"{b} pincode"
    return b


st.markdown("""<div class="hero"><div><div class="eyebrow">Meesho DICE S3 · Team Prod Gods, IIT Kanpur</div>
<h1>Valmo <span>Pakka</span></h1><p>Pehle Pakka, Phir Package. Score every order at checkout, confirm the risky ones on WhatsApp
before pickup, and make cancelling cheap and early.</p></div>
<div class="chips"><span>Score at checkout</span><span>Confirm before pickup</span><span>Cancel at ₹0, not ₹170</span></div></div>""",
            unsafe_allow_html=True)

tabs = st.tabs(["① Score an order", "② Customer flow", "③ How we know it works"])

# ================================================================== TAB 1 · SCORE
PRESETS = {
    "First-time rural COD": dict(prior_orders=0, prior_rto=0, prior_ref=0, days_since=0, new_addr=True, pin=False, landmark=True,
                                 tier="Tier-3", pin_rate=26, prom=8, cod=True, cat="Apparel", value=449),
    "Habitual refuser": dict(prior_orders=6, prior_rto=4, prior_ref=3, days_since=12, new_addr=False, pin=False, landmark=False,
                             tier="Tier-2", pin_rate=19, prom=5, cod=True, cat="Apparel", value=899),
    "Repeat buyer, new address": dict(prior_orders=3, prior_rto=0, prior_ref=0, days_since=45, new_addr=True, pin=False, landmark=True,
                                      tier="Tier-2", pin_rate=17, prom=5, cod=True, cat="Home & kitchen", value=549),
    "Loyal metro, prepaid": dict(prior_orders=14, prior_rto=0, prior_ref=0, days_since=9, new_addr=False, pin=True, landmark=False,
                                 tier="Metro", pin_rate=11, prom=3, cod=False, cat="Beauty", value=299),
}


def apply_preset():
    name = st.session_state.get("preset")
    if name in PRESETS:
        for k, v in PRESETS[name].items():
            st.session_state["in_" + k] = v


if "in_prior_orders" not in st.session_state:
    st.session_state.preset = "First-time rural COD"
    apply_preset()

with tabs[0]:
    st.markdown('<p class="intro">Pick an example or change any detail. Pakka scores the order from 0 to 100 and decides what to do <b>before the parcel ships</b>.</p>',
                unsafe_allow_html=True)
    st.segmented_control("Example orders", list(PRESETS), key="preset", on_change=apply_preset, label_visibility="collapsed")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="sectionlabel">Who is ordering</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            po = c1.number_input("Past orders", 0, 60, key="in_prior_orders")
            if po > 0:
                st.session_state.in_prior_rto = min(st.session_state.in_prior_rto, po)
                pr = c2.number_input("Of those, came back (RTO)", 0, int(po), key="in_prior_rto")
                st.session_state.in_prior_ref = min(st.session_state.in_prior_ref, pr)
                c3, c4 = st.columns(2)
                prf = c3.number_input("Refused at the door (COD)", 0, int(pr), key="in_prior_ref")
                ds = c4.number_input("Days since last order", 0, 365, key="in_days_since")
                na = st.checkbox("Ordering to a new address", key="in_new_addr")
            else:
                c2.caption("First order, so there's no history yet.")
                pr, prf, ds, na = 0, 0, np.nan, True
        with st.container(border=True):
            st.markdown('<div class="sectionlabel">Where it\'s going</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            tier = c1.selectbox("Area", ["Metro", "Tier-2", "Tier-3"], key="in_tier")
            prom = c2.slider("Promised delivery (days)", 2, 10, key="in_prom")
            pinr = st.slider("How often parcels to this pincode come back (%)", 5, 40, key="in_pin_rate")
            mp = st.checkbox("Customer dropped a map pin", key="in_pin")
            lm = st.checkbox("Landmark only, no house number (e.g. 'Mandir ke paas')", key="in_landmark")
        with st.container(border=True):
            st.markdown('<div class="sectionlabel">What\'s ordered</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns([1.3, 1.2, 1])
            cod = c1.toggle("Cash on delivery", key="in_cod")
            cat = c2.selectbox("Item", CATEGORIES, key="in_cat")
            val = c3.number_input("Value (₹)", 50, 20000, step=50, key="in_value")

    row = pd.DataFrame([{
        "prior_orders": po, "prior_rto": pr if po else 0, "prior_cod_refusals": prf if po else 0,
        "prior_rto_rate": (pr / po) if po else np.nan, "days_since_last_order": ds if po else np.nan,
        "pincode_rto_rate": pinr / 100, "payment_mode": "COD" if cod else "Prepaid",
        "is_new_address": int(na), "map_pin_dropped": int(mp), "landmark_only_address": int(lm),
        "promised_delivery_days": prom, "pincode_tier": tier,
        # not used by the scorecard; kept so the row has every column the binning code expects
        "distance_to_hub_km": 8.0, "value_to_pincode_aov": 1.0, "category": cat, "multiple_sizes_in_cart": 0,
        "order_hour": 14, "festival_sale": 0, "size_variant_item": int(cat in ("Apparel", "Footwear")),
    }])
    p = float(SC.predict_proba(row)[0])
    pts = float(SC.points(row)[0])
    t = tier_of(pts)
    alt = row.copy(); alt["payment_mode"] = "Prepaid" if cod else "COD"
    p_alt = float(SC.predict_proba(alt)[0])
    p_pre = p_alt if cod else p
    cap = max(0.0, ((p if cod else p_alt) - p_pre) * RTO_LOSS)
    cashback = 15 if cap >= 15 else 10 if cap >= 10 else 0
    token = (30 if val < 400 else 50) if t == "A" else 0

    with right:
        col = TIER_COLOR[t]
        st.markdown(f"""<div class="scorebox">{ring(pts, col)}<div>
<div class="lbl">Risk score</div><div class="big">{pts:.0f}<small> / 100</small></div>
<div class="p">About <b>{round(p * 10)} in 10</b> orders like this come back · {p / SC.base_rate:.1f}× the average</div>
<span class="pill" style="background:{col}">{TIER_NAME[t]}</span></div></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="scale">
<div style="width:{CUT_LO}%;background:{GREEN};border-radius:6px 0 0 6px"></div>
<div style="width:{CUT_HI - CUT_LO}%;background:{AMBER}"></div>
<div style="width:{100 - CUT_HI}%;background:{RED};border-radius:0 6px 6px 0"></div>
<span class="mk" style="left:calc({pts:.1f}% - 2px)"></span>
<span class="t" style="left:0">Ship</span>
<span class="t" style="left:{CUT_LO}%">Remind ({CUT_LO}+)</span>
<span class="t" style="left:{CUT_HI + 4}%">Confirm ({CUT_HI}+)</span></div>""", unsafe_allow_html=True)

        if t == "A":
            pay = (f"Offer ₹{cashback} back for paying by UPI, or a ₹{token} token now with the rest on delivery." if cod and cashback
                   else f"Offer a ₹{token} UPI token, rest on delivery." if cod else "Already paid, so no payment nudge.")
            steps = ["WhatsApp within 15 minutes, before the seller packs: <b>Confirm / Fix address / Cancel</b>.", pay,
                     "No reply in 6 hours: automated call. Still nothing: ship it, and the rider calls first."]
        elif t == "B":
            steps = ["One WhatsApp reminder with a one-tap cancel link.",
                     f"Offer ₹{cashback} back for paying by UPI." if cod and cashback else "No payment nudge.",
                     "No reply needed. It ships as normal."]
        else:
            steps = ["No message. It ships as normal.", "Messaging low-risk orders adds friction without cutting RTO."]
        st.markdown(f'<div class="next" style="--c:{col}"><h4>What happens next</h4><ol>' + "".join(f"<li>{s}</li>" for s in steps) +
                    "</ol><div style='font-size:13px;color:#5B6572;margin-top:6px'>See it from the customer's side in tab ②.</div></div>",
                    unsafe_allow_html=True)

        expl = SC.explain(row)
        up = expl[expl.points >= 0.5].head(4)
        down = expl[expl.points <= -0.5].sort_values("points").head(4)

        def items(d, color):
            if d.empty:
                return "<div class='reason'><span style='color:#8A929C'>Nothing significant</span></div>"
            return "".join(f"<div class='reason'><span>{plain(r.key, r.bin)}</span><b style='color:{color}'>{r.points:+.0f}</b></div>" for r in d.itertuples())

        st.markdown(f"""<div class="why"><div class="col"><h5 style="color:{RED}">Pushing risk up</h5>{items(up, RED)}</div>
<div class="col"><h5 style="color:{GREEN}">Pulling risk down</h5>{items(down, GREEN)}</div></div>""", unsafe_allow_html=True)

    st.session_state.order_ctx = dict(
        id="#MSH-48213", name="Priya", item=("Cotton kurta, size M" if cat == "Apparel" else cat),
        value=int(val), cod=bool(cod), addr=("Hanuman Mandir ke paas, Kalyanpur, Kanpur" if lm else "H.No. 117/42, Gali 3, Kalyanpur, Kanpur"),
        p=p, p_prepaid=p_pre, points=pts, tier=t, tier_label=TIER_NAME[t], cashback=cashback if cod else 0, token=token if cod else 0)

# ================================================================== TAB 2 · FLOW
with tabs[1]:
    ctx = st.session_state.order_ctx
    flow.render(ctx, ctx["tier"])

# ================================================================== TAB 3 · HOW WE KNOW IT WORKS
with tabs[2]:
    E = R["eval"]
    J = json.loads((ROOT / "data" / "amazon_scorecard_results.json").read_text())
    st.markdown('<p class="intro">Trained on 9 months of synthetic Valmo orders and tested on 3 months it never saw.</p>', unsafe_allow_html=True)
    g30 = float(E["Scorecard"]["gains"].captured.iloc[30])
    st.markdown(f"""<div class="stats">
<div class="stat"><div class="v">{E['Scorecard']['top10_capture']:.0%}</div><div class="k">of all RTOs caught by messaging just the riskiest 10% of orders</div></div>
<div class="stat"><div class="v">{g30:.0%}</div><div class="k">caught by messaging the riskiest 30% (confirm + reminder tiers)</div></div>
<div class="stat"><div class="v">{E['Scorecard']['top10_rate'] / E['Scorecard']['base']:.1f}×</div><div class="k">RTO rate in the confirm tier vs the average order</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sectionlabel" style="margin-top:10px">Rules → scorecard → ML</div>', unsafe_allow_html=True)
    meta = [("Rules v0", "Baseline", "Simple rules"), ("Scorecard", "Our pick", "Pakka scorecard"),
            ("Gradient boosting", "Challenger", "Machine learning"), ("Truth (hidden)", "Ceiling", "Best possible")]
    st.markdown('<div class="models">' + "".join(
        f'<div class="mcard{" pick" if m == "Scorecard" else ""}"><div class="tag">{tag}</div><div class="name">{name}</div>'
        f'<div class="auc">{E[m]["auc"]:.3f}</div><div class="sub">AUC</div>'
        f'<div class="sub" style="margin-top:8px"><b>{E[m]["top10_capture"]:.0%}</b> of RTOs caught by messaging the riskiest 10%</div></div>'
        for m, tag, name in meta) + "</div>", unsafe_allow_html=True)
    st.markdown(f"**Why the scorecard:** it beats simple rules by {E['Scorecard']['auc'] - E['Rules v0']['auc']:.3f} AUC, and machine learning adds only "
                f"{E['Gradient boosting']['auc'] - E['Scorecard']['auc']:.3f} more, so we keep a model that can explain every flag to ops, sellers and customers. "
                "'Best possible' scores orders with the true probabilities the data was generated from.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        fig = go.Figure()
        for m, colr, w in (("Scorecard", RED, 3), ("Rules v0", AMBER, 1.8)):
            gg = E[m]["gains"]
            fig.add_trace(go.Scatter(x=gg.coverage, y=gg.captured, name="Pakka scorecard" if m == "Scorecard" else "Simple rules",
                                     line=dict(color=colr, width=w), hovertemplate="Message top %{x:.0%} → catch %{y:.0%} of RTOs<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Messaging at random", line=dict(color=GREY, width=1, dash="dash"), hoverinfo="skip"))
        fig.add_vrect(x0=0, x1=0.1, fillcolor=RED, opacity=0.08, line_width=0, annotation_text="confirm tier", annotation_position="top left")
        fig.update_layout(title=dict(text="RTOs caught vs orders messaged", font=dict(family="Sora", size=16)), xaxis_tickformat=".0%",
                          yaxis_tickformat=".0%", xaxis_title="Orders messaged, riskiest first", yaxis_title="Share of RTOs caught", height=380,
                          margin=dict(l=10, r=10, t=46, b=10), legend=dict(orientation="h", y=-0.28), paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(gridcolor="#E6E8E1"), yaxis=dict(gridcolor="#E6E8E1"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        cal = R["calibration"]
        mx = float(cal.predicted.max()) * 1.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], name="Perfect", line=dict(color=GREY, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=cal.predicted, y=cal.actual, mode="markers+lines", name="Pakka scorecard", line=dict(color=ACCENT, width=2.5),
                                 marker=dict(size=9, color=ACCENT), hovertemplate="Predicted %{x:.1%}<br>Actual %{y:.1%}<extra></extra>"))
        fig.update_layout(title=dict(text="Is a 30% prediction really 30%?", font=dict(family="Sora", size=16)),
                          xaxis_title="Predicted chance of RTO (10 groups)", yaxis_title="Actual RTO rate", xaxis_tickformat=".0%",
                          yaxis_tickformat=".0%", height=380, margin=dict(l=10, r=10, t=46, b=10), legend=dict(orientation="h", y=-0.28),
                          paper_bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="#E6E8E1"), yaxis=dict(gridcolor="#E6E8E1"))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="sectionlabel">Health checks</div>', unsafe_allow_html=True)
    hab = TE[TE.prior_orders > 0]
    hab_pts = SC.points(hab)
    is_hab = hab._habitual_refuser.values == 1
    checks = [
        (f"{R['train_auc']['Scorecard']:.3f} → {E['Scorecard']['auc']:.3f}", "AUC on training vs unseen months: no overfitting"),
        (f"{R['brier']['Scorecard']:.3f}", f"Brier score, lower is better (ML: {R['brier']['Gradient boosting']:.3f}): equally well calibrated"),
        (f"{R['psi']:.2f}", "Score drift over time (PSI): under 0.1 is stable, 0.1 to 0.25 worth watching"),
        (f"{(hab_pts[is_hab] >= CUT_HI).mean():.0%} vs {(hab_pts[~is_hab] >= CUT_HI).mean():.0%}",
         "hidden habitual refusers sent to confirmation, vs other repeat buyers"),
    ]
    st.markdown('<div class="stats" style="grid-template-columns:repeat(4,minmax(0,1fr))">' + "".join(
        f'<div class="stat"><div class="v" style="font-size:24px">{v}</div><div class="k">{k}</div></div>' for v, k in checks) + "</div>",
        unsafe_allow_html=True)
    st.caption("The last check is a recovery test: the data generator secretly marked 7% of customers as habitual refusers. "
               "The model never sees that flag, only their past behaviour.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="sectionlabel" style="margin-top:10px">Checked on real orders</div>', unsafe_allow_html=True)
        st.markdown(f"On **{J['rows_labelled']:,} real Amazon.in orders** the same method still ranks risk (AUC {J['test_eval']['auc']:.2f}), "
                    "but public data has no customer history or payment mode. Those are the strongest signals, and they're in Valmo's order logs.")
    with c2:
        st.markdown('<div class="sectionlabel" style="margin-top:10px">About the data</div>', unsafe_allow_html=True)
        rto_by = df.groupby("payment_mode").rto.mean()
        st.markdown(f"**{len(df) / 1e5:.1f} lakh synthetic orders**, {df.customer_id.nunique():,} customers, 1,200 pincodes. Calibrated to the case pack: "
                    f"{df.rto.mean():.0%} RTO overall, {rto_by['COD']:.0%} COD, {rto_by['Prepaid']:.0%} prepaid. No Valmo data is used.")

        @st.cache_data(show_spinner=False)
        def _csv_gz() -> bytes:
            import gzip
            return gzip.compress(load_data().to_csv(index=False).encode())

        st.download_button("Download the dataset (.csv.gz)", _csv_gz(), "valmo_synthetic_orders.csv.gz", "application/gzip")
