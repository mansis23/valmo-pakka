"""
Synthetic Valmo order book.

Every number that is *published* comes from the DICE S3 case pack / our deck:
  overall RTO 17%, COD share 80%, COD RTO 20%, prepaid RTO 5%,
  RTO cause shares 40/28/18/9/5 (refusal / unavailable / address / fake / carrier).
Everything else (how customers, pincodes and orders behave) is an ASSUMPTION,
written down in ASSUMPTIONS below so anyone can challenge it.

Design: outcomes are driven by *hidden* traits (a customer's true refusal habit,
how reachable they are, how messy a pincode is). The model only ever sees what
Valmo would see at checkout: past behaviour, address, pincode history, cart.
So the model has to infer the hidden traits, the way it would on real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGETS = {
    "overall_rto": 0.17, "cod_share": 0.80, "cod_rto": 0.20, "prepaid_rto": 0.05,
    "cause_share": {"refusal": 0.40, "unavailable": 0.28, "address": 0.18, "fake": 0.09, "carrier": 0.05},
}
CAUSES = ["refusal", "unavailable", "address", "fake", "carrier"]

# Per-channel RTO rates (%) by payment mode that reproduce the published totals:
#   COD 20%, prepaid 5%, 80/20 mix -> 17% overall, cause shares 40/28/18/9/5.
CHANNEL_TARGET = {
    "cod":     {"refusal": 8.425, "unavailable": 5.6, "address": 3.425, "fake": 1.85, "carrier": 0.70},
    "prepaid": {"refusal": 0.30,  "unavailable": 1.6, "address": 1.80,  "fake": 0.10, "carrier": 1.20},
}

ASSUMPTIONS = [
    ("Customers", "~7% are habitual refusers (hidden trait). The rest vary around a normal level. Matches the industry note that 3-8% of buyers drive most RTO."),
    ("Customers", "Riskier customers choose COD more often. COD share calibrated to 80%."),
    ("Customers", "Orders per customer over 12 months are heavy-tailed: most order 1-3 times, a few order 20+."),
    ("Pincodes", "1,200 pincodes: 25% metro, 35% tier-2, 40% tier-3/rural. Tier-3 is farther from hubs and has more landmark-only addresses."),
    ("Pincodes", "Each pincode has a hidden 'difficulty' that raises every failure type a little."),
    ("Refusal", "Driven by the customer's refusal habit, plus first orders, size-variant apparel, multiple sizes in cart, value far above local average, long promised delivery and late-night orders."),
    ("Not available", "Driven by how reachable the customer is, distance to hub, rural tier and no map pin."),
    ("Address", "Driven by new address, no map pin, landmark-only address and rural tier."),
    ("Fake order", "Driven by first orders, very high value, multiple sizes and new address."),
    ("Carrier", "Driven by distance, rural tier and festival peaks."),
    ("Calibration", "Channel intercepts are solved so totals land on 17% / 20% / 5% and the 40/28/18/9/5 cause mix."),
    ("History", "Past-order features are point-in-time: an order only sees orders placed before it. Outcomes are assumed known by the next order."),
]

CATEGORIES = ["Apparel", "Footwear", "Home & kitchen", "Beauty", "Electronics acc.", "Other"]
CAT_P = [0.45, 0.08, 0.20, 0.12, 0.10, 0.05]
CAT_MEDIAN_VALUE = {"Apparel": 420, "Footwear": 520, "Home & kitchen": 380, "Beauty": 260, "Electronics acc.": 560, "Other": 340}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(z, target, iters=50):
    lo, hi = -15.0, 5.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _sig(mid + z).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(n_customers: int = 80_000, seed: int = 42, start="2025-10-01", days=365) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------------- pincodes ----------------
    P = 1200
    tier = rng.choice(["Metro", "Tier-2", "Tier-3"], size=P, p=[0.25, 0.35, 0.40])
    tier_off = np.select([tier == "Metro", tier == "Tier-2"], [-0.30, 0.0], 0.35)
    pin_difficulty = rng.normal(0, 0.35, P) + tier_off
    dist_mean = np.select([tier == "Metro", tier == "Tier-2"], [5.0, 9.0], 16.0)
    pin_aov = np.select([tier == "Metro", tier == "Tier-2"], [520, 430], 380) * rng.lognormal(0, 0.12, P)
    pin_landmark_p = np.select([tier == "Metro", tier == "Tier-2"], [0.08, 0.18], 0.32)
    pin_weight = np.select([tier == "Metro", tier == "Tier-2"], [1.6, 1.0], 0.7) * rng.lognormal(0, 0.6, P)
    pin_weight /= pin_weight.sum()
    pin_codes = np.array([f"{d}{rng.integers(10000, 99999)}" for d in rng.integers(1, 9, P)])

    # ---------------- customers ----------------
    C = n_customers
    c_pin = rng.choice(P, size=C, p=pin_weight)
    habitual = rng.random(C) < 0.07
    refusal_trait = np.where(habitual, rng.normal(2.4, 0.5, C), rng.normal(0, 0.45, C))
    reach_trait = rng.normal(0, 0.55, C)
    cod_logit = _solve_intercept(0.7 * refusal_trait + 0.3 * tier_off[c_pin], TARGETS["cod_share"]) + 0.7 * refusal_trait + 0.3 * tier_off[c_pin]
    p_cod = _sig(cod_logit)
    n_orders = 1 + rng.negative_binomial(1.1, 0.40, C)          # heavy tail, mean ~2.6
    n_orders = np.minimum(n_orders, 40)

    # ---------------- orders ----------------
    cust = np.repeat(np.arange(C), n_orders)
    N = len(cust)
    t = rng.random(N) * days
    df = pd.DataFrame({"customer_id": cust, "t": t}).sort_values(["customer_id", "t"]).reset_index(drop=True)
    cust = df.customer_id.values
    pin = c_pin[cust]
    k = df.groupby("customer_id").cumcount().values             # 0 = first order

    # addresses: a customer moves to a new address 12% of the time after the first order
    move = (rng.random(N) < 0.12) & (k > 0)
    addr_idx = pd.Series(move.astype(int)).groupby(cust).cumsum().values
    first_use = ~pd.Series(list(zip(cust, addr_idx))).duplicated().values
    is_new_address = first_use.astype(int)
    # landmark-only is a property of the address; map pin dropped more often on repeat addresses
    addr_key = cust.astype(np.int64) * 100 + addr_idx
    addr_u = pd.Series(addr_key).map(pd.Series(rng.random(len(np.unique(addr_key))), index=np.unique(addr_key))).values
    landmark_only = (addr_u < pin_landmark_p[pin]).astype(int)
    map_pin = (rng.random(N) < np.where(is_new_address == 1, 0.45, 0.70)).astype(int)

    is_cod = (rng.random(N) < p_cod[cust]).astype(int)
    category = rng.choice(CATEGORIES, size=N, p=CAT_P)
    size_variant = np.isin(category, ["Apparel", "Footwear"]).astype(int)
    multi_variant = (rng.random(N) < np.where(size_variant == 1, 0.12, 0.03)).astype(int)
    med = pd.Series(category).map(CAT_MEDIAN_VALUE).values
    order_value = np.round(med * rng.lognormal(0, 0.55, N) * (1 + 0.6 * multi_variant), 0)
    value_ratio = order_value / pin_aov[pin]
    distance_km = np.round(rng.gamma(2.0, dist_mean[pin] / 2.0), 1)
    promised_days = np.clip(np.round(2 + distance_km / 6 + np.where(tier[pin] == "Tier-3", 1.5, 0) + rng.normal(0, 1, N)), 2, 10).astype(int)
    hw = np.array([2, 1.5, 1, .6, .5, .6, 1, 2, 3, 4, 5, 5.5, 6, 6, 5.5, 5, 5, 5.5, 6, 6.5, 7, 7, 6, 4], float)
    hour = rng.choice(24, size=N, p=hw / hw.sum())
    late_night = ((hour >= 0) & (hour < 5)).astype(int)
    date = pd.Timestamp(start) + pd.to_timedelta(df.t.values, unit="D")
    festival = (((date >= "2025-10-10") & (date <= "2025-10-25")) | ((date >= "2026-09-20"))).astype(int)

    first_order = (k == 0).astype(int)
    tier3 = (tier[pin] == "Tier-3").astype(int)
    pdiff = pin_difficulty[pin]
    rt = refusal_trait[cust]

    Z = {
        "refusal": 1.15 * rt + 0.45 * first_order + 0.30 * size_variant + 0.50 * multi_variant
                   + 0.45 * (value_ratio > 3) + 0.12 * (promised_days - 5) + 0.30 * late_night + 0.25 * festival + 0.5 * pdiff,
        "unavailable": 0.85 * reach_trait[cust] + 0.025 * distance_km + 0.30 * tier3 + 0.25 * (1 - map_pin) + 0.4 * pdiff,
        "address": 0.90 * is_new_address + 0.60 * (1 - map_pin) + 1.00 * landmark_only + 0.30 * tier3 + 0.015 * distance_km + 0.4 * pdiff,
        "fake": 0.90 * first_order + 0.95 * (value_ratio > 3) + 0.50 * multi_variant + 0.55 * is_new_address + 0.4 * rt,
        "carrier": 0.035 * distance_km + 0.30 * tier3 + 0.25 * festival + 0.3 * pdiff,
    }
    probs = np.zeros((N, 5))
    for g, mask in (("cod", is_cod == 1), ("prepaid", is_cod == 0)):
        for j, c in enumerate(CAUSES):
            a = _solve_intercept(Z[c][mask], CHANNEL_TARGET[g][c] / 100)
            probs[mask, j] = _sig(a + Z[c][mask])
    tot = probs.sum(1)
    scale = np.where(tot > 0.95, 0.95 / tot, 1.0)
    probs *= scale[:, None]
    p_true = probs.sum(1)
    u = rng.random(N)
    cum = probs.cumsum(1)
    cause_idx = (u[:, None] < cum).argmax(1)
    rto = (u < p_true).astype(int)
    cause = np.where(rto == 1, np.array(CAUSES)[cause_idx], "delivered")

    out = pd.DataFrame({
        "order_id": [f"VLM{1000000 + i}" for i in range(N)],
        "order_date": date.normalize(),
        "order_hour": hour,
        "customer_id": [f"C{100000 + c}" for c in cust],
        "pincode": pin_codes[pin],
        "pincode_tier": tier[pin],
        "distance_to_hub_km": distance_km,
        "promised_delivery_days": promised_days,
        "payment_mode": np.where(is_cod == 1, "COD", "Prepaid"),
        "category": category,
        "size_variant_item": size_variant,
        "multiple_sizes_in_cart": multi_variant,
        "order_value": order_value,
        "pincode_avg_order_value": np.round(pin_aov[pin], 0),
        "is_new_address": is_new_address,
        "map_pin_dropped": map_pin,
        "landmark_only_address": landmark_only,
        "festival_sale": festival,
        "rto": rto,
        "rto_cause": cause,
        # hidden truth, kept only for the recovery test. NEVER a model feature.
        "_true_p_rto": np.round(p_true, 4),
        "_habitual_refuser": habitual[cust].astype(int),
    })
    out = add_history_features(out)
    return out.sort_values("order_date", kind="stable").reset_index(drop=True)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time features: each order sees only orders placed strictly before it."""
    df = df.sort_values(["order_date", "order_hour", "order_id"]).reset_index(drop=True)
    g = df.groupby("customer_id", sort=False)
    df["prior_orders"] = g.cumcount()
    df["prior_rto"] = g.rto.cumsum() - df.rto
    refused_cod = ((df.rto_cause == "refusal") & (df.payment_mode == "COD")).astype(int)
    df["prior_cod_refusals"] = refused_cod.groupby(df.customer_id).cumsum() - refused_cod
    df["prior_delivered"] = df.prior_orders - df.prior_rto
    df["prior_rto_rate"] = np.where(df.prior_orders > 0, df.prior_rto / df.prior_orders.clip(lower=1), np.nan)
    prev_date = g.order_date.shift(1)
    first_date = g.order_date.transform("min")
    df["days_since_last_order"] = (df.order_date - prev_date).dt.days
    df["customer_tenure_days"] = (df.order_date - first_date).dt.days
    # pincode historical RTO rate, smoothed toward the network average (k = 30 orders)
    gp = df.groupby("pincode", sort=False)
    prior_n = gp.cumcount()
    prior_r = gp.rto.cumsum() - df.rto
    df["pincode_rto_rate"] = np.round((prior_r + 30 * TARGETS["overall_rto"]) / (prior_n + 30), 4)
    df["value_to_pincode_aov"] = np.round(df.order_value / df.pincode_avg_order_value, 2)
    return df


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    d = generate()
    print(len(d), "orders", d.customer_id.nunique(), "customers", f"{time.time()-t0:.1f}s")
    print("RTO", d.rto.mean().round(4), "COD share", (d.payment_mode == "COD").mean().round(3))
    print(d.groupby("payment_mode").rto.mean().round(4))
    print(d[d.rto == 1].rto_cause.value_counts(normalize=True).round(3))
    print("habitual share of customers", d.groupby("customer_id")._habitual_refuser.first().mean().round(3),
          "share of RTO from habitual", d[d.rto == 1]._habitual_refuser.mean().round(3))
    print(d.groupby(pd.cut(d.prior_rto_rate, [-0.01, 0, .25, .5, 1.01])).rto.agg(["mean", "count"]))
    print(d.groupby(d.prior_orders.clip(upper=10)).rto.agg(["mean", "count"]))
    if len(sys.argv) > 1:
        d.to_csv(sys.argv[1], index=False, compression="gzip")
