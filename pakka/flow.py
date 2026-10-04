"""Clickable WhatsApp / IVR confirmation flow, driven by st.session_state."""
from __future__ import annotations
import streamlit as st

MSG_COST, IVR_COST = 0.13, 0.25
LADDER = [("At checkout", 0), ("Before pickup", 0), ("In transit", 11), ("1st attempt", 21), ("Marked RTO", 170)]

CSS = """
<style>
.phone{max-width:380px;margin:0 auto;border:10px solid #1C2026;border-radius:30px;overflow:hidden;background:#E8E2D8}
.phead{background:#1E3A34;color:#EAF3EF;padding:10px 14px;font-size:14px}
.phead small{opacity:.8;font-size:11.5px;display:block}
.chat{padding:12px 10px;display:flex;flex-direction:column;gap:8px;min-height:360px}
.bub{max-width:86%;border-radius:10px;padding:8px 10px;font-size:13.5px;line-height:1.45;color:#1B1F24;box-shadow:0 1px 1px rgba(0,0,0,.08)}
.bub.in{background:#fff;align-self:flex-start;border-top-left-radius:2px}
.bub.out{background:#D7F2C6;align-self:flex-end;border-top-right-radius:2px}
.bub .tag{font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;opacity:.6;display:block;margin-bottom:3px}
.bub .tm{display:block;text-align:right;font-size:10.5px;opacity:.55;margin-top:2px}
.sys{align-self:center;font-size:11.5px;background:rgba(0,0,0,.08);padding:3px 10px;border-radius:6px;color:#1B1F24}
.ladder{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}
.ladder div{border-radius:6px;padding:8px 4px;text-align:center;font-size:11.5px;border:2px solid transparent;background:rgba(128,128,128,.12)}
.ladder div b{display:block;font-size:15px}
.ladder div.hit{border-color:#1F5F8B;background:rgba(31,95,139,.15)}
.st-key-flowbtns{margin-top:22px;padding:14px;border-radius:14px;background:#FFFFFF;border:1px solid #DCE0D8}
.tapnote{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#5B6572;font-weight:700;margin-bottom:2px}
.st-key-flowbtns button{min-height:54px;padding:6px 10px;border-radius:12px;border:2px solid #1E3A34;background:#E9F6EF;color:#1E3A34;box-shadow:0 2px 6px rgba(30,58,52,.12)}
.st-key-flowbtns button p{font-size:15px !important;font-weight:700 !important;white-space:normal !important;line-height:1.25}
.st-key-flowbtns button:hover{background:#1E3A34;color:#FFFFFF;border-color:#1E3A34}
.st-key-flowbtns button:hover p{color:#FFFFFF}
</style>
"""


def _state():
    if "flow" not in st.session_state:
        st.session_state.flow = None
    return st.session_state


def reset(ctx: dict, tier: str):
    st.session_state.flow = {"tier": tier, "t": 0, "msgs": [], "log": [], "chips": [], "stage": 1,
                             "title": "", "body": "", "step": "start", "ctx": ctx}
    f = st.session_state.flow
    _sys(f, "Today · order placed 11:40")
    c = ctx
    if tier == "A":
        _in(f, f"Namaste {c['name']} ji,<br>Aapka order <b>{c['id']}</b> dispatch se pehle confirm karna hai.<br><br>"
               f"{c['item']}<br>₹{c['value']:,} · {'Cash on delivery' if c['cod'] else 'Prepaid'}<br>Address: {c['addr']}<br><br>"
               "Kya hum ise kal bhej dein?", "Order confirmation")
        _log(f, "WhatsApp utility template sent (3 quick-reply buttons)", MSG_COST)
        _out_state(f, "Waiting for the customer", "The seller has not packed yet, so any answer is still free to act on.", 1)
        f["step"] = "ask"
    else:
        extra = f"<br>Abhi UPI se pay karein aur ₹{c['cashback']} wapas paayein." if c["cod"] and c["cashback"] else ""
        _in(f, f"Namaste {c['name']} ji,<br>Aapka order <b>{c['id']}</b> ({c['item']}, ₹{c['value']:,}) kal dispatch hoga."
               f"{'<br>Delivery par cash ya UPI ready rakhiye.' if c['cod'] else ''}<br><br>Plan badal gaya? Neeche se free mein cancel karein.{extra}",
            "Dispatch reminder")
        _log(f, "WhatsApp reminder sent (link buttons, no reply needed)", MSG_COST)
        _out_state(f, "Reminder sent", "The soft tier needs no reply. The order ships unless the customer acts.", 1)
        f["step"] = "remind"


def _clock(f):
    m = 11 * 60 + 42 + f["t"]
    return f"{(m // 60) % 24:02d}:{m % 60:02d}"


def _in(f, html, tag=None):
    f["msgs"].append(("in", html, tag, _clock(f)))


def _outb(f, html):
    f["msgs"].append(("out", html, None, _clock(f)))


def _sys(f, text):
    f["msgs"].append(("sys", text, None, None))


def _log(f, ev, cost):
    f["log"].append((_clock(f), ev, cost))


def _out_state(f, title, body, stage=None):
    f["title"], f["body"] = title, body
    if stage is not None:
        f["stage"] = stage


def render(ctx: dict, default_tier: str):
    s = _state()
    st.markdown(CSS, unsafe_allow_html=True)
    top = st.columns([1, 1, 2])
    tier = top[0].radio("Message", ["A", "B"], format_func=lambda x: "Pre-confirmation (interactive)" if x == "A" else "Soft reminder",
                        index=0 if default_tier != "B" else 1, key="flow_tier", horizontal=False)
    if top[1].button("Restart flow", width="stretch") or s.flow is None or s.flow["tier"] != tier or s.flow["ctx"]["id"] != ctx["id"] or s.flow["ctx"]["value"] != ctx["value"] or s.flow["ctx"]["cod"] != ctx["cod"]:
        reset(ctx, tier)
    f = s.flow
    top[2].caption(f"Order {ctx['id']} · score **{ctx['points']:.0f}** · P(RTO) **{ctx['p']:.1%}** · {ctx['tier_label']}. "
                   "Change the order in the Score tab and it carries over here.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        html = ['<div class="phone"><div class="phead"><b>Valmo Delivery</b><small>Business account · concept mock</small></div><div class="chat">']
        for kind, body, tag, tm in f["msgs"]:
            if kind == "sys":
                html.append(f'<div class="sys">{body}</div>')
            else:
                html.append(f'<div class="bub {kind}">{f"<span class=tag>{tag}</span>" if tag else ""}{body}<span class="tm">{tm}{" ✓✓" if kind == "out" else ""}</span></div>')
        html.append("</div></div>")
        st.markdown("".join(html), unsafe_allow_html=True)
        with st.container(key="flowbtns"):
            st.markdown('<div class="tapnote">Tap what the customer does</div>', unsafe_allow_html=True)
            _buttons(f)
    with right:
        with st.container(border=True):
            st.markdown(f"#### {f['title']}")
            st.write(f["body"])
            if f["chips"]:
                st.markdown(" · ".join(f"**{c}**" for c in f["chips"]))
            st.markdown("**When the order stops: cost to Valmo**")
            st.markdown('<div class="ladder">' + "".join(
                f'<div class="{"hit" if i == f["stage"] else ""}"><b>₹{c}</b>{l}</div>' for i, (l, c) in enumerate(LADDER)) + "</div>",
                unsafe_allow_html=True)
            st.caption("A cancel before pickup costs ₹0. The same decision at the door costs ₹170.")
        with st.container(border=True):
            st.markdown("**Event log**")
            tot = sum(c for _, _, c in f["log"] if isinstance(c, (int, float)))
            rows = [{"Time": t, "Event": e, "Cost (₹)": c} for t, e, c in f["log"]]
            rows.append({"Time": "", "Event": "Spend so far", "Cost (₹)": round(tot, 2)})
            st.dataframe(rows, hide_index=True, width="stretch")


def _buttons(f):
    c = f["ctx"]
    step = f["step"]
    k = f"{step}-{len(f['msgs'])}"
    if step == "ask":
        cols = st.columns(3)
        if cols[0].button("Haan, bhej do", key="a1" + k, width="stretch"):
            f["t"] += 3; _outb(f, "Haan, bhej do"); _in(f, "Shukriya! Aapka order kal dispatch hoga.")
            _log(f, "Customer confirmed", 0); f["chips"].append("Confirmed")
            _out_state(f, "Confirmed before pickup", "Each confirmation is a labelled outcome. The pilot measures how often confirmed COD orders still fail.", 1)
            _offer(f); st.rerun()
        if cols[1].button("Fix address", key="a2" + k, width="stretch"):
            f["t"] += 2; _outb(f, "Address theek karna hai"); f["step"] = "fix"
            _in(f, "Ghar / flat number aur gali bhejiye, aur location share kijiye.", "Address update")
            _log(f, "Address form opened (WhatsApp Flow)", 0)
            _out_state(f, "Fixing the address", "The customer adds a house number and shares a live location pin.", 1); st.rerun()
        if cols[2].button("Cancel karo", key="a3" + k, width="stretch"):
            f["t"] += 2; _outb(f, "Order cancel karo"); f["step"] = "reason"
            _in(f, "Koi baat nahi. Cancel karne ki wajah batayenge? Isse humein madad milti hai.", "List message"); st.rerun()
        if st.button("Customer doesn't reply for 6 hours", key="nr" + k):
            f["t"] += 360; _sys(f, "6 hours later · no reply"); _log(f, "IVR call placed", IVR_COST); f["step"] = "ivr"
            _out_state(f, "IVR fallback", "An automated call asks the same three questions: press 1 to confirm, 2 to change address, 3 to cancel.", 1); st.rerun()
    elif step == "fix":
        h = st.text_input("Ghar / flat number", "H.No. 117/42, 2nd floor", key="fh")
        g = st.text_input("Gali / mohalla", "Gali 3, near Hanuman Mandir", key="fg")
        if st.button("Share location and save", key="fs" + k, width="stretch"):
            f["t"] += 2; _outb(f, f"<b>Location shared</b><br>26.4861° N, 80.2718° E<br>{h}, {g}")
            _in(f, "Address update ho gaya. Parcel sahi hub se niklega.")
            _log(f, "Map pin + house number saved before label print", 0); f["chips"] += ["Address fixed", "Pin captured for Valmo Pata"]
            _out_state(f, "Address fixed before pickup", "Re-routed before the label prints. The pin also becomes a verified coordinate for this phone number.", 1)
            _offer(f); st.rerun()
    elif step == "reason":
        reasons = ["Abhi paise nahi hain", "Galti se order ho gaya", "Kahin aur se le liya", "Delivery bahut late hai", "Size / design pasand nahi"]
        cols = st.columns(2)
        for i, r in enumerate(reasons):
            if cols[i % 2].button(r, key=f"r{i}" + k, width="stretch"):
                f["t"] += 1; _outb(f, r)
                _in(f, f"Order {c['id']} cancel ho gaya. {'Koi charge nahi.' if c['cod'] else 'Refund 2-3 din mein.'}")
                _log(f, f'Cancelled before pickup · reason code "{r}"', 0); f["chips"] += ["Cancelled at ₹0", "Reason code captured"]
                _out_state(f, "Cancelled before pickup: ₹0",
                           f"No pickup, no forward trip, no reverse trip. On this order's risk the expected RTO loss avoided is ₹{c['p'] * 170:.0f} "
                           f"({c['p']:.1%} × ₹170). The sale is lost too, which is why the success metric is cost per delivered order, not cancellation rate.", 1)
                f["step"] = "done"; st.rerun()
    elif step == "offer":
        cols = st.columns(3)
        if c["cashback"] and cols[0].button(f"UPI se ₹{c['value']:,} pay karo", key="o1" + k, width="stretch"):
            _pay_upi(f); st.rerun()
        if c["token"] and cols[1].button(f"₹{c['token']} abhi, baaki COD", key="o2" + k, width="stretch"):
            f["t"] += 2; _outb(f, f"UPI payment · ₹{c['token']} (token)"); _in(f, f"₹{c['token']} mil gaya. Baaki ₹{c['value'] - c['token']:,} delivery par dijiye.")
            _log(f, "Partial-COD token paid by UPI", 0); f["chips"].append(f"₹{c['token']} token paid")
            _out_state(f, "Confirmed with a partial-COD token", f"The customer now has ₹{c['token']} at stake. If they still refuse, the token covers most or all of the ₹50 forward trip.", 1)
            f["step"] = "done"; st.rerun()
        if cols[2].button("Poora COD theek hai", key="o3" + k, width="stretch"):
            _outb(f, "Poora COD theek hai"); _log(f, "Ships as COD next day", 0)
            _out_state(f, "Confirmed, ships as COD", f"Model RTO chance before the confirmation effect: {c['p']:.1%}. The pilot measures how much confirming lowers it.", 1)
            f["step"] = "done"; st.rerun()
    elif step == "remind":
        cols = st.columns(2)
        if c["cod"] and c["cashback"] and cols[0].button(f"UPI se pay karo (₹{c['cashback']} wapas)", key="b1" + k, width="stretch"):
            _pay_upi(f); st.rerun()
        if cols[1].button("Order cancel karo", key="b2" + k, width="stretch"):
            f["t"] += 2; _outb(f, "Order cancel karo"); f["step"] = "reason"
            _in(f, "Koi baat nahi. Cancel karne ki wajah batayenge?", "List message"); st.rerun()
        if st.button("Customer doesn't reply", key="b3" + k):
            _sys(f, "No reply"); _log(f, "No reply. Ships as normal", 0)
            _out_state(f, "No reply: ships as normal", "The soft tier never waits on the customer. The reminder alone is the intervention.", 1)
            f["step"] = "done"; st.rerun()
    elif step == "ivr":
        st.info('Automated call: "Aapke order ke liye 1 dabaiye confirm karne ke liye, 2 address badalne ke liye, 3 cancel karne ke liye."')
        cols = st.columns(4)
        if cols[0].button("Press 1", key="i1" + k, width="stretch"):
            _sys(f, "Confirmed on call (pressed 1)"); _log(f, "Confirmed via IVR", 0); f["chips"].append("Confirmed via IVR")
            _out_state(f, "Confirmed on the call", "Ships as normal. Payment offers are skipped on IVR to keep the call short.", 1); f["step"] = "done"; st.rerun()
        if cols[1].button("Press 2", key="i2" + k, width="stretch"):
            _sys(f, "Pressed 2 · location link sent by SMS"); _log(f, "Location-share link sent by SMS", 0.10); f["chips"].append("Address fix pending")
            _out_state(f, "Address link sent", "An SMS link collects the pin. The order is held until the address is fixed, or 12 hours at most.", 1); f["step"] = "done"; st.rerun()
        if cols[2].button("Press 3", key="i3" + k, width="stretch"):
            _sys(f, "Cancelled on call (pressed 3)"); _log(f, "Cancelled via IVR · no reason code", 0); f["chips"].append("Cancelled at ₹0")
            _out_state(f, "Cancelled before pickup: ₹0", "Same ₹0 outcome as a WhatsApp cancel, but no reason code. One reason WhatsApp goes first.", 1); f["step"] = "done"; st.rerun()
        if cols[3].button("No answer", key="i4" + k, width="stretch"):
            _sys(f, "Call not answered"); _log(f, "No answer. Ships flagged Unconfirmed; rider calls first", 0); f["chips"].append("Unconfirmed")
            _out_state(f, "Ships as Unconfirmed", f"The riskiest bucket there is. It ships, the rider calls before the attempt, and the pilot reads it separately. Model RTO chance {c['p']:.1%}+.", 3)
            f["step"] = "done"; st.rerun()
    else:
        st.caption("Flow finished. Press Restart flow to try another path.")


def _offer(f):
    c = f["ctx"]
    if not c["cod"] or not (c["cashback"] or c["token"]):
        _log(f, "Ships as normal next day", 0)
        f["step"] = "done"
        return
    f["t"] += 1
    lines = []
    if c["cashback"]:
        lines.append(f"Abhi UPI se pay karein aur <b>₹{c['cashback']} wapas</b> paayein.")
    if c["token"]:
        lines.append(f"Ya sirf <b>₹{c['token']}</b> abhi dein, baaki ₹{c['value'] - c['token']:,} delivery par.")
    _in(f, "<br>".join(lines), "Payment option")
    _log(f, "Prepaid offer shown in the same thread", 0)
    f["step"] = "offer"


def _pay_upi(f):
    c = f["ctx"]
    f["t"] += 2
    _outb(f, f"UPI payment · ₹{c['value']:,}")
    _in(f, f"Payment mil gaya. ₹{c['cashback']} cashback 24 ghante mein.")
    _log(f, "Paid by UPI · cashback", c["cashback"]); f["chips"].append("Converted to prepaid")
    save = (c["p"] - c["p_prepaid"]) * 170
    _out_state(f, "Converted to prepaid", f"Model RTO chance falls from {c['p']:.1%} to {c['p_prepaid']:.1%}. Expected saving ₹{save:.0f} against ₹{c['cashback']} cashback.", 1)
    f["step"] = "done"
