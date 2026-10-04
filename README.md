# Valmo Pakka · RTO risk scorecard

**Pehle Pakka, Phir Package.** Score every order at checkout. Confirm the risky ones on WhatsApp before pickup, nudge them to prepaid, and make cancelling cheap and early.

Team Prod Gods · IIT Kanpur · Meesho DICE S3 (Business track)

## What's in the app

| Tab | What it shows |
|---|---|
| 1. Score an order | Pick an example or edit an order. Get a 0-100 risk score, what happens next, and the top reasons pushing risk up or down |
| 2. Customer flow | Clickable WhatsApp / IVR flow: Confirm, Fix address, Cancel with reason code, prepaid or partial-COD offer, no-reply fallback |
| 3. How we know it works | Rules vs scorecard vs ML cards, RTOs caught vs orders messaged, calibration chart, health checks, the real-data check, and the dataset |

## Why a scorecard and not ML

The progression is rules, then a statistical scorecard, then ML. On held-out months:

| Model | AUC | RTOs caught by contacting the riskiest 10% |
|---|---|---|
| Rules v0 (hand-set points) | 0.657 | 20% |
| **WoE scorecard (our pick)** | **0.702** | **24%** |
| Gradient boosting | 0.707 | 25% |
| Hidden truth (ceiling) | 0.748 | 32% |

Gradient boosting adds about 0.005 AUC. That isn't worth losing a model where every point can be explained to ops, sellers and customers. Revisit ML once there are real labels and new signals.

## The data is synthetic

The app generates the dataset on start-up (fixed seed 42, so it's identical every time). It has about 2.1 lakh orders from 80,000 customers in 1,200 pincodes over 12 months, calibrated to the case-pack figures:

- 17% RTO overall, 80% COD, 20% COD RTO, 5% prepaid RTO
- RTO cause mix 40 / 28 / 18 / 9 / 5

Every other behaviour is an assumption, listed in pakka/data_gen.py. **No Valmo data is used.**

## Run it on your laptop

    pip install -r requirements.txt
    streamlit run app.py

## Project layout

    app.py                       Streamlit app (3 tabs)
    pakka/data_gen.py            synthetic order generator + assumptions
    pakka/scorecard.py           binning, WoE/IV, logistic scorecard, rules v0, GBM challenger, metrics
    pakka/flow.py                WhatsApp / IVR flow
    data/                        Amazon real-data results
    scripts/amazon_scorecard.py  reproduces the real-data check from the public Amazon CSV
    .streamlit/config.toml       theme
    requirements.txt
