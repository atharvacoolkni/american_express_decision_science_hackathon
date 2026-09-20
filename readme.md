# Offers Personalization: Click Probability Prediction

A machine learning project that predicts the probability a customer will click on a digital offer, given that they have seen it. The predictions are used to rank offers so the most relevant ones appear at the top, increasing clicks and overall engagement.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Digital Offers Ecosystem](#digital-offers-ecosystem)
- [Target Variable](#target-variable)
- [Data](#data)
- [Features](#features)
- [Evaluation Metric](#evaluation-metric)
- [Approach](#approach)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Output Format](#output-format)
- [Guidelines and Constraints](#guidelines-and-constraints)

---

## Overview

Card issuers show customers a variety of offers (cashback, bonus reward points, extra miles, and similar) across channels such as the website, mobile app and email. With many offers available, choosing which ones to show first has a large effect on engagement.

This project builds a predictive model that scores each customer-offer pair with a click probability and ranks offers accordingly.

**Example offers:**

| Offer type | Example |
|---|---|
| Shopping | +5 reward points per eligible dollar spent, up to 2,500 points |
| Shopping | Spend $50 or more, get $10 back |
| Travel | Earn 3 additional miles per dollar spent |
| Dining | Get 10% back on purchases, up to $5 |

## Problem Statement

- Predict the probability that a customer clicks on a given offer on a particular day, **given that the customer has seen the offer**.
- The end objective is to place the most relevant offers in the top ranks, which increases clicks and overall engagement.

## Digital Offers Ecosystem

The personalization pipeline has three stages:

1. **Offer setup and eligibility**: targeting based on customer segments (customer profile, card product, geography, spend behavior).
2. **Arbitration**: a personalization decision engine that ranks a variety of content using behavioral patterns to present the best offer. *This project focuses on this stage.*
3. **Delivery**: the engine serves offers across channels and placements (website, mobile app, email).

## Target Variable

A binary outcome:

- `1` if the customer clicked on the offer
- `0` otherwise

This is conditioned on the customer having seen the offer, so the model estimates **P(Click | Impression)**.

The interaction funnel is:

1. **Extension**: the placement and offer loaded successfully on the page.
2. **Impression**: the customer scrolled over the defined area, confirming the offer was seen.
3. **Click**: the customer clicked the call-to-action button.

## Data

| File | Description |
|---|---|
| `train_data.parquet` | Training data with features and the binary click label |
| `test_data.parquet` | Test data to score |
| `add_event.parquet` | Events data (impressions and clicks) |
| `add_trans.parquet` | Customer transaction data |
| `offer_metadata.parquet` | Offer attributes (discount rate, industry, brand, dates) |
| `data_dictionary.csv` | Descriptions of all variables |

> The data is not included in this repository. Variable names are masked and no personally identifiable information (PII) is used.

### Additional datasets

**Events data**

| Column | Description |
|---|---|
| `id2` | Customer ID (masked) |
| `id3` | Offer ID |
| `id6` | Placement ID |
| `id4` | Impression timestamp |
| `id7` | Click timestamp |

**Transaction data**

| Column | Description |
|---|---|
| `id2` | Customer ID (masked) |
| `f367` | Transaction amount |
| `f368` | Product ID |
| `f369` | Transaction debit/credit |
| `f370` | Transaction date |
| `f371` | Time of transaction |
| `f372` | Year-month of transaction |
| `f374` | Industry description |
| `id8` | Industry code |

**Offer metadata**

| Column | Description |
|---|---|
| `id3` | Offer ID |
| `id9` | Offer name |
| `f375` | Redemption frequency |
| `f376` | Discount rate |
| `f377` | Random |
| `id10` | Industry code |
| `id11` | Brand name |
| `f378` | Offer body |
| `f374` | Industry name |
| `id8` | Industry code |
| `id12` | Start timestamp |
| `id13` | End timestamp |

## Features

The dataset has **366 features** across four categories:

| Category | Examples |
|---|---|
| **Customer profile and card usage** | Product, spend patterns, tenure, payment behavior, location |
| **Offer / treatment features** | Popularity, industry, duration and expiry, discount rate |
| **Customer past interactions** | Interaction with the same offer, browsing on web/app, past interaction with similar offers |
| **Others** | Interaction with co-brand partners, engagement with rewards |

Additional engineered features can be built from the events, transactions and offer metadata files.

## Evaluation Metric

**MAP@7 (Mean Average Precision at 7)** measures how well offers are rank ordered.

- True positives are weighted by their rank, so a click captured at rank 1 counts for more than one captured at rank 2.
- Average Precision (AP) is computed for each customer, then averaged over all customers in the test set.
- Evaluation is done on out-of-time data.

## Approach

> Update this section with your own modeling details.

1. **Exploratory data analysis**: understand click rates by offer, placement, industry and customer segment.
2. **Feature engineering**:
   - Customer spend aggregates from transactions (by industry, recency, frequency)
   - Customer-offer affinity from past events (past impressions and clicks on the same or similar offers)
   - Offer attributes (discount rate, duration, days to expiry)
3. **Modeling**: gradient boosting models (e.g. LightGBM, XGBoost, CatBoost) trained on the click label.
4. **Validation**: time-based split to mimic out-of-time scoring.
5. **Ranking**: sort each customer's offers by predicted probability and evaluate with MAP@7.




## Guidelines and Constraints

- Existing variables can be used and new variables derived from them.
- Identifier variables must not be used as model features.
- The provided data must not be altered, and no rows may be added.
- The solution must score every unique identifier in the test data.
- The solution should be scalable for real-world use.

## License

Add your license here (e.g. MIT).
