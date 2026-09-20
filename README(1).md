# Personalized Offer Ranking Model

A machine learning system for ranking personalized offers by the probability that a customer will click on them. The goal is to place the most relevant offers in the top positions and improve user engagement.

## Problem Statement

Given a customer and an offer that has been shown to them, predict the probability that the customer will click the offer.

The model then ranks the available offers by predicted click probability so that the most relevant offers appear at the top.

The primary evaluation metric is **MAP@7 (Mean Average Precision at 7)**, which rewards correctly ranked clicked offers more highly when they appear near the top of the list.

## Dataset

The dataset contains customer, offer, interaction, and transaction information.

### Main Data

- **366 input features** across different customer, offer, and interaction categories
- Binary target:
  - `1` → customer clicked the offer after seeing it
  - `0` → customer did not click the offer after seeing it

### Additional Data Sources

Three supporting datasets are available for feature engineering:

- **Event data** — customer ID, offer ID, placement ID, impression timestamp, click timestamp
- **Transaction data** — transaction amount, product ID, transaction date/time, industry information
- **Offer metadata** — offer ID, offer name, redemption frequency, discount rate, industry, brand, offer description, start/end timestamps

Identifier variables are excluded from modeling.

## Approach

### 1. Data Exploration

- Examined customer, offer, event, and transaction data
- Analyzed click behavior and class imbalance
- Identified useful behavioral and transactional signals

### 2. Feature Engineering

Created features capturing customer behavior and offer context, including:

- Time between offers
- Offer fatigue
- Historical click-through rate
- Customer spending behavior
- Offer/category preferences
- Time-of-day behavior
- Text-based features using TF-IDF
- Transaction and event-based features

### 3. Feature Selection

Used **SHAP** to identify the most informative features and reduce the feature set from the original 366 variables to a smaller set of high-value features.

### 4. Model

Used **LightGBM** to estimate the click probability for each customer-offer pair.

To handle class imbalance, negative samples were downsampled using a **5:1 negative-sampling strategy**.

### 5. Validation

Used **customer-grouped 5-fold cross-validation** so that data from the same customer does not leak across training and validation folds.

### 6. Ranking

For each customer, offers were ranked by predicted click probability and the top 7 offers were selected for evaluation.

## Results

| Experiment | MAP@7 |
|---|---:|
| Initial baseline | 0.180 |
| After feature engineering | 0.444 |
| Final ranking model | **0.531** |

The final model improved MAP@7 from **0.180 to 0.531**, a relative improvement of approximately **195%**.

Feature engineering alone improved MAP@7 from **0.180 to 0.444**, accounting for a substantial portion of the overall improvement.

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **LightGBM**
- **Scikit-learn**
- **SHAP**
- **TF-IDF**
- **Jupyter Notebook**

## Project Pipeline

```text
Customer + Offer + Event + Transaction Data
                    |
                    v
             Data Exploration
                    |
                    v
           Feature Engineering
                    |
                    v
             Feature Selection
                  (SHAP)
                    |
                    v
                LightGBM
                    |
                    v
          Click Probability Score
                    |
                    v
       Rank Offers for Each Customer
                    |
                    v
                 Top 7
                    |
                    v
                MAP@7
```

## Key Takeaways

- Behavioral signals such as offer timing, category preference, and time of day were more informative than many demographic features.
- Feature engineering played a major role in improving ranking quality.
- Customer-grouped validation helped reduce the risk of data leakage.
- The final system focuses on **ranking the most relevant offers**, rather than only predicting individual click probabilities.

## Future Improvements

- Try learning-to-rank objectives such as LambdaMART
- Add richer sequential customer behavior features
- Improve text embeddings for offer descriptions
- Explore personalized models for different customer segments
- Evaluate ranking stability across different time periods
