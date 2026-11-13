# Model Evaluation & Performance Comparison Table

| model                |   hits |   total_customers |   hit_rate_at_5 |   hit_rate_pct |   standard_error |   ci_95_lower_pct |   ci_95_upper_pct |   ci_margin_pct |   improvement_over_level1_pct | status                           |
|:---------------------|-------:|------------------:|----------------:|---------------:|-----------------:|------------------:|------------------:|----------------:|------------------------------:|:---------------------------------|
| Level 0 - Random     |    146 |               781 |        0.18694  |          18.69 |           0.014  |             16.12 |             21.58 |            2.89 |                        -60.86 | -60.86% below Level 1            |
| Level 1 - Popularity |    373 |               781 |        0.477593 |          47.76 |           0.0179 |             44.28 |             51.26 |            3.51 |                          0    | Reference Baseline (Level 1)     |
| Logistic Regression  |    586 |               781 |        0.75032  |          75.03 |           0.0155 |             71.88 |             77.94 |            2.91 |                         57.1  | +57.10% improvement over Level 1 |
| Decision Tree        |    594 |               781 |        0.760563 |          76.06 |           0.0153 |             72.94 |             78.92 |            2.86 |                         59.25 | +59.25% improvement over Level 1 |
