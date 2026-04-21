# UCB-ML-AI-Capstone-Loan-Default-Risk-Prediction
Final Capstone Project for Module 24

### Project Title : Loan Default Risk Prediction Using Machine Learning


#### Executive summary

This project develops a predictive machine learning framework to identify high-risk borrowers and mitigate financial losses for lending institutions. Analyzing a dataset of over 148K records, the project evolved from a linear baseline to a high-performance **XGBoost** champion model. By implementing a leak-proof preprocessing pipeline and strategic threshold optimization (0.321), the final model achieved a **Recall of 80.01%** (a 22.2% relative improvement over the baseline). The framework provides actionable insights via **SHAP explainability**, identifying key risk drivers such as Debt-to-Income (DTI), Property Value, and Negative Amortization.


#### Rationale

Loan defaults are one of the most significant costs for financial institutions. Even a small reduction in the default rate can save a large bank millions of dollars. Accurate prediction models help lenders:
* Reduce financial losses and preserve capital.
* Improve credit approval strategies and operational efficiency.
* Ensure regulatory compliance through transparent decision-making.
* Promote responsible lending by identifying early warning signs of borrower distress.


#### Research Question

Can we accurately predict the likelihood of a borrower defaulting on a loan by leveraging non-linear relationships between financial characteristics, loan-specific attributes, and structural risk factors?


#### Data Sources

The dataset used in this project is sourced from Kaggle:
[Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset)

This dataset contains approximately 148,000 records with features including:
- Borrower financial information (income, debt-to-income ratio)
- Credit attributes (credit score, credit type)
- Loan details (loan amount, interest rate, loan purpose)
- Target variable: Status ( 0 for No Default, 1 for Default).


#### Methodology

* **Leak-Proof Preprocessing:** Implemented a robust Scikit-Learn Pipeline and ColumnTransformer to ensure all imputation and scaling (StandardScaler) happened post-split, eliminating data leakage.
* **Advanced Imputation:** Utilized IterativeImputer (BayesianRidge) for complex missing-at-random (MAR) features like Property Value and DTI, forcing the model to learn from substantive data rather than procedural artifacts.
* **Feature Engineering:** Engineered `loan_to_income` (log-scaled loan burden) and `ltv_risk` (domain-knowledge mortgage underwriting tiers) to capture non-linear loan burdens. Applied log transformations to normalise skewed financial distributions.
* **Architectural Progression:** Evaluated models in three stages:
    1. Baseline: Logistic Regression with class_weight='balanced'.
    2. Advanced Ensembles & Neural Networks: Implementation of Random Forest, MLP (Multi-Layer Perceptron), XGBoost, and LightGBM to capture complex, non-linear relationships.
    3. Optimization: Hyperparameter tuning via RandomizedSearchCV with a primary focus on Recall.
* **Threshold Optimization:** Tuned the decision boundary from 0.50 to 0.321 to maximize risk capture in a 1:3 imbalanced environment.


#### Results

* **Champion Model Performance:** The tuned XGBoost model emerged as the champion, achieving 80.01% Recall and a PR-AUC of 0.832.
* **Risk Mitigation Impact:** Through threshold tuning (0.321), the model achieved a 90.67% capture rate, identifying 781 additional defaulters compared to the standard model—a 53.3% reduction in missed defaults (False Negatives).
* **Key Risk Drivers (SHAP):** Identified Property Value, Debt-to-Income (DTI), and Loan-to-Income as the primary drivers of risk. Negative Amortization is confirmed as a top-tier structural risk signal.
* **Operational Efficiency:** XGBoost proved to be the most scalable solution, training 5x faster than the MLP and LightGBM tuned alternatives while providing superior risk detection.
* **Model Specialization:** The MLP Classifier achieved the highest Precision (91.04%), making it ideal for a conservative strategy focused on minimizing "false alarms." However, XGBoost is selected as the final champion because its superior Recall (80.01%) better serves the project's primary goal of maximizing risk capture.


#### Next steps

* **Cost-Sensitive Learning:** Integrate a business-defined cost matrix to further refine the financial trade-off between False Negatives and False Positives.
* **Model Monitoring:** Implement a real-time scoring API to monitor performance shifts during changing macroeconomic cycles.
* **Advanced Embeddings:** Explore deep learning architectures with entity embeddings for high-cardinality categorical features.
* **Operational Pilot:** Launch an A/B test in a controlled lending environment to measure the impact on real-world approval rates.


#### Project Structure

```
UCB-ML-AI-Capstone-Loan-Default-Risk-Prediction/
├── CAPSTONE_Loan_Default_Prediction.ipynb    # Main notebook
│   ├── A) Data Understanding (EDA)
│   ├── B) Data Preparation (leak-proof pipeline)
│   ├── C) Modeling (3-step progression)
│   ├── D) Evaluation (performance comparison)
│   ├── E) Threshold Optimization (tuning)
│   └── F) SHAP Analysis & Conclusion
├── Capstone Project_Final Report.docx
├── README.md            
└── data/
    └── Loan_Default.csv                      # Original dataset (148,670 rows, 34 columns)
└── images/                                   # Plotly images
    └── loan_amount_vs_income.html           
    └── loan_probability_distribution.html
```

#### Technologies Used

- **Core ML:** Python, scikit-learn, pandas, NumPy
- **Modelling:** XGBoost, LightGBM, Random Forest, MLP (Neural Networks)
- **Advanced:** SHAP (feature importance), IterativeImputer (MICE imputation), imbalanced-learn (SMOTE)
- **Visualisation:** Matplotlib, Seaborn, Plotly
- **Evaluation:** Precision-Recall curves, ROC-AUC, F1-score, Confusion Matrices

#### Project links

- [Capstone Juyter Notebook](https://github.com/kumes121/UCB-ML-AI-Capstone-Loan-Default-Risk-Prediction/blob/main/CAPSTONE_Loan_Default_Prediction.ipynb)
- [Plotly images](https://github.com/kumes121/UCB-ML-AI-Capstone-Loan-Default-Risk-Prediction/tree/main/images)
- [Loan Default Dataset](https://github.com/kumes121/UCB-ML-AI-Capstone-Loan-Default-Risk-Prediction/tree/main/data)
