# Customer Churn Prediction

This project simulates a machine learning pipeline to predict customer churn using synthetic data. It reflects a real-world use case frequently encountered in telecom, SaaS, and subscription-based businesses.

As a Senior Business Analyst, the objective is to collaborate with Data Scientists and stakeholders to ensure data is preprocessed, modeled, evaluated, and ready for insights and business decision-making.

---

## 💼 Business Objective

To identify customers who are likely to churn in order to enable early intervention and improve customer retention strategies.

Customer churn represents a critical risk to recurring revenue models, particularly in telecom, SaaS, and financial services. By anticipating churn behavior through data-driven modeling, companies can proactively deploy retention campaigns, optimize customer lifetime value (CLV), and reduce acquisition costs.

This project simulates a churn prediction framework from scratch, ensuring that stakeholders across marketing, product, and operations can leverage predictive insights to take timely and targeted actions. It also demonstrates the Business Analyst’s role in translating raw data into measurable business outcomes.

---

## 🔍 How It Works

1. **Data Generation**
   - A synthetic dataset of 100,000 customer records is generated with demographic and behavioral features, including account age, service usage, spend, and support interactions.

2. **Preprocessing & Encoding**
   - Irrelevant identifiers (e.g., CustomerID) are removed.
   - Categorical values are encoded, and data is prepared for modeling.

3. **Model Training**
   - A logistic regression model is trained using `scikit-learn`.
   - Class imbalance is addressed using `class_weight='balanced'` to ensure churn class is recognized.

4. **Model Evaluation**
   - Performance is assessed through confusion matrix and classification report metrics (Accuracy, Precision, Recall, F1-score).

---

## 📈 Model Performance

| Metric     | Value (Example) |
|------------|-----------------|
| Accuracy   | 0.84            |
| Precision  | 0.78            |
| Recall     | 0.67            |
| F1-score   | 0.72            |

> *Note: Confusion matrix and full classification report are available in the `/data/` directory.*

---

## 🛠️ Technologies Used

- Python 3.12
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Modular scripting with PyCharm
- Linux terminal for file management and repository control

---


---

## 💡 Business Impact

- 🔎 **Customer Retention**  
  Enables early detection of at-risk customers for proactive retention strategies.

- 💰 **Revenue Protection**  
  Reduces churn-driven revenue loss and protects recurring business models.

- 🎯 **Operational Efficiency**  
  Improves targeting for sales and customer success teams, prioritizing high-risk segments.

- 🧠 **Strategic Planning**  
  Provides insight into churn-driving behaviors to inform long-term product and loyalty initiatives.

---

## 👤 Author

**DEVANSHI GULATI** 
Student 
[GitHub Profile](https://github.com/DevanshiGulati)


