# 🧠 Customer Segmentation Project

## 🎯 Project Overview
K-Means clustering on **500 customers** to identify **4 behavioral segments** using Python and scikit-learn.

**Internship Task 2** | Due: 15 Jun 2026

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `customer_segmentation.py` | Full Python script – data, clustering, visualizations |
| `customer_data.csv` | Raw dataset (500 customers, 9 features) |
| `customer_segmented.csv` | Output with Cluster + Segment labels |
| `Customer_Segmentation_Report.pdf` | Full analysis report with charts |
| `fig1_elbow.png` | Elbow + Silhouette chart |
| `fig2_pca.png` | PCA cluster scatter plot |
| `fig3_profiles.png` | Segment feature profiles |
| `fig4_scatter.png` | Income vs Spend & Age vs Frequency |
| `fig5_distribution.png` | Segment distribution by Region & Category |

---

## 🔍 Dataset Features

| Feature | Description |
|---------|-------------|
| Age | Customer age (18–75) |
| Annual_Income | Yearly income in ₹ |
| Monthly_Spend | Average monthly spend in ₹ |
| Purchase_Freq | Purchases per month |
| Tenure_Years | Years as a customer |
| Gender | Male / Female |
| Region | North / South / East / West |
| Category_Pref | Preferred shopping category |

---

## 🤖 Model: K-Means Clustering

- **Algorithm:** K-Means (scikit-learn)
- **Optimal k:** 4 (Elbow Method + Silhouette Score)
- **Silhouette Score:** 0.393
- **Preprocessing:** StandardScaler normalization
- **Visualization:** PCA (2 components)

---

## 👥 Customer Segments

| Segment | Count | Avg Age | Avg Income | Avg Spend | Strategy |
|---------|-------|---------|------------|-----------|----------|
| 💼 Budget Shoppers | 122 | 24 | ₹29,922 | ₹21,800 | Discounts, EMI, bundles |
| 🛍 Occasional Buyers | 142 | 34 | ₹54,584 | ₹44,600 | Flash sales, loyalty points |
| ⭐ Regular Customers | 137 | 45 | ₹76,372 | ₹65,000 | Membership perks, early access |
| 👑 Premium Loyalists | 99 | 57 | ₹95,353 | ₹85,300 | VIP program, concierge service |

---

## 🛠️ Tools & Libraries

```
Python 3.x
pandas, numpy
scikit-learn (KMeans, PCA, StandardScaler, silhouette_score)
matplotlib, seaborn
```

## ▶️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python customer_segmentation.py
```

---

## 📌 Submission
- **Task:** Customer Segmentation Project
- **Internship Task 2** | ✅ Completed
