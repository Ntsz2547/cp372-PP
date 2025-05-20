# CP372 Final Project – Bank Target Marketing

This repository contains the final project for CP372: Data Analytics and Business Intelligence. The project focuses on analyzing direct marketing campaign data from a banking institution to generate actionable insights and recommendations for improving campaign performance.

---
## Table of Contents
- [Member](#member)
- [Project Overview](#project-overview)
- [Tools & Technologies](#tools--technologies)
- [Data Preparation](#data-preparation)
  - [1. Data Cleaning](#1-data-cleaning)
  - [2. Feature Engineering](#2-feature-engineering)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [In-Depth Analysis](#in-depth-analysis)
- [Deliverables](#deliverables)

---

##  Member

- ปริชญ์ ผ่องใส (ID: 65102010118)

---

##  Overview

The goal of this project is to explore and analyze customer demographic and campaign data to better understand patterns in marketing campaign success. We aim to provide insights that can help marketing teams improve customer targeting and campaign effectiveness.

We use Python for data processing and Tableau for interactive data visualization.

We use data from :
https://www.kaggle.com/datasets/seanangelonathanael/bank-target-marketing

Tableau Dashboard:
https://public.tableau.com/views/CP372_Final_Project_118/Dashboard?:language=th-TH&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

Google Colab Link:
https://colab.research.google.com/drive/1PU7nOFtGB13nbsXPqoprLXMluRMEvWJS?usp=sharing

---

## Tools & Technologies

- Python (pandas, numpy, matplotlib, seaborn, scikit-learn, etc.)
- Google Colab
- Tableau Public
- GitHub

---

##  Data Preparation

### 1.Import  Important Library and File from KAGGLE
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("seanangelonathanael/bank-target-marketing")

print("Path to dataset files:", path)
```
```python
import pandas as pd
import numpy as np
```
### 2. Data Cleaning
- Checked file shape , Verified correct data types etc.
```python
data.shape
data.columns
data.info()
data.describe()
```
- Checked and confirmed there were no missing values.
```python
data.isnull().sum()
```
- Checked and confirmed there were no duplicated rows if It's have We will remove it.
```python
data.duplicated().sum()
df.drop_duplicates()
```

### 3. Feature Engineering

- Created new features such as:

  - Balance Category
  ```python
  data["balance_category"] = pd.cut(data["balance"], bins=[-10000, 0, 1000, 5000, 100000],
                                labels=["Negative", "Low", "Medium", "High"])
  ```

  - Age Category
  ```python
  # กลุ่มอายุ
  data["age_group"] = pd.cut(data["age"], bins=[0, 30, 45, 60, 100],
                         labels=["Young", "Adult", "Senior", "Elderly"])
  ```
  - If have no contact data change it to unknown
  ```python
  # ช่องทางการติดต่อ
  data["contact_known"] = data["contact"].apply(lambda x: x != "unknown")
  ```

  - Month to number 
  ```python
  # เดือนเป็นตัวเลข
  month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
  data["month_num"] = data["month"].apply(lambda x: month_order.index(x.lower()) + 1 if x.lower() in month_order else None)
  ```

  - deposit flag if deposit = yes = 1 then = 0
  ```python
  #deposit flag
  data["deposit_flag"] = data["deposit"].apply(lambda x: 1 if x == "yes" else 0)
  ```
  
### 4. Download data that we alredy do feature engineer 
```python 
data.to_csv("Bank_Target_Marketing_Dataset_feature_engineered.csv", index=False)

from google.colab import files
files.download('Bank_Target_Marketing_Dataset_feature_engineered.csv')
```
---

##  Exploratory Data Analysis, EDA
### 1. Number of Calls vs. Average Deposit Flag 
- This scatter plot shows the relationship between the total number of calls (x-axis) and the average deposit flag (y-axis), segmented by job categories.
  - Jobs like "Student" and "Retired" have higher average deposit flags with fewer contact.
  - Jobs like "blue-collar" require more calls but have lower average deposit flags.

<img width="1258" alt="Response by Occupation" src="https://github.com/user-attachments/assets/43e395ad-41a4-4e14-8466-ac56bf4a47b6" />

---

### 2. Balance vs. Deposit Decision
- This box-and-whisker plot shows the distribution of account balances for customers who made deposits (`yes` in green) and those who did not (`no` in red).
  - Customers with  balances more than 0 are likely to deposit.

<img width="1258" alt="ความสัมพันธ์ระหว่างยอดเงินคงเหลือกับการฝากเงินเพิ่ม" src="https://github.com/user-attachments/assets/e016516f-2343-4099-946c-d3e305232213" />

---

### 3. Age vs. Balance
- This scatter plot visualizes the relationship between customer age (x-axis) and account balance (y-axis). The color intensity represents the deposit flag (`1` for deposit, `0` for no deposit).
  - Customers with higher balances are more likely to deposit (Green points).
  - Younger customers tend to have lower balances, while older customers show a wider range of balances.

<img width="1244" alt="ความสัมพันธ์อายุ vs ยอดเงินฝาก" src="https://github.com/user-attachments/assets/e6e1a497-4bc3-43cb-acc8-9eeffd80ca99" />

---

### 4. Monthly Campaign Deposits
- This line chart shows the total number of deposits `deposit_flag` over the months.
  - The highest number of deposits occurs in March and May.
  - There is a significant drop in deposits during the summer months (June to August).

<img width="1244" alt="แนวโน้มการตอบรับแคมเปญรายเดือน" src="https://github.com/user-attachments/assets/5aa4bea2-cea8-4296-a0c2-c27fdcc3add3" />

---

### 5. Monthly Campaign Response Rate
- This line chart tracks the response rate (y-axis) over the months (x-axis).
  - The highest response rate occurs in March, followed by December.
  - The response rate dips significantly in the middle months. `(May to August)`

<img width="1258" alt="แนวโน้มอัตราการตอบรับแคมเปญตามเวลา" src="https://github.com/user-attachments/assets/4932644d-0541-4f21-96f4-cbec67c1c6e3" />

---

### 6.  Campaign Effectiveness by Number of Calls**  
- This bar chart shows the conversion rate (y-axis) based on the number of calls (x-axis).
  - The highest conversion rate is observed with fewer calls `(1–2 calls)`.
  - As the number of calls increases, the conversion rate drops significantly.

<img width="1244" alt="ประสิทธิภาพของแคมเปญ" src="https://github.com/user-attachments/assets/0999c4b1-c7d2-4047-ae41-e35469bfc5df" />

---

### 7. Response by Day and Month**  
- This calendar-style heatmap shows the response rate for each day of the month across different months.
  - Darker squares indicate higher response rates.
  - Certain days in `March` a show higher response rates.

<img width="1244" alt="วันไหนเดือนใดมีอัตตราการตอบรับแคมเปญเท่าไหร่" src="https://github.com/user-attachments/assets/eddcf26e-0f33-45e4-aaf2-0406e8c03044" />

---

**8. Response Rate by Occupation**  
- This bar chart shows the number of deposits (`yes` in green, `no` in red) segmented by job categories.
  - The highest response rate is from "management" and "blue-collar" jobs, but the actual deposit rate (green portion) is relatively low compared to the total.

<img width="1258" alt="อัตราตอบรับตามอาชีพ" src="https://github.com/user-attachments/assets/6021d7c5-1aeb-4de6-97c2-c53b799e46e8" />

---

### 9. Response by Age Group
- This bar chart shows the response rate segmented by age groups (`Young`, `Adult`, `Senior`, `Elderly`).
  - The "Elderly" group has the highest response rate (42.26%).
  - Younger groups (Young and Adult) have significantly lower response rates.

<img width="1258" alt="อัตราตอบรับแยกตามกลุ่มอายุ" src="https://github.com/user-attachments/assets/a711f355-97eb-4d6f-af56-4e4db74594b9" />

---

## In-Depth Analysis

### Analysis Question 1  
**"Which variables have the most impact on campaign response (deposit = yes)?"**

To answer this question, we used Logistic Regression to analyze which variables most influence the likelihood that a customer will accept the campaign.
- Target : `deposit_flag`
- Feature Select :  `age`, `balance`, `campaign`,`previous`.

**Code**

```python 
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions & probabilities
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {auc:.3f}\n")

# Coefficients interpretation
coeff_df = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0]
}).sort_values(by='coefficient', ascending=False)
print("Feature Coefficients (Logistic Regression):\n", coeff_df)
```
#### Result 
- Model Performance:
  - Accuracy: ~88%
  - ROC AUC Score: 0.633 

- Feature Importance:
  - previous (+0.0999): Customers who responded positively to previous campaigns are more likely to respond again.
  - age (+0.0073): Older customers have a slightly higher likelihood of subscribing.
  - balance (+0.00003): Higher balances marginally increase the likelihood of subscription.
  - campaign (-0.1358): More frequent contact reduces the likelihood of a positive response, possibly due to annoyance.

---
### Analysis Question 2 
**We Need To Segment the Customer**

To segment customers based on behavior and likelihood of responding to campaigns.
We use KMeans Clustering 

- Feature Select : `age`, `balance`, `campaign`, `previous` to group customers into 4 clusters.

**code**

```python
# KMeans Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Fit KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Calculate response rate per cluster
cluster_rates = df.groupby('cluster')['deposit_flag'].mean().reset_index()
print("\nResponse Rate by Customer Cluster:")
print(cluster_rates)

# Plot response rate per cluster
plt.figure()
plt.bar(cluster_rates['cluster'].astype(str), cluster_rates['deposit_flag'])
plt.xlabel('Cluster')
plt.ylabel('Response Rate')
plt.title('Response Rate by Customer Cluster')
plt.show()
```
### Result
- Cluster Response Rates:
  - Cluster 0: 15% (highest response rate, primary target group).
  - Cluster 1: 11.8%.
  - Cluster 2: 4.3% (lowest response rate, avoid or redesign campaigns for this group).
  - Cluster 3: 12%.
--- 
## Resource

### Dataset Source

**Sean Angelo Nathanael, "Bank Target Marketing Dataset," Kaggle.**
- https://www.kaggle.com/datasets/seanangelonathanael/bank-target-marketing
 
### Python Libraries

**pandas: Data manipulation and analysis.**
https://pandas.pydata.org/
**numpy: Numerical computing.**
https://numpy.org/
**matplotlib: Data visualization.**
https://matplotlib.org/
**scikit-learn: Machine learning library.**
https://scikit-learn.org/

### Visualization Tool

**Tableau Public: Interactive data visualization.**
https://public.tableau.com/


---


