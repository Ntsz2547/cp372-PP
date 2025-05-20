# CP372 Final Project – Bank Target Marketing

This repository contains the final project for CP372: Data Analysis and Visualization. The project focuses on analyzing direct marketing campaign data from a banking institution to generate actionable insights and recommendations for improving campaign performance.

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
https://public.tableau.com/views/CP372_Final_Project/LineChart?:language=th-TH&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

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
### 1. Number of Calls vs. Average Deposit Flag**  
- This scatter plot shows the relationship between the total number of calls (x-axis) and the average deposit flag (y-axis), segmented by job categories.
<img width="1258" alt="Response by Occupation" src="https://github.com/user-attachments/assets/43e395ad-41a4-4e14-8466-ac56bf4a47b6" />

---

### 2. Balance vs. Deposit Decision**  
- This box-and-whisker plot shows the distribution of account balances for customers who made deposits (yes in green) and those who did not (no in red).

<img width="1258" alt="ความสัมพันธ์ระหว่างยอดเงินคงเหลือกับการฝากเงินเพิ่ม" src="https://github.com/user-attachments/assets/e016516f-2343-4099-946c-d3e305232213" />

---

### 3. Age vs. Deposit Amount**  
- This plot displays the relationship between customer age and the amount they deposited.

<img width="1258" alt="Age vs. Deposit Amount" src="https://github.com/user-attachments/assets/c5d1a4a2-67a0-4754-ad4a-9c7fa86ed6ee" />

---

### 4. Monthly Campaign Response Trend**  
- This chart shows the trend of campaign responses over different months.

<img width="1258" alt="แนวโน้มการตอบรับแคมเปญรายเดือน" src="https://github.com/user-attachments/assets/3d843b63-1fa0-47dd-8c75-8d6a73f3b9fd" />

---

### 5. Campaign Response Rate Over Time**  
- This visualization tracks the response rate to the campaign over time.

<img width="1258" alt="แนวโน้มอัตราการตอบรับแคมเปญตามเวลา" src="https://github.com/user-attachments/assets/4932644d-0541-4f21-96f4-cbec67c1c6e3" />

---

### 6. Campaign Effectiveness**  
- This image summarizes the overall effectiveness of the marketing campaign.

<img width="1258" alt="ประสิทธิภาพของแคมเปญ" src="https://github.com/user-attachments/assets/2f3fb123-c6ae-4e93-ad64-671cb5deda1e" />

---

### 7. Daily/Monthly Response Rate**  
- This chart breaks down the campaign response rate by specific days or months.

<img width="1258" alt="วันไหนเดือนใดมีอัตตราการตอบรับแคมเปญเท่าไหร่" src="https://github.com/user-attachments/assets/4c1574b5-acd7-48ef-a613-5ec238574c15" />

---

**8. Response Rate by Occupation**  
- This visualization compares the campaign response rates across different occupations.

<img width="1258" alt="อัตราตอบรับตามอาชีพ" src="https://github.com/user-attachments/assets/6021d7c5-1aeb-4de6-97c2-c53b799e46e8" />

---

### 9. Response Rate by Age Group**  
- This plot shows the response rate to the campaign segmented by age groups.

<img width="1258" alt="อัตราตอบรับแยกตามกลุ่มอายุ" src="https://github.com/user-attachments/assets/a711f355-97eb-4d6f-af56-4e4db74594b9" />

---

## In-Depth Analysis

### Analysis Question 1  
**"Which variables have the most impact on campaign response (deposit = yes)?"**

To answer this question, we used Logistic Regression to analyze which variables most influence the likelihood that a customer will accept the campaign.

---

## Visualizations
<img width="1261" alt="Dashboard" src="https://github.com/user-attachments/assets/55ba5875-8980-46a8-b6c2-7d45cd660c7a" />

---

##  Deliverables

- Jupyter Notebook: Full data analysis and cleaning process
- Tableau Workbook (.twbx): Interactive dashboards
- Project Canvas (PDF): Summarized project scope, objectives, metrics, and timeline

---


