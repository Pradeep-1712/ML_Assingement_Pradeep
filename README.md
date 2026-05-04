# 📊 Machine Learning Assignment

## Product Performance Prediction using Machine Learning

------------------------------------------------------------------------

## 📌 Project Overview

This project focuses on analyzing and predicting **product ratings**
using machine learning techniques.\
The dataset contains various product attributes such as price, stock
quantity, warranty period, and category.

The main objective is to build a **regression model** that can predict
product ratings and provide insights into product performance.

------------------------------------------------------------------------

## 📂 Dataset Information

-   **Source:** Marketing and Product Performance Dataset (Kaggle)\
-   **Rows:** \~10,000\
-   **Columns:** 14\
-   **Target Variable:** Product Ratings

### Features Include:

-   Product Name\
-   Product Category\
-   Price\
-   Stock Quantity\
-   Warranty Period\
-   Product Dimensions\
-   Manufacturing Date\
-   Expiration Date\
-   SKU\
-   Product Tags

------------------------------------------------------------------------

## ⚙️ Technologies Used

-   Python 🐍\
-   Pandas\
-   NumPy\
-   Matplotlib\
-   Seaborn\
-   Scikit-learn

------------------------------------------------------------------------

## 🧠 Machine Learning Model

-   **Linear Regression**

### Workflow:

1.  Load dataset using Pandas\
2.  Clean data (remove null values and duplicates)\
3.  Normalize column names\
4.  Select numeric features\
5.  Split dataset into training (80%) and testing (20%)\
6.  Apply feature scaling (StandardScaler)\
7.  Train Linear Regression model\
8.  Predict and evaluate results

------------------------------------------------------------------------

## 📏 Evaluation Metrics

-   Mean Absolute Error (MAE)\
-   Mean Squared Error (MSE)\
-   R² Score

------------------------------------------------------------------------

## 📊 Research Questions (RQ1 -- RQ7)

### 🔹 RQ1: Average Price by Category

Analyzes average product price across categories after normalization.

### 🔹 RQ2: Stock Quantity by Category

Examines total stock distribution among categories.

### 🔹 RQ3: Average Ratings by Category

Evaluates average product ratings per category.

### 🔹 RQ4: Warranty Period Analysis

Analyzes average warranty period by category.

### 🔹 RQ5: Top 5 Expensive Products

Identifies the most expensive products in the dataset.

### 🔹 RQ6: Price vs Ratings

Explores relationship between product price and ratings using scatter
plot.

### 🔹 RQ7: Product Count by Category

Counts number of products in each category.

------------------------------------------------------------------------

## 📈 Outputs Generated

Each RQ generates: - 📄 CSV file (table) - 📊 Graph (PDF)

Example: - RQ1 → `RQ1_table.csv`, `RQ1_figure.pdf` - RQ2 →
`RQ2_table.csv`, `RQ2_figure.pdf` - ... - RQ7 → `RQ7_table.csv`,
`RQ7_figure.pdf`

------------------------------------------------------------------------

## 📂 Project Structure

    ├── products.csv
    ├── 1.py  (RQ1)
    ├── 2.py  (RQ2)
    ├── 3.py  (RQ3)
    ├── 4.py  (RQ4)
    ├── 5.py  (RQ5)
    ├── 6.py  (RQ6)
    ├── 7.py  (RQ7)
    ├── README.md

------------------------------------------------------------------------

## 🚀 How to Run the Project

### Step 1: Install Dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 2: Run Scripts

``` bash
python3 1.py
python3 2.py
python3 3.py
python3 4.py
python3 5.py
python3 6.py
python3 7.py
```

------------------------------------------------------------------------

## 📊 Key Insights

-   Product performance is influenced by price, stock, and warranty\
-   Different categories show varying trends in ratings\
-   Data preprocessing improves model accuracy\
-   Linear Regression provides baseline performance

------------------------------------------------------------------------

## ✅ Conclusion

This project demonstrates how machine learning can be applied to: -
Predict product ratings\
- Analyze business data\
- Generate meaningful insights

It serves as a strong foundation for further improvements using advanced
models like Random Forest or XGBoost.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Use advanced models (Random Forest, XGBoost)\
-   Hyperparameter tuning\
-   Feature engineering\
-   Deploy model as web app

------------------------------------------------------------------------

🔥 Clean • Professional • GitHub Ready
