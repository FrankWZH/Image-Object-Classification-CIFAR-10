import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 补充缺失的引用
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional for interpretation
# import shap

import kagglehub

# ==========================================
# 1. Data Loading and Initial Inspection
# ==========================================

# 下载数据集 (保留你原有的逻辑)
path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")
file_path = path + "/loan_approval_dataset.csv"

# Task 1.1: Load the dataset
df = pd.read_csv(file_path)

# CRITICAL STEP: Clean column names (remove leading spaces common in this dataset)
df.columns = df.columns.str.strip()

# Task 1.2: Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Task 1.3: Print info and describe
print("\nDataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# Task 1.4: Identify missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Task 1.5: Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================

# Example 2.1: Visualize distributions (Corrected column names)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# Handling potential infinite values for plotting
df['income_annum'] = df['income_annum'].replace([np.inf, -np.inf], np.nan)

sns.histplot(df['income_annum'], kde=True, ax=axes[0,0])
axes[0,0].set_title('Income Annum Distribution')

sns.histplot(df['loan_amount'], kde=True, ax=axes[0,1])
axes[0,1].set_title('Loan Amount Distribution')

sns.countplot(x='education', data=df, ax=axes[0,2])
axes[0,2].set_title('Education Count')

sns.countplot(x='self_employed', data=df, ax=axes[1,0])
axes[1,0].set_title('Self Employed Count')

sns.histplot(df['loan_term'], kde=True, ax=axes[1,1])
axes[1,1].set_title('Loan Term Distribution')

sns.histplot(df['cibil_score'], kde=True, ax=axes[1,2])
axes[1,2].set_title('CIBIL Score Distribution')

plt.tight_layout()
plt.show()

# Task 2.1: Visualize the distribution of the target variable (loan_status)
plt.figure(figsize=(6, 4))
sns.countplot(x='loan_status', data=df)
plt.title('Distribution of Target Variable (Loan Status)')
plt.show()

# Task 2.2: Explore Feature Relationships
# Relationship 1: Loan Status vs CIBIL Score (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(x='loan_status', y='cibil_score', data=df)
plt.title('Loan Status vs CIBIL Score')
plt.show()

# Relationship 2: Loan Status vs Income (Boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(x='loan_status', y='income_annum', data=df)
plt.title('Loan Status vs Annual Income')
plt.show()

# ==========================================
# 3. Data Preprocessing
# ==========================================

print(pd.__version__)

# Create a copy to avoid modifying the original DataFrame
df_processed = df.copy()

# Task 3.1: Handle Missing Values
# Note: This dataset usually has no missing values, but we implement the logic as per requirements.
numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df_processed.select_dtypes(include=['object']).columns

# Imputer for numerical (using median)
imputer_num = SimpleImputer(strategy='median')
df_processed[numerical_cols] = imputer_num.fit_transform(df_processed[numerical_cols])

# Imputer for categorical (using most_frequent)
# Only applying if there are actually categorical columns left (excluding target for now)
if len(categorical_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df_processed[categorical_cols] = imputer_cat.fit_transform(df_processed[categorical_cols])


# Task 4.1: Feature Engineering (Moved here before encoding/scaling/splitting)
# ------------------------------------------------------------------------
# Create at least two new meaningful features

# Feature 1: Total Assets
# Rationale: Combining all assets gives a better picture of the applicant's net worth/collateral.
df_processed['total_assets'] = (df_processed['residential_assets_value'] +
                                df_processed['commercial_assets_value'] +
                                df_processed['luxury_assets_value'] +
                                df_processed['bank_asset_value'])

# Feature 2: Loan to Income Ratio
# Rationale: Measures the burden of the loan relative to income. Higher ratio = riskier.
# Add a small epsilon to avoid division by zero
df_processed['loan_to_income_ratio'] = df_processed['loan_amount'] / (df_processed['income_annum'] + 1)

print("\nNew Features Created: 'total_assets', 'loan_to_income_ratio'")


# Task 3.2: Encode Categorical Variables
# ------------------------------------------------------------------------
label_encoder = LabelEncoder()

# Encode 'education' (Graduate/Not Graduate)
df_processed['education'] = label_encoder.fit_transform(df_processed['education'])

# Encode 'self_employed' (Yes/No)
df_processed['self_employed'] = label_encoder.fit_transform(df_processed['self_employed'])

# Encode Target 'loan_status' (Approved/Rejected)
# Usually Approved=1, Rejected=0. Check labels after encoding or map manually to be safe.
df_processed['loan_status'] = df_processed['loan_status'].map({' Approved': 1, ' Rejected': 0}).fillna(0).astype(int)
# Note: Because of potential whitespace in values, using strip first or map with spaces is safer.
# Since we stripped columns, let's ensure values are clean too.
# Alternative robust method:
# df_processed['loan_status'] = label_encoder.fit_transform(df_processed['loan_status'])


# Task 3.3: Feature Scaling
# ------------------------------------------------------------------------
# We should scale numerical features, especially for Logistic Regression / SVM
scaler = StandardScaler()

# Identify features to scale (exclude target)
features_to_scale = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
                     'cibil_score', 'residential_assets_value', 'commercial_assets_value',
                     'luxury_assets_value', 'bank_asset_value', 'total_assets', 'loan_to_income_ratio']

df_processed[features_to_scale] = scaler.fit_transform(df_processed[features_to_scale])


# Task 3.4: Data Splitting
# ------------------------------------------------------------------------
X = df_processed.drop('loan_status', axis=1)
y = df_processed['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Using 0.3 (70/30 split as per PDF suggestion)

print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# ==========================================
# 5. Model Development
# ==========================================

# Task 5.1: Initialize and train at least three models

# Model 1: Logistic Regression
model_lr = LogisticRegression(random_state=42, solver='liblinear')
model_lr.fit(X_train, y_train)

# Model 2: Decision Tree Classifier (Completing the blank)
model_dt = DecisionTreeClassifier(random_state=42, max_depth=5) # Added max_depth to prevent overfitting
model_dt.fit(X_train, y_train)

# Model 3: Random Forest Classifier (PDF asks for three models)
model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
model_rf.fit(X_train, y_train)

print("\nModels Trained: Logistic Regression, Decision Tree, Random Forest")

# ==========================================
# 6. Model Evaluation
# ==========================================

# Task 6.1: Make predictions on the test set
y_pred_lr = model_lr.predict(X_test)
y_pred_dt = model_dt.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Helper function to print metrics
def print_metrics(y_true, y_pred, model_name):
    print(f"\n### {model_name} Evaluation ###")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# Task 6.2 & 6.3: Calculate and print evaluation metrics
print_metrics(y_test, y_pred_lr, "Logistic Regression")
print_metrics(y_test, y_pred_dt, "Decision Tree")
print_metrics(y_test, y_pred_rf, "Random Forest") # Bonus model for the requirement

# Task 6.4: Compare models (Text Discussion Placeholder)
print("\nDiscussion:")
print("Compare the Accuracy and F1-Scores above. Usually, Random Forest performs best on this dataset.")
print("Check the Confusion Matrix to see which model minimizes False Positives (Bad Loans approved).")