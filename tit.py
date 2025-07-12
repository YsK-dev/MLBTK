# %%
"""
# 🚢 Titanic Survival Prediction - Complete Analysis and Modeling

## Overview
This notebook provides a comprehensive analysis of the Titanic dataset from Kaggle. We'll explore the data, engineer features, and build machine learning models to predict passenger survival.

## Competition Goal
Predict which passengers survived the Titanic shipwreck based on passenger data including:
- Personal information (age, sex, class)
- Family relationships (siblings, parents/children)
- Ticket information (fare, cabin, embarkation port)

## Approach
1. **Exploratory Data Analysis (EDA)** - Understanding patterns in the data
2. **Feature Engineering** - Creating meaningful features from raw data
3. **Statistical Analysis** - Testing relationships between features and survival
4. **Model Building** - Training multiple models with hyperparameter tuning
5. **Ensemble Methods** - Combining models for better performance
"""

# %%
"""
## 📊 Data Loading and Initial Exploration

Let's start by loading the data and getting a basic understanding of what we're working with.
"""

# Titanic Dataset Analysis - Enhanced EDA and Feature Engineering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from scipy.stats import chi2_contingency
import re

# Set style and ignore warnings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Configure plotting
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# %%
"""
### 🔍 Dataset Overview

First, let's load the training and test datasets and examine their basic structure.

**Key Observations:**
- Training set contains survival labels (our target variable)
- Test set is what we'll make predictions on for submission
- Both datasets should have similar feature distributions
"""

# Load the datasets
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

print("=== DATASET OVERVIEW ===")
print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")
print(f"Total features: {train.shape[1]}")

print("\n=== TRAINING SET INFO ===")
print(train.info())

print("\n=== TEST SET INFO ===")
print(test.info())

print("\n=== SURVIVAL STATISTICS ===")
survival_rate = train['Survived'].mean()
print(f"Overall survival rate: {survival_rate:.2%}")
print(f"Survived: {train['Survived'].sum()} passengers")
print(f"Did not survive: {len(train) - train['Survived'].sum()} passengers")

# Display first few rows
print("\n=== SAMPLE DATA ===")
print("Training data:")
print(train.head())
print("\nTest data:")
print(test.head())

# Basic statistics
print("\n=== BASIC STATISTICS ===")
print(train.describe())

# %%
"""
## 🔎 Missing Values Analysis

Understanding missing data is crucial for proper preprocessing. Different missing patterns might indicate different underlying reasons:

- **Age**: Likely missing at random - we can impute based on other features
- **Cabin**: High missingness - might indicate lower class passengers
- **Embarked**: Very few missing - can use mode imputation

**Strategy:**
- Visualize missing patterns
- Analyze if missingness correlates with survival
- Choose appropriate imputation methods
"""

# Check missing values in both datasets
print("=== MISSING VALUES ANALYSIS - TRAINING ===")
train_missing = train.isnull().sum()
train_missing_pct = 100 * train_missing / len(train)

train_missing_table = pd.DataFrame({
    'Missing Count': train_missing,
    'Missing Percentage': train_missing_pct
}).sort_values('Missing Percentage', ascending=False)

print(train_missing_table[train_missing_table['Missing Count'] > 0])

print("\n=== MISSING VALUES ANALYSIS - TEST ===")
test_missing = test.isnull().sum()
test_missing_pct = 100 * test_missing / len(test)

test_missing_table = pd.DataFrame({
    'Missing Count': test_missing,
    'Missing Percentage': test_missing_pct
}).sort_values('Missing Percentage', ascending=False)

print(test_missing_table[test_missing_table['Missing Count'] > 0])

# Visualize missing values
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training set missing counts
if len(train_missing_table[train_missing_table['Missing Count'] > 0]) > 0:
    train_missing_table[train_missing_table['Missing Count'] > 0]['Missing Count'].plot(
        kind='bar', ax=axes[0,0], color='lightcoral')
    axes[0,0].set_title('Training Set - Missing Values Count')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Training set missing percentages
    train_missing_table[train_missing_table['Missing Count'] > 0]['Missing Percentage'].plot(
        kind='bar', ax=axes[0,1], color='orange')
    axes[0,1].set_title('Training Set - Missing Values Percentage')
    axes[0,1].set_ylabel('Percentage')
    axes[0,1].tick_params(axis='x', rotation=45)

# Test set missing counts
if len(test_missing_table[test_missing_table['Missing Count'] > 0]) > 0:
    test_missing_table[test_missing_table['Missing Count'] > 0]['Missing Count'].plot(
        kind='bar', ax=axes[1,0], color='lightblue')
    axes[1,0].set_title('Test Set - Missing Values Count')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Test set missing percentages
    test_missing_table[test_missing_table['Missing Count'] > 0]['Missing Percentage'].plot(
        kind='bar', ax=axes[1,1], color='lightgreen')
    axes[1,1].set_title('Test Set - Missing Values Percentage')
    axes[1,1].set_ylabel('Percentage')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Analyze relationship between missing values and survival
print("\n=== MISSING VALUES vs SURVIVAL ===")
if train['Cabin'].isnull().sum() > 0:
    cabin_survival = train.groupby(train['Cabin'].isnull())['Survived'].mean()
    print("Survival rate by Cabin availability:")
    print(f"Has Cabin: {cabin_survival[False]:.2%}")
    print(f"No Cabin: {cabin_survival[True]:.2%}")

if train['Age'].isnull().sum() > 0:
    age_survival = train.groupby(train['Age'].isnull())['Survived'].mean()
    print("\nSurvival rate by Age availability:")
    print(f"Has Age: {age_survival[False]:.2%}")
    print(f"No Age: {age_survival[True]:.2%}")

# %%
"""
## 📊 Basic Exploratory Data Analysis

Let's explore the relationships between features and survival rates.
"""

# Survival by categorical features
categorical_features = ['Pclass', 'Sex', 'Embarked']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, feature in enumerate(categorical_features):
    # Calculate survival rates
    survival_by_feature = train.groupby(feature)['Survived'].agg(['mean', 'count'])
    survival_by_feature['survival_rate'] = (survival_by_feature['mean'] * 100).round(1)
    
    print(f"\n=== SURVIVAL BY {feature.upper()} ===")
    print(survival_by_feature)
    
    # Plot
    survival_by_feature['survival_rate'].plot(kind='bar', ax=axes[i], color='skyblue')
    axes[i].set_title(f'Survival Rate by {feature}')
    axes[i].set_ylabel('Survival Rate (%)')
    axes[i].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for j, v in enumerate(survival_by_feature['survival_rate']):
        axes[i].text(j, v + 1, f'{v}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Age and Fare distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Age distribution by survival
axes[0,0].hist(train[train['Survived']==0]['Age'].dropna(), alpha=0.7, label='Not Survived', bins=20)
axes[0,0].hist(train[train['Survived']==1]['Age'].dropna(), alpha=0.7, label='Survived', bins=20)
axes[0,0].set_title('Age Distribution by Survival')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Count')
axes[0,0].legend()

# Fare distribution by survival
axes[0,1].hist(train[train['Survived']==0]['Fare'].dropna(), alpha=0.7, label='Not Survived', bins=20)
axes[0,1].hist(train[train['Survived']==1]['Fare'].dropna(), alpha=0.7, label='Survived', bins=20)
axes[0,1].set_title('Fare Distribution by Survival')
axes[0,1].set_xlabel('Fare')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()

# Age vs Fare scatter plot
colors = ['red' if x == 0 else 'green' for x in train['Survived']]
axes[1,0].scatter(train['Age'], train['Fare'], c=colors, alpha=0.6)
axes[1,0].set_title('Age vs Fare by Survival')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Fare')

# Family size analysis
train['Family_Size'] = train['SibSp'] + train['Parch'] + 1
family_survival = train.groupby('Family_Size')['Survived'].mean()
family_survival.plot(kind='bar', ax=axes[1,1], color='lightgreen')
axes[1,1].set_title('Survival Rate by Family Size')
axes[1,1].set_xlabel('Family Size')
axes[1,1].set_ylabel('Survival Rate')
axes[1,1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

print("\n=== FAMILY SIZE SURVIVAL ANALYSIS ===")
family_analysis = train.groupby('Family_Size')['Survived'].agg(['mean', 'count'])
family_analysis['survival_rate'] = (family_analysis['mean'] * 100).round(1)
print(family_analysis)

# %%
"""
## 🛠️ Advanced Feature Engineering

Feature engineering is often the key to winning Kaggle competitions. We'll create meaningful features that capture important patterns in the data.

### New Features Created:
1. **Title Extraction** - Social status from names (Mr, Mrs, Dr, etc.)
2. **Family Features** - Family size, family survival rate
3. **Ticket Analysis** - Ticket prefixes and numbers
4. **Cabin Analysis** - Deck information and cabin availability
5. **Age Groups** - Meaningful age categories
6. **Fare Analysis** - Fare per person, fare bins
7. **Social Class** - Estimated social status
8. **Woman and Child** - "Women and children first" policy
9. **Interaction Features** - Combined feature effects

### Why These Features Matter:
- **Titles** reveal social status and gender (survival patterns)
- **Family size** affects survival chances (small families better)
- **Cabin deck** indicates class and location on ship
- **Age groups** capture different survival policies
"""

def create_features(df):
    """Create advanced features for the dataset"""
    df = df.copy()
    
    # 1. Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group titles strategically
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Officer', 'Rev': 'Officer', 'Col': 'Officer', 'Capt': 'Officer', 'Major': 'Officer',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Don': 'Royalty', 'Dona': 'Royalty', 'Lady': 'Royalty', 'the Countess': 'Royalty',
        'Sir': 'Royalty', 'Jonkheer': 'Royalty'
    }
    df['Title'] = df['Title'].map(title_mapping).fillna('Rare')
    
    # 2. Family features
    df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
    df['Family_Size_Cat'] = df['Family_Size'].apply(lambda x: 'Alone' if x == 1 
                                                   else 'Small' if x <= 3 
                                                   else 'Medium' if x <= 5 
                                                   else 'Large')
    
    # 3. Ticket analysis
    df['Ticket_Prefix'] = df['Ticket'].str.extract(r'([A-Za-z]+)', expand=False).fillna('None')
    df['Has_Ticket_Prefix'] = (df['Ticket_Prefix'] != 'None').astype(int)
    
    # 4. Cabin analysis
    df['Has_Cabin'] = (~df['Cabin'].isnull()).astype(int)
    df['Cabin_Deck'] = df['Cabin'].str[0]
    
    # 5. Age groups (after imputation)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 80], 
                            labels=['Child', 'Teenager', 'Young_Adult', 'Adult', 'Senior'])
    
    # 6. Fare analysis
    df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']
    
    # 7. Social class estimation
    df['Social_Class'] = 'Lower'
    df.loc[(df['Pclass'] == 1) | (df['Title'].isin(['Royalty', 'Officer'])), 'Social_Class'] = 'Upper'
    df.loc[(df['Pclass'] == 2) | (df['Fare'] > df['Fare'].quantile(0.75)), 'Social_Class'] = 'Middle'
    
    # 8. Woman and child first policy
    df['Woman_Child'] = ((df['Sex'] == 'female') | (df['Age'] < 16)).astype(int)
    
    # 9. Interaction features
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
    df['Age_Pclass'] = df['Age'] * df['Pclass']
    
    return df

# Combine datasets for consistent preprocessing
df = pd.concat([train, test], sort=False).reset_index(drop=True)

print("=== FEATURE ENGINEERING ===")
print(f"Original features: {len(train.columns)}")

# Handle missing values first
print("\n=== HANDLING MISSING VALUES ===")

# Age imputation based on Title, Pclass, Sex
age_median = df.groupby(['Sex', 'Pclass'])['Age'].median()
for idx, row in df[df['Age'].isnull()].iterrows():
    key = (row['Sex'], row['Pclass'])
    if key in age_median.index:
        df.loc[idx, 'Age'] = age_median[key]
    else:
        df.loc[idx, 'Age'] = df['Age'].median()

# Fare imputation
df['Fare'].fillna(df.groupby(['Pclass'])['Fare'].transform('median'), inplace=True)

# Embarked imputation
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("Missing values after imputation:")
print(df.isnull().sum().sum())

# Create features
df = create_features(df)

print(f"Features after engineering: {len(df.columns)}")
print("New features created:", [col for col in df.columns if col not in train.columns])

# Create fare bins after feature engineering
df['Fare_Bin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very_High'])

# Display feature engineering results
print("\n=== FEATURE ENGINEERING RESULTS ===")
print("Title distribution:")
print(df['Title'].value_counts())

print("\nFamily size categories:")
print(df['Family_Size_Cat'].value_counts())

print("\nSocial class distribution:")
print(df['Social_Class'].value_counts())

# %%
"""
## 📈 Statistical Survival Analysis

We'll perform rigorous statistical testing to understand which features truly matter for survival prediction.

### Statistical Tests Used:
- **Chi-square tests** for categorical variables
- **P-value analysis** to determine statistical significance
- **Survival rate calculations** by feature groups

### Key Insights Expected:
- Gender will be the strongest predictor
- Class and fare will show clear survival gradients  
- Age will show "women and children first" pattern
- Family size will have optimal ranges
"""

# Enhanced survival analysis with statistical tests
def comprehensive_survival_analysis(df):
    """Comprehensive survival analysis with statistical significance"""
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Family_Size_Cat', 
                           'Age_Group', 'Fare_Bin', 'Social_Class', 'Has_Cabin']
    
    print("=== SURVIVAL ANALYSIS BY CATEGORICAL FEATURES ===")
    
    # Create subplots for better visualization
    n_features = len(categorical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(categorical_features):
        if i >= len(axes):
            break
            
        # Calculate survival rates
        survival_by_feature = df.groupby(feature)['Survived'].agg(['mean', 'count']).round(3)
        survival_by_feature['survival_rate'] = (survival_by_feature['mean'] * 100).round(1)
        
        # Chi-square test for statistical significance
        contingency_table = pd.crosstab(df[feature], df['Survived'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        print(f"\n{feature}:")
        print(survival_by_feature)
        print(f"Chi-square test: χ² = {chi2:.3f}, p-value = {p_value:.4f}")
        print("Statistically significant" if p_value < 0.05 else "Not statistically significant")
        
        # Plot
        ax = axes[i]
        survival_by_feature['survival_rate'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'{feature} - Survival Rate\n(p-value: {p_value:.4f})')
        ax.set_ylabel('Survival Rate (%)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for j, v in enumerate(survival_by_feature['survival_rate']):
            ax.text(j, v + 1, f'{v}%', ha='center', va='bottom')
    
    # Hide unused subplots
    for i in range(len(categorical_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

comprehensive_survival_analysis(df)

# %%
"""
## 📊 Advanced Data Visualization

Comprehensive visualizations help us understand complex patterns and relationships in the data.

### Visualization Strategy:
1. **Interactive Plotly charts** for detailed exploration
2. **Multi-panel comparisons** across different dimensions
3. **Feature correlation analysis** to identify redundant features
4. **Mutual information scoring** for feature importance

### What We're Looking For:
- Clear survival patterns by demographics
- Feature interactions and correlations
- Outliers and data quality issues
- Feature importance for model selection
"""

# Advanced visualization dashboard
def create_advanced_visualizations(df):
    """Create comprehensive visualization dashboard"""
    
    # 1. Age distribution analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Age Distribution by Survival', 'Age vs Fare by Survival',
                       'Family Size vs Survival Rate', 'Fare Distribution by Class'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Age distribution
    fig.add_trace(go.Histogram(x=df[df['Survived']==0]['Age'], name='Not Survived', 
                              opacity=0.7, nbinsx=20), row=1, col=1)
    fig.add_trace(go.Histogram(x=df[df['Survived']==1]['Age'], name='Survived', 
                              opacity=0.7, nbinsx=20), row=1, col=1)
    
    # Age vs Fare scatter
    colors = ['red' if x == 0 else 'green' for x in df['Survived']]
    fig.add_trace(go.Scatter(x=df['Age'], y=df['Fare'], mode='markers',
                            marker=dict(color=colors, opacity=0.6),
                            name='Age vs Fare'), row=1, col=2)
    
    # Family size survival rate
    family_survival = df.groupby('Family_Size')['Survived'].mean().reset_index()
    fig.add_trace(go.Bar(x=family_survival['Family_Size'], y=family_survival['Survived'],
                        name='Survival Rate by Family Size'), row=2, col=1)
    
    # Fare by class
    for pclass in sorted(df['Pclass'].unique()):
        fig.add_trace(go.Violin(y=df[df['Pclass']==pclass]['Fare'], 
                               name=f'Class {pclass}', box_visible=True), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=True, title_text="Titanic Dataset - Advanced Analysis")
    fig.show()
    
    # 2. Correlation heatmap with enhanced features
    plt.figure(figsize=(16, 12))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # 3. Feature importance visualization (using mutual information)
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data for mutual information
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col != 'Survived':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Calculate mutual information
    feature_cols = [col for col in df_encoded.columns if col not in ['Survived', 'PassengerId']]
    train_data = df_encoded[:len(train)]
    
    mi_scores = mutual_info_classif(train_data[feature_cols], train_data['Survived'])
    mi_scores = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    mi_scores.head(15).plot(kind='barh', color='lightcoral')
    plt.title('Top 15 Features by Mutual Information Score', fontsize=14, pad=20)
    plt.xlabel('Mutual Information Score')
    plt.tight_layout()
    plt.show()
    
    return mi_scores

mi_scores = create_advanced_visualizations(df)

# %%
"""
## 🤖 Machine Learning Model Development

Now we'll build and optimize machine learning models using best practices for Kaggle competitions.

### Model Selection Strategy:
1. **Multiple Algorithm Testing** - Random Forest, Gradient Boosting, Logistic Regression
2. **Hyperparameter Optimization** - Grid search with cross-validation
3. **Feature Selection** - Using mutual information scores
4. **Cross-Validation** - Robust performance estimation

### Why These Models:
- **Random Forest**: Handles mixed data types, reduces overfitting
- **Gradient Boosting**: Often wins competitions, handles complex patterns  
- **Logistic Regression**: Interpretable baseline, good for linear relationships
"""

# Enhanced model preparation with feature selection
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

def prepare_model_data(df, mi_scores):
    """Prepare data for modeling with feature selection"""
    
    # Split back to train and test
    train_df = df[:len(train)].copy()
    test_df = df[len(train):].copy()
    
    # Remove target from test set
    test_df = test_df.drop('Survived', axis=1)
    
    # Select top features based on mutual information
    top_features = mi_scores.head(15).index.tolist()
    
    # Prepare features and target
    X = train_df[top_features]
    y = train_df['Survived']
    X_test_final = test_df[top_features]
    
    print("=== SELECTED FEATURES ===")
    print(f"Total features selected: {len(top_features)}")
    print("Features:", top_features)
    
    # Handle categorical variables
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Numerical features: {len(numerical_features)}")
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, X_test_final, preprocessor, train_df

X, y, X_test_final, preprocessor, train_df = prepare_model_data(df, mi_scores)

# %%
"""
## 🔧 Hyperparameter Optimization with Grid Search

Grid search helps us find the optimal model configurations. This is crucial for maximizing performance on the leaderboard.

### Grid Search Strategy:
- **Comprehensive parameter grids** covering important hyperparameters
- **5-fold cross-validation** for robust evaluation
- **Multiple scoring metrics** to avoid overfitting
- **Computational efficiency** using parallel processing

### Parameters Being Optimized:
- **Random Forest**: n_estimators, max_depth, min_samples_split, max_features
- **Gradient Boosting**: learning_rate, max_depth, n_estimators, subsample
- **Logistic Regression**: C (regularization), penalty type, solver

### Expected Improvements:
- 2-5% accuracy boost from optimization
- Better generalization to test set
- Reduced overfitting through regularization
"""

# Enhanced model training with Grid Search
def train_and_evaluate_models_with_gridsearch(X, y, preprocessor):
    """Train multiple models with grid search and compare performance"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models with parameter grids
    models_and_params = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['sqrt', 'log2', None]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1, 0.15],
                'classifier__max_depth': [3, 5, 7],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2],
                'classifier__subsample': [0.8, 0.9, 1.0]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'classifier__C': [0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            }
        }
    }
    
    results = {}
    best_models = {}
    
    print("=== GRID SEARCH MODEL EVALUATION ===")
    
    for name, model_config in models_and_params.items():
        print(f"\nOptimizing {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model_config['model'])
        ])
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            model_config['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)
        
        # Cross-validation on best model
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Store results
        results[name] = {
            'best_estimator': best_model,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'val_accuracy': accuracy_score(y_val, y_pred_val),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'grid_search': grid_search
        }
        
        best_models[name] = best_model
        
        print(f"\n{name} Results:")
        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"  Best CV Score: {grid_search.best_score_:.4f}")
        print(f"  Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        print(f"  Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
        print(f"  CV Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['best_cv_score'])
    best_pipeline = results[best_model_name]['best_estimator']
    
    print(f"\n=== BEST MODEL: {best_model_name} ===")
    print(f"Best CV Score: {results[best_model_name]['best_cv_score']:.4f}")
    print(f"Best Parameters: {results[best_model_name]['best_params']}")
    
    # Create comparison visualization
    plt.figure(figsize=(15, 5))
    
    # 1. CV Scores comparison
    plt.subplot(1, 3, 1)
    model_names = list(results.keys())
    cv_scores = [results[name]['best_cv_score'] for name in model_names]
    bars = plt.bar(model_names, cv_scores, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Best CV Scores by Model')
    plt.ylabel('CV Score')
    plt.xticks(rotation=45)
    for i, v in enumerate(cv_scores):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Validation accuracy comparison
    plt.subplot(1, 3, 2)
    val_scores = [results[name]['val_accuracy'] for name in model_names]
    bars = plt.bar(model_names, val_scores, color=['lightcoral', 'lightblue', 'lightgray'])
    plt.title('Validation Accuracy by Model')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    for i, v in enumerate(val_scores):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. Overfitting analysis (Train vs Val accuracy)
    plt.subplot(1, 3, 3)
    train_scores = [results[name]['train_accuracy'] for name in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    plt.bar(x - width/2, train_scores, width, label='Train', alpha=0.7)
    plt.bar(x + width/2, val_scores, width, label='Validation', alpha=0.7)
    plt.title('Train vs Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Models')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance for best model if available
    if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
        feature_names = (preprocessor.named_transformers_['num'].get_feature_names_out().tolist() + 
                        preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
        
        importances = best_pipeline.named_steps['classifier'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importances - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print(f"\nTop 10 Feature Importances for {best_model_name}:")
        print(feature_importance.head(10))
    
    return results, best_pipeline, X_val, y_val, best_models

# Run grid search
results, best_pipeline, X_val, y_val, best_models = train_and_evaluate_models_with_gridsearch(X, y, preprocessor)

# %%
"""
## 🎯 Ensemble Methods for Maximum Performance

Ensemble methods are the secret weapon of Kaggle winners. By combining multiple models, we can achieve better performance than any single model.

### Ensemble Strategy:
1. **Voting Classifier** - Combines predictions from best models
2. **Soft Voting** - Uses prediction probabilities for better results
3. **Model Diversity** - Different algorithms capture different patterns
4. **Cross-Validation** - Ensures ensemble generalization

### Why Ensembles Work:
- **Bias-Variance Tradeoff** - Different models have different biases
- **Error Cancellation** - Individual model errors cancel out
- **Robustness** - Less sensitive to outliers and noise
- **Competition Edge** - Often provides the winning margin

### Expected Benefits:
- 1-3% accuracy improvement over best single model
- More robust predictions
- Better handling of edge cases
"""

# Advanced model ensemble and stacking
def create_ensemble_model(best_models, X_train, y_train, X_val, y_val):
    """Create ensemble model using voting classifier"""
    from sklearn.ensemble import VotingClassifier
    
    print("=== CREATING ENSEMBLE MODEL ===")
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'  # Use probability predictions
    )
    
    # Fit ensemble
    voting_clf.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = voting_clf.predict(X_train)
    y_pred_val = voting_clf.predict(X_val)
    
    # Cross-validation
    cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')
    
    print(f"Ensemble Train Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"Ensemble Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
    print(f"Ensemble CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Compare with individual models
    print("\n=== ENSEMBLE VS INDIVIDUAL MODELS ===")
    print(f"{'Model':<20} {'Val Accuracy':<15} {'CV Score':<15}")
    print("-" * 50)
    
    for name, model in best_models.items():
        val_pred = model.predict(X_val)
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        print(f"{name:<20} {accuracy_score(y_val, val_pred):<15.4f} {cv_score:<15.4f}")
    
    print(f"{'Ensemble':<20} {accuracy_score(y_val, y_pred_val):<15.4f} {cv_scores.mean():<15.4f}")
    
    return voting_clf

# Split data for ensemble
X_train_ens, X_val_ens, y_train_ens, y_val_ens = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Transform data for ensemble
X_train_transformed = preprocessor.fit_transform(X_train_ens)
X_val_transformed = preprocessor.transform(X_val_ens)

# Create ensemble with transformed models
best_models_transformed = {}
for name, model in best_models.items():
    # Extract the classifier from the pipeline
    classifier = model.named_steps['classifier']
    # Fit on transformed data
    classifier.fit(X_train_transformed, y_train_ens)
    best_models_transformed[name] = classifier

ensemble_model = create_ensemble_model(best_models_transformed, X_train_transformed, y_train_ens, X_val_transformed, y_val_ens)

# %%
"""
## 🎯 Final Predictions and Submission

Time to generate our final predictions for the Kaggle submission!

### Prediction Strategy:
1. **Two Submissions** - Best single model and ensemble
2. **Probability Analysis** - Understanding prediction confidence
3. **Agreement Analysis** - How often models agree
4. **Submission Format** - Proper Kaggle competition format

### Quality Checks:
- Survival rate should be reasonable (~30-40%)
- No missing predictions
- Proper passenger ID mapping
- Logical prediction patterns

### Files Generated:
- `titanic_submission_single_model.csv` - Best single model predictions
- `titanic_submission_ensemble.csv` - Ensemble model predictions  
- `prediction_comparison.csv` - Detailed prediction analysis

### Next Steps:
1. Submit both files to Kaggle
2. Compare public leaderboard scores
3. Analyze prediction differences
4. Iterate based on feedback
"""

# Final predictions with ensemble
def generate_final_predictions(best_pipeline, ensemble_model, X_test_final, test_df, preprocessor):
    """Generate final predictions using both best single model and ensemble"""
    
    print("=== GENERATING FINAL PREDICTIONS ===")
    
    # Single model predictions
    single_predictions = best_pipeline.predict(X_test_final)
    single_probabilities = best_pipeline.predict_proba(X_test_final)[:, 1]
    
    # Ensemble predictions
    X_test_transformed = preprocessor.transform(X_test_final)
    ensemble_predictions = ensemble_model.predict(X_test_transformed)
    ensemble_probabilities = ensemble_model.predict_proba(X_test_transformed)[:, 1]
    
    # Create submissions
    submission_single = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': single_predictions
    })
    
    submission_ensemble = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': ensemble_predictions
    })
    
    # Detailed comparison
    comparison = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Single_Model': single_predictions,
        'Single_Prob': single_probabilities,
        'Ensemble': ensemble_predictions,
        'Ensemble_Prob': ensemble_probabilities,
        'Agreement': (single_predictions == ensemble_predictions).astype(int)
    })
    
    agreement_rate = comparison['Agreement'].mean()
    
    print(f"Single Model - Predicted survivors: {single_predictions.sum()}")
    print(f"Single Model - Survival rate: {single_predictions.mean():.2%}")
    print(f"Ensemble Model - Predicted survivors: {ensemble_predictions.sum()}")
    print(f"Ensemble Model - Survival rate: {ensemble_predictions.mean():.2%}")
    print(f"Model Agreement Rate: {agreement_rate:.2%}")
    
    # Save submissions
    submission_single.to_csv("titanic_submission_single_model.csv", index=False)
    submission_ensemble.to_csv("titanic_submission_ensemble.csv", index=False)
    comparison.to_csv("prediction_comparison.csv", index=False)
    
    print("\nFiles saved:")
    print("- titanic_submission_single_model.csv")
    print("- titanic_submission_ensemble.csv")
    print("- prediction_comparison.csv")
    
    return submission_single, submission_ensemble, comparison

# Generate final predictions
test_with_id = test.copy()
submission_single, submission_ensemble, comparison = generate_final_predictions(
    best_pipeline, ensemble_model, X_test_final, test_with_id, preprocessor
)

print("\n=== ENHANCED TITANIC ANALYSIS WITH GRID SEARCH COMPLETED ===")


