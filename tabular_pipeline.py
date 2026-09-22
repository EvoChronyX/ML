import os, math
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    VarianceThreshold
)


sns.set(style="whitegrid")

print("Libraries installed successfully..")

def tabular_pipeline(
    file_path, 
    target_col=None, 
    task_type="auto", 
    method="hybrid", 
    k=20, 
    variance_threshold=0.0, 
    random_state=42,
    run_vif=True,
    run_univariate=True
):
    """
    Unified Tabular Data Pipeline: EDA, Preprocessing, Feature Engineering, and Feature Selection.
    """

    # =========================================================
    # 1. TABULAR EDA
    # =========================================================
    def perform_tabular_eda(file_path, target_col=None):
        print(f"\n[ROUTE] Successfully sent to Tabular EDA.")
        print("\n")
        print(f"-> Processing data file: '{os.path.basename(file_path)}'")
        print("\n\n")

        # Structural & Shape Assessment
        try:
            df = pd.read_csv(
                file_path,
                skipinitialspace=True,
                encoding="utf-8",
                on_bad_lines="skip",
                low_memory=False
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                file_path,
                encoding="latin1",
                on_bad_lines="skip",
                low_memory=False
            )

        empty_columns = df.columns[df.isna().all()].tolist()
        if empty_columns:
            df = df.drop(columns=empty_columns)
            print(f"Removed entirely empty columns: {empty_columns}")
        
        print(f"The shape of the dataset: {df.shape}")
        print(df.head())
        if df.shape[1] <= 50:
            print(df.tail())
            print(df.sample(5, random_state=42))

        print("\n")
        # Quality & Cleanliness
        missing = pd.DataFrame({
            "Missing": df.isnull().sum(),
            "Percentage": df.isnull().mean() * 100
        })

        if df.shape[1] <= 50:
            print(missing.sort_values("Percentage", ascending=False))
        else:
            print(f"Missing values: {int(missing['Missing'].sum())}")
        print(f"Duplicate Values: {df.duplicated().sum()}")
        
        if df.shape[1] <= 50:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False)
            plt.title('Missing Values Heatmap')
            plt.show()

        numeric_columns = df.select_dtypes(include=['number']).copy()
        numeric_columns = numeric_columns.loc[:, numeric_columns.notna().any()]

        categorical_cols = [
            col for col in df.select_dtypes(include=['category', 'str']).columns
            if df[col].nunique() <= 15
        ]

        high_cardinality_cols = [
            col for col in df.select_dtypes(include=['category', 'str']).columns
            if df[col].nunique() > 15
        ]

        if run_univariate and df.shape[1] <= 50:
            print("\n")
            # Univariate Analysis for numerical Columns
            print("Central Tendency: ")
            for col in df.select_dtypes(include=['number']).columns:
                mean_val = df[col].mean()
                median_val = df[col].median()
                mode_val = df[col].mode()
                mode_value = mode_val.iloc[0] if not mode_val.empty else np.nan
                skew_val = df[col].skew()
                kurtosis_val = df[col].kurt()
                print(f"\nColumn: '{col}'")
                print(f"  -> Mean:   {mean_val:.2f}")
                print(f"  -> Median: {median_val:.2f}")
                print(f"  -> Mode:   {mode_value}")
                print(f"  -> skew:   {skew_val}")
                print(f"  -> Kurtosis:   {kurtosis_val}")
                print(df[col].describe())

        print("\n")
        # Univariate Analysis for Categorical Columns
        small_cat_cols = categorical_cols[:20] if run_univariate and df.shape[1] <= 50 else []
        for col in small_cat_cols:
            print(f"\n--- Analysis for {col} ---")
            print("Frequency Counts:")
            print(df[col].value_counts())

            print("Percentages:")
            print(df[col].value_counts(normalize=True) * 100)

            # Visualization
            plt.figure(figsize=(6, 3))
            sns.countplot(data=df, x=col)
            plt.title(f'Univariate Analysis of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.show()

        print("\n")
        # Bivariate Analysis
        # ---- Correlation Matrix
        if not numeric_columns.empty:
            feature_numeric = numeric_columns.drop(columns=[target_col], errors='ignore')

            if target_col in df.columns and not feature_numeric.empty:
                target_values = pd.to_numeric(df[target_col], errors='coerce')
                if target_values.isna().all():
                    target_values = pd.Series(
                        pd.factorize(df[target_col])[0],
                        index=df.index,
                        dtype='float64'
                    )

                correlation_matrix = feature_numeric.assign(
                    **{target_col: target_values}
                ).corr()
                top_features = (
                    correlation_matrix[target_col]
                    .abs()
                    .drop(target_col)
                    .sort_values(ascending=False)
                    .head(10)
                    .index
                )
            else:
                top_features = feature_numeric.columns[:10]

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                feature_numeric[top_features].corr(),
                annot=True,
                cmap='coolwarm',
                fmt=".2f",
                linewidths=1.5,
                linecolor='yellow'
            )
            plt.title("Correlation Heatmap of Top 10 Numeric Features")
            plt.show()


        print('\n')
        # ---- MultiCollinearity
        if not run_vif:
            print("MultiCollinearity analysis skipped for this experiment.")
        elif len(numeric_columns.columns) > 1:
            vif_df = numeric_columns.fillna(numeric_columns.median())
            vif_df = vif_df.loc[:, vif_df.var() > 0]

            if vif_df.shape[1] > 1:
                max_vif_features = 50
                if vif_df.shape[1] > max_vif_features:
                    selected_vif_cols = (
                        vif_df.var()
                        .nlargest(max_vif_features)
                        .index
                    )
                    vif_df = vif_df[selected_vif_cols]
                    print(
                        f"VIF calculated for the {max_vif_features} highest-variance "
                        "numeric features for speed and stability."
                    )

                try:
                    # VIF equals the diagonal of the inverse feature correlation
                    # matrix. Pseudo-inverse handles collinear features without
                    # repeatedly fitting unstable OLS models.
                    correlation = vif_df.corr().to_numpy()
                    vif_values = np.diag(np.linalg.pinv(correlation))
                    vif_data = pd.DataFrame({
                        'features': vif_df.columns,
                        'VIF': np.maximum(vif_values, 1.0)
                    })
                    print(vif_data.sort_values(by="VIF", ascending=False).head(20))
                except (ValueError, np.linalg.LinAlgError) as error:
                    print(f"Could not calculate MultiCollinearity: {error}")
            else:
                print("Not enough variable numeric fields to compute MultiCollinearity.")
        else:
            print("Not enough numeric fields to compute MultiCollinearity.")

        print("\n")
        plot_columns = numeric_columns.columns if run_univariate and df.shape[1] <= 50 else []
        for col in plot_columns:
            values = pd.to_numeric(numeric_columns[col], errors='coerce').dropna()
            if values.empty:
                continue

            plt.figure(figsize=(12, 4))

            # Histogram
            plt.subplot(1, 2, 1)
            sns.histplot(values, kde=True)
            plt.title(f'{col} Distribution')

            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=values.to_numpy())
            plt.title(f'{col} Boxplot')

            plt.tight_layout()
            plt.show()

        return df

    # =========================================================
    # 2. TABULAR PREPROCESSING
    # =========================================================
    def tabular_preprocessing(df):
        df = df.copy()

        # -----------------------------
        # 1. Type Handling
        # -----------------------------
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    df[col] = df[col].astype('category')

        # -----------------------------
        # 2. Missing Value Handling
        # -----------------------------
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['category']).columns

        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")

        # -----------------------------
        # 3. Outlier Capping (IQR)
        # -----------------------------
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            df[col] = np.clip(df[col], lower, upper)

        # -----------------------------
        # 4. Encoding (One-Hot for low cardinality)
        # -----------------------------
        low_card_cols = [col for col in cat_cols if df[col].nunique() <= 10]

        df = pd.get_dummies(df, columns=low_card_cols, drop_first=True)

        remaining_cat_cols = df.select_dtypes(include=['category', 'object']).columns
        if len(remaining_cat_cols) > 0:
            df = df.drop(columns=remaining_cat_cols)

        # -----------------------------
        # 5. Scaling (Standardization)
        # -----------------------------
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        print("Preprocessing Completed")
        return df, scaler

    # =========================================================
    # 3. FEATURE ENGINEERING
    # =========================================================
    def tabular_feature_engineering(df):
        df = df.copy()

        num_cols = df.select_dtypes(include=['number']).columns

        # -----------------------------
        # 1. Log Transform (Skew Handling)
        # -----------------------------
        for col in num_cols:
            if df[col].skew() > 1 and (df[col] >= 0).all():
                df[col] = np.log1p(df[col])

        # -----------------------------
        # 2. Interaction Features
        # -----------------------------
        if len(num_cols) >= 2:
            df["interaction_feature"] = df[num_cols[0]] * df[num_cols[1]]

        # -----------------------------
        # 3. Binning Example (if applicable)
        # -----------------------------
        for col in num_cols[:2]:  # limit to avoid explosion
            df[f"{col}_bin"] = pd.qcut(df[col], q=4, labels=False, duplicates='drop')

        # -----------------------------
        # 4. Remove Highly Correlated Features
        # -----------------------------
        corr_matrix = df.select_dtypes(include=['number']).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        drop_cols = [col for col in upper.columns if any(upper[col] > 0.9)]
        df.drop(columns=drop_cols, inplace=True)

        print(" Feature Engineering Completed")
        return df

    # =========================================================
    # 4. FEATURE SELECTION PIPELINE
    # =========================================================
    def feature_selection_pipeline(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "auto",
        method: str = "hybrid",
        k: int = 20,
        variance_threshold: float = 0.0,
        random_state: int = 42
    ):
        if k <= 0:
            k = min(20, X.shape[1])

        # --- Step 1: Infer Task Type ---
        if task_type == "auto":
            if y.nunique() < 20 and y.dtype in ["int64", "object", "category"]:
                task_type = "classification"
            else:
                task_type = "regression"

        # --- Step 2: Variance Threshold ---
        vt = VarianceThreshold(threshold=variance_threshold)
        X_vt = vt.fit_transform(X)
        retained_cols = X.columns[vt.get_support()]
        X = pd.DataFrame(X_vt, columns=retained_cols)

        # --- Step 3: Filter Methods ---
        if method in ["filter", "hybrid"]:
            if task_type == "classification":
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression

            selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
            X_filtered = selector.fit_transform(X, y)
            selected_cols_filter = X.columns[selector.get_support()]

        # --- Step 4: Embedded Methods ---
        if method in ["embedded", "hybrid"]:
            if task_type == "classification":
                model = RandomForestClassifier(random_state=random_state)
            else:
                model = RandomForestRegressor(random_state=random_state)

            model.fit(X, y)
            importances = pd.Series(model.feature_importances_, index=X.columns)

            selected_cols_embedded = importances.nlargest(min(k, len(importances))).index

        # --- Step 5: Combine Methods ---
        if method == "filter":
            selected_features = list(selected_cols_filter)
        elif method == "embedded":
            selected_features = list(selected_cols_embedded)
        else:  # hybrid
            selected_features = list(
                set(selected_cols_filter).intersection(set(selected_cols_embedded))
            )
            if len(selected_features) < max(5, k // 4):
                selected_features = list(
                    set(selected_cols_filter).union(set(selected_cols_embedded))
                )[:k]

        X_selected = X[selected_features]
        return selected_features, X_selected

    # =========================================================
    # 5. PIPELINE EXECUTION
    # =========================================================
    print("\n[PIPELINE] Starting Tabular Data Processing Pipeline...\n")

    raw_df = perform_tabular_eda(file_path, target_col=target_col)

    if target_col and target_col in raw_df.columns:
        y = raw_df[target_col]
        feature_df = raw_df.drop(columns=[target_col])
    else:
        y = None
        feature_df = raw_df

    preprocessed_df, scaler = tabular_preprocessing(feature_df)

    engineered_df = tabular_feature_engineering(preprocessed_df).fillna(0)

    # Split feature matrix (X) and target variable (y) if target_col specified
    if y is not None:
        X = engineered_df
        
        selected_features, X_selected = feature_selection_pipeline(
            X, y,
            task_type=task_type,
            method=method,
            k=k,
            variance_threshold=variance_threshold,
            random_state=random_state
        )
    else:
        print("\n[INFO] Target column omitted or not found — skipping feature selection.")
        X_selected = engineered_df
        selected_features = list(engineered_df.columns)

    print("\n[PIPELINE COMPLETE]")
    print(f"Final Feature Shape: {X_selected.shape}")

    return {
        "X": X_selected,
        "y": y,
        "scaler": scaler,
        "selected_features": selected_features
    }