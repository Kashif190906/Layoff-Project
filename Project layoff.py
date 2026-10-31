# LAYOFFS DATA ANALYSIS — ALL SIX PRACTICALS


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    silhouette_score
)


# ==========================================
# LOAD DATASET
# ==========================================

file_path = r"C:\Users\HP\Downloads\layoffs_dataset_from_screenshots.csv"

df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()

print("\n========== DATASET ==========")

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ==========================================
# PRACTICAL 1: EXPLORING AND HANDLING DATA
# ==========================================

print("\n========== PRACTICAL 1: DATA EXPLORATION ==========")

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nIndustry Counts:")
print(df["Industry"].value_counts(dropna=False))

print("\nHeadquarters Counts:")
print(df["Headquarters"].value_counts(dropna=False))

print("\nAnnouncement Year Counts:")
print(df["Announcement Year"].value_counts(dropna=False).sort_index())


# Industry-wise Bar Chart

plt.figure(figsize=(10, 5))

df["Industry"].fillna("Unknown").value_counts().plot(kind="bar")

plt.title("Layoff Announcements by Industry")
plt.xlabel("Industry")
plt.ylabel("Number of Announcements")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# Monthly Layoff Announcements for 2026

df["Announcement Year"] = pd.to_numeric(
    df["Announcement Year"], errors="coerce"
)

dates = pd.to_datetime(
    df["Announced Month/Year"],
    errors="coerce"
)

df["Month"] = dates.dt.month

# Fallback for month names when date parsing fails

month_mapping = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12
}

month_text = (
    df["Announced Month/Year"]
    .astype(str)
    .str.lower()
    .str.extract(
        r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sept|sep|october|oct|november|nov|december|dec)"
    )[0]
)

month_fallback = month_text.map(month_mapping)

df["Month"] = df["Month"].fillna(month_fallback)

month_names = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

monthly_data = df[df["Announcement Year"] == 2026].copy()

month_counts = (
    monthly_data["Month"]
    .value_counts()
    .reindex(range(1, 13), fill_value=0)
    .sort_index()
)

plt.figure(figsize=(12, 6))

plt.bar(month_names, month_counts.values)

plt.title("Monthly Layoff Announcements in 2026")
plt.xlabel("Months")
plt.ylabel("Number of Layoff Announcements")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ==========================================
# DATA CLEANING AND PREPARATION
# ==========================================

df["No. of Jobs"] = (
    df["No. of Jobs"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["No. of Jobs"] = pd.to_numeric(
    df["No. of Jobs"].replace({
        "": np.nan,
        "nan": np.nan,
        "None": np.nan
    }),
    errors="coerce"
)

df["Company"] = df["Company"].fillna("Unknown")
df["Industry"] = df["Industry"].fillna("Unknown")
df["Headquarters"] = df["Headquarters"].fillna("Unknown")

df["Month"] = df["Month"].fillna(1)

df["Announcement Year"] = df["Announcement Year"].fillna(
    df["Announcement Year"].median()
)

model_df = df.dropna(subset=["No. of Jobs"]).copy()

if len(model_df) < 5:
    raise ValueError(
        "Too few rows have numeric job counts for machine-learning practicals."
    )

model_df["Layoff Level"] = (
    model_df["No. of Jobs"] >= model_df["No. of Jobs"].median()
).astype(int)

print("\nRows in Original Dataset:", len(df))
print("Rows Used for Modelling:", len(model_df))
print("Rows Excluded Due to Missing Job Counts:", len(df) - len(model_df))


# ==========================================
# PRACTICAL 2: DATA PREPROCESSING
# ==========================================

print("\n========== PRACTICAL 2: DATA PREPROCESSING ==========")

print("\nMissing Values Before Preprocessing:")
print(model_df.isnull().sum())

numeric_features = ["Month", "Announcement Year"]

categorical_features = ["Industry", "Headquarters"]

X = model_df[numeric_features + categorical_features]

y_class = model_df["Layoff Level"]

y_reg = model_df["No. of Jobs"]


# Numerical Data Preprocessing

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical Data Preprocessing

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Combined Preprocessor

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])


# Train-Test Split

class_counts = y_class.value_counts()

if len(class_counts) < 2 or class_counts.min() < 2:
    raise ValueError(
        "Classification requires at least two classes with at least "
        "two rows in each class."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.25,
    random_state=42,
    stratify=y_class
)


X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessed Training Shape:", X_train_processed.shape)

print("Preprocessed Testing Shape:", X_test_processed.shape)

print("\nMissing Values After Cleaning:")
print(model_df.isnull().sum())


# ==========================================
# PRACTICAL 3: REGRESSION MODELLING
# ==========================================

print("\n========== PRACTICAL 3: REGRESSION MODELLING ==========")

reg_features = ["Month", "Announcement Year"]

X_reg = model_df[reg_features]

y = model_df["No. of Jobs"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg,
    y,
    test_size=0.25,
    random_state=42
)


# 3.1 SIMPLE LINEAR REGRESSION

simple_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", LinearRegression())
])

simple_model.fit(Xr_train[["Month"]], yr_train)

simple_predictions = simple_model.predict(Xr_test[["Month"]])

print("\n--- Simple Linear Regression ---")

print("R2 Score:", r2_score(yr_test, simple_predictions))

print("Mean Squared Error:", mean_squared_error(yr_test, simple_predictions))


# 3.2 MULTIPLE LINEAR REGRESSION

multiple_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", LinearRegression())
])

multiple_model.fit(Xr_train, yr_train)

multiple_predictions = multiple_model.predict(Xr_test)

print("\n--- Multiple Linear Regression ---")

print("R2 Score:", r2_score(yr_test, multiple_predictions))

print("Mean Squared Error:", mean_squared_error(yr_test, multiple_predictions))


# 3.3 POLYNOMIAL REGRESSION

polynomial_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("regressor", LinearRegression())
])

polynomial_model.fit(Xr_train, yr_train)

polynomial_predictions = polynomial_model.predict(Xr_test)

print("\n--- Polynomial Regression ---")

print("R2 Score:", r2_score(yr_test, polynomial_predictions))

print("Mean Squared Error:", mean_squared_error(yr_test, polynomial_predictions))


# 3.4 OLS REGRESSION

ols_data = model_df[reg_features + ["No. of Jobs"]].dropna()

X_ols = sm.add_constant(ols_data[reg_features])

y_ols = ols_data["No. of Jobs"]

ols_model = sm.OLS(y_ols, X_ols).fit()

print("\n--- OLS Regression Summary ---")

print(ols_model.summary())


# 3.5 CORRELATION ANALYSIS

correlation_data = model_df[
    ["Month", "Announcement Year", "No. of Jobs"]
].corr()

print("\n--- Correlation Matrix ---")

print(correlation_data)

plt.figure(figsize=(7, 5))

sns.heatmap(
    correlation_data,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()


# 3.6 LOGISTIC REGRESSION

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

print("\n--- Logistic Regression ---")

print("Accuracy:", accuracy_score(y_test, logistic_predictions))

print(classification_report(
    y_test,
    logistic_predictions,
    zero_division=0
))


# ==========================================
# PRACTICAL 4: KNN AND NAIVE BAYES
# ==========================================

print("\n========== PRACTICAL 4: KNN AND NAIVE BAYES ==========")


# 4.1 K-NEAREST NEIGHBORS

knn_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

print("\n--- K-Nearest Neighbors ---")

print("Accuracy:", accuracy_score(y_test, knn_predictions))

print(classification_report(
    y_test,
    knn_predictions,
    zero_division=0
))


# 4.2 GAUSSIAN NAIVE BAYES

# Dense output fixes the sparse-matrix error.

nb_preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]), categorical_features)
])

nb_model = Pipeline([
    ("preprocessor", nb_preprocessor),
    ("classifier", GaussianNB())
])

nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

print("\n--- Gaussian Naive Bayes ---")

print("Accuracy:", accuracy_score(y_test, nb_predictions))

print(classification_report(
    y_test,
    nb_predictions,
    zero_division=0
))


# ==========================================
# PRACTICAL 5: DECISION TREE AND SVM
# ==========================================

print("\n========== PRACTICAL 5: DECISION TREE AND SVM ==========")


# 5.1 DECISION TREE CLASSIFIER

tree_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        class_weight="balanced"
    ))
])

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

print("\n--- Decision Tree ---")

print("Accuracy:", accuracy_score(y_test, tree_predictions))

print(classification_report(
    y_test,
    tree_predictions,
    zero_division=0
))


# 5.2 SUPPORT VECTOR MACHINE

svm_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

print("\n--- Support Vector Machine ---")

print("Accuracy:", accuracy_score(y_test, svm_predictions))

print(classification_report(
    y_test,
    svm_predictions,
    zero_division=0
))


# CONFUSION MATRICES

for model_name, predictions in [
    ("Logistic Regression", logistic_predictions),
    ("KNN", knn_predictions),
    ("Naive Bayes", nb_predictions),
    ("Decision Tree", tree_predictions),
    ("SVM", svm_predictions)
]:

    print(f"\n--- {model_name} Confusion Matrix ---")

    print(confusion_matrix(y_test, predictions))


# ==========================================
# PRACTICAL 6: CLUSTERING
# ==========================================

print("\n========== PRACTICAL 6: CLUSTERING ==========")

cluster_features = model_df[
    ["Month", "Announcement Year", "No. of Jobs"]
].replace([np.inf, -np.inf], np.nan)

cluster_imputer = SimpleImputer(strategy="median")

cluster_scaler = StandardScaler()

cluster_values = cluster_imputer.fit_transform(cluster_features)

cluster_values = cluster_scaler.fit_transform(cluster_values)

unique_rows = np.unique(cluster_values, axis=0)

if len(unique_rows) >= 2:

    n_clusters = min(3, len(unique_rows))


    # 6.1 K-MEANS CLUSTERING

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    kmeans_labels = kmeans.fit_predict(cluster_values)

    model_df["KMeans Cluster"] = kmeans_labels

    print("\n--- K-Means Cluster Counts ---")

    print(model_df["KMeans Cluster"].value_counts().sort_index())

    if len(set(kmeans_labels)) > 1:

        print(
            "K-Means Silhouette Score:",
            silhouette_score(cluster_values, kmeans_labels)
        )


    # 6.2 HIERARCHICAL CLUSTERING

    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    hierarchical_labels = hierarchical.fit_predict(cluster_values)

    model_df["Hierarchical Cluster"] = hierarchical_labels

    print("\n--- Hierarchical Cluster Counts ---")

    print(
        model_df["Hierarchical Cluster"]
        .value_counts()
        .sort_index()
    )

    if len(set(hierarchical_labels)) > 1:

        print(
            "Hierarchical Silhouette Score:",
            silhouette_score(cluster_values, hierarchical_labels)
        )


    # 6.3 DENDROGRAM

    linkage_matrix = linkage(
        cluster_values,
        method="ward"
    )

    plt.figure(figsize=(12, 6))

    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=25
    )

    plt.title("Hierarchical Clustering Dendrogram")

    plt.xlabel("Cluster or Sample")

    plt.ylabel("Distance")

    plt.tight_layout()

    plt.show()


    # 6.4 PCA CLUSTER VISUALIZATION

    pca = PCA(n_components=2)

    cluster_2d = pca.fit_transform(cluster_values)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        cluster_2d[:, 0],
        cluster_2d[:, 1],
        c=kmeans_labels
    )

    plt.title("K-Means Clusters Using PCA")

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.tight_layout()

    plt.show()

else:

    print("Clustering requires at least two distinct data rows.")


# ==========================================
# SAVE RESULTS
# ==========================================

model_df.to_csv(
    "layoffs_all_practicals_results.csv",
    index=False
)

print("\n========== COMPLETED ==========")

print("All six practicals completed successfully.")

print("Results saved as layoffs_all_practicals_results.csv")

