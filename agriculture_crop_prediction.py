import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = "agriculture_data.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n==============================================")
print(" AGRICULTURE CROP PRODUCTION PREDICTION")
print("==============================================\n")


if not os.path.exists(DATA_FILE):
    print("ERROR: agriculture_data.csv was not found.")
    print("Place agriculture_data.csv in the same folder as this Python file.")
    exit()


df = pd.read_csv(DATA_FILE)


print("Dataset loaded successfully.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)


print("\nColumns:")
for column in df.columns:
    print("-", column)


target_candidates = [
    "production",
    "crop_production",
    "crop_production_tonnes",
    "production_tonnes",
    "yield",
    "crop_yield"
]


target = None

for column in target_candidates:
    if column in df.columns:
        target = column
        break


if target is None:

    production_columns = [
        column
        for column in df.columns
        if "production" in column or "yield" in column
    ]

    if production_columns:
        target = production_columns[0]


if target is None:

    print("\nERROR: Production column was not detected.")
    print("Available columns:")
    print(list(df.columns))
    print("\nPlease rename the target column to 'Production'.")
    exit()


print("\nTarget column:", target)


df = df.drop_duplicates()

df[target] = pd.to_numeric(
    df[target],
    errors="coerce"
)

df = df.dropna(
    subset=[target]
)


if len(df) < 5:

    print("\nERROR: At least 5 valid records are required.")
    print("Current valid records:", len(df))
    exit()


print("\nMissing Values:")
print(df.isnull().sum())


numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


numeric_features = [
    column
    for column in numeric_columns
    if column != target
]


categorical_features = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()


categorical_features = [
    column
    for column in categorical_features
    if column != target
]


features = numeric_features + categorical_features


if not features:

    print("\nERROR: No usable features were detected.")
    exit()


X = df[features]

y = df[target]


print("\nSelected Features:")

for feature in features:
    print("-", feature)


print("\nTarget Statistics:")
print(y.describe())


plt.figure(figsize=(9, 5))

plt.hist(
    y,
    bins=min(10, len(y))
)

plt.xlabel("Crop Production")
plt.ylabel("Frequency")
plt.title("Distribution of Crop Production")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "production_distribution.png"
    )
)

plt.close()


if len(numeric_features) >= 2:

    correlation = df[
        numeric_features + [target]
    ].corr()

    plt.figure(figsize=(10, 7))

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Feature Correlation")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_correlation.png"
        )
    )

    plt.close()


if len(numeric_features) > 0:

    first_feature = numeric_features[0]

    plt.figure(figsize=(9, 5))

    plt.scatter(
        df[first_feature],
        y
    )

    plt.xlabel(first_feature)
    plt.ylabel("Crop Production")

    plt.title(
        f"{first_feature} vs Crop Production"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{first_feature}_vs_production.png"
        )
    )

    plt.close()


print("\nPreparing training and testing data...")


test_size = 0.40


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)


numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


transformers = []


if numeric_features:

    transformers.append(
        (
            "numeric",
            numeric_transformer,
            numeric_features
        )
    )


if categorical_features:

    transformers.append(
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers
)


linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


models = {
    "Linear Regression": linear_model,
    "Random Forest": random_forest_model
}


results = []

predictions = {}


print("\n==============================================")
print(" MODEL TRAINING")
print("==============================================\n")


for model_name, model in models.items():

    print("Training:", model_name)

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    predictions[model_name] = prediction


    mae = mean_absolute_error(
        y_test,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    if len(y_test) >= 2:

        r2 = r2_score(
            y_test,
            prediction
        )

    else:

        r2 = 0.0


    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2
        }
    )


    print(
        "MAE:",
        round(mae, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "R2 Score:",
        round(r2, 4)
    )

    print()


results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


print("==============================================")
print(" MODEL COMPARISON")
print("==============================================")

print(
    results_df.to_string(
        index=False
    )
)


best_model_name = results_df.loc[
    results_df["R2 Score"].idxmax(),
    "Model"
]


best_model = models[
    best_model_name
]


best_prediction = predictions[
    best_model_name
]


print(
    "\nBest Model:",
    best_model_name
)


plt.figure(figsize=(9, 6))


plt.scatter(
    y_test,
    best_prediction,
    alpha=0.7
)


minimum = min(
    y_test.min(),
    best_prediction.min()
)


maximum = max(
    y_test.max(),
    best_prediction.max()
)


if minimum != maximum:

    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )


plt.xlabel(
    "Actual Production"
)

plt.ylabel(
    "Predicted Production"
)

plt.title(
    f"Actual vs Predicted - {best_model_name}"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted.png"
    )
)


plt.close()


comparison = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": best_prediction
    }
)


comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "prediction_results.csv"
    ),
    index=False
)


print("\n==============================================")
print(" SAMPLE PREDICTIONS")
print("==============================================\n")


sample_count = min(
    5,
    len(X_test)
)


sample_data = X_test.iloc[
    :sample_count
]


sample_predictions = best_model.predict(
    sample_data
)


for index in range(
    sample_count
):

    print(
        f"Sample {index + 1}: "
        f"{sample_predictions[index]:.2f}"
    )


summary = f"""
Agriculture Crop Production Prediction
--------------------------------------

Dataset Records       : {len(df)}
Features Used         : {len(features)}
Training Records      : {len(X_train)}
Testing Records       : {len(X_test)}
Target Column         : {target}

Models Tested:
1. Linear Regression
2. Random Forest

Best Model:
{best_model_name}

Model Performance:

{results_df.to_string(index=False)}

Generated Outputs:
1. production_distribution.png
2. feature_correlation.png
3. actual_vs_predicted.png
4. model_comparison.csv
5. prediction_results.csv
6. project_results.txt
"""


with open(
    os.path.join(
        OUTPUT_DIR,
        "project_results.txt"
    ),
    "w",
    encoding="utf-8"
) as file:

    file.write(
        summary
    )


print("\n==============================================")
print(" PROJECT COMPLETED SUCCESSFULLY")
print("==============================================")

print(
    "\nResults saved inside the 'outputs' folder."
)

print("\nGenerated files:")

print(
    "- production_distribution.png"
)

if len(numeric_features) >= 2:

    print(
        "- feature_correlation.png"
    )

if len(numeric_features) > 0:

    print(
        f"- {numeric_features[0]}_vs_production.png"
    )

print(
    "- actual_vs_predicted.png"
)

print(
    "- model_comparison.csv"
)

print(
    "- prediction_results.csv"
)

print(
    "- project_results.txt"
)

print("\nThank you.")