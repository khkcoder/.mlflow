import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def print_divider(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

# --- Part 1 (Client) & 2: Init Client & Search Experiments ---
print_divider("Part 1 & 2: Init Client & View All Experiments")
client = MlflowClient()
print(f"MLflow client is configured to track at: {client.tracking_uri}")

all_experiments = client.search_experiments()
print("Searching all experiments...")
for exp in all_experiments:
    print("  Name: {}".format(exp.name))
    print("  Experiment ID: {}".format(exp.experiment_id))
    print("  Artifact Location: {}".format(exp.artifact_location))
    print("  Lifecycle Stage: {}".format(exp.lifecycle_stage))
    print("  Tags: {}".format(exp.tags))
    print("---")


# --- Part 3: Display Default Experiment Details ---
print_divider("Part 3: Default Experiment Details")
try:
    # The default experiment always has ID '0'
    default_exp = client.get_experiment("0")
    print(f"Default Experiment Name: {default_exp.name}")
    print(f"Default Lifecycle Stage: {default_exp.lifecycle_stage}")
except Exception as e:
    print(f"Could not find default experiment '0': {e}")


# --- Part 4: Create & Search 'Apples' Experiment ---
print_divider("Part 4: Create & Search 'Apples' Experiment")
APPLE_EXP_NAME = "apples_experiment"
exp_tags = {
    "project_name": "apple_demand_forecast",
    "team": "data_science_alpha"
}

try:
    exp_id = client.create_experiment(APPLE_EXP_NAME, tags=exp_tags)
    print(f"Successfully created experiment '{APPLE_EXP_NAME}' with ID: {exp_id}")
except mlflow.exceptions.MlflowException:
    print(f"Experiment '{APPLE_EXP_NAME}' already exists. Setting it.")
    exp_id = client.get_experiment_by_name(APPLE_EXP_NAME).experiment_id

# Use search_experiments() to find it by tag
filter_string = "tags.project_name = 'apple_demand_forecast'"
print(f"\nSearching for experiments with filter: '{filter_string}'")

found_experiments = client.search_experiments(
    filter_string=filter_string, 
    view_type=ViewType.ACTIVE_ONLY
)

for exp in found_experiments:
    print(f"Found experiment: {exp.name} (ID: {exp.experiment_id})")
    print(f"Tags: {exp.tags}")


# --- Part 5: Generate Synthetic Apple Sales Dataset ---
print_divider("Part 5: Generating Synthetic Dataset")
np.random.seed(42)
num_days = 365 * 3  # 3 years of data
dates = pd.date_range(start='2022-01-01', periods=num_days)

# Create seasonality (sine wave for the year)
day_of_year = dates.dayofyear
seasonality = 100 * (1 + np.sin(2 * np.pi * (day_of_year - 80) / 365.25))

# Create inflation/trend (slowly increasing base)
inflation = 0.05 * np.arange(num_days)

# Create noise
noise = np.random.normal(0, 10, num_days)

# Combine into 'sales'
base_sales = 50
sales = base_sales + seasonality + inflation + noise
sales = np.maximum(sales, 0) # Sales can't be negative

# Create DataFrame
data = pd.DataFrame({
    'date': dates,
    'day_of_year': day_of_year,
    'days_since_start': np.arange(num_days),
    'sales': sales
})
data = data.set_index('date')
print(f"Dataset created with {len(data)} rows.")
print(data.head())


# --- Part 6: Train and Log the Model ---
print_divider("Part 6: Train and Log Model")

# Set the experiment we want to log to
mlflow.set_experiment(APPLE_EXP_NAME)

# Define features (X) and target (y)
features = ['day_of_year', 'days_since_start']
target = 'sales'
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)

# Start an MLflow Run
with mlflow.start_run() as run:
    print(f"\nStarted run with ID: {run.info.run_id}")

    # 1. Log parameters
    model_params = {"model_type": "LinearRegression", "fit_intercept": True}
    mlflow.log_params(model_params)
    print(f"Logged params: {model_params}")

    # 2. Train the model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)

    # 3. Make predictions and log metrics
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    metrics = {"r2": r2, "mse": mse}
    mlflow.log_metrics(metrics)
    print(f"Logged metrics: {metrics}")

    # 4. Log tags
    mlflow.set_tag("run_purpose", "initial_baseline")

    # 5. Log the model
    mlflow.sklearn.log_model(model, "model")
    print("Model has been logged.")

print("\nAssignment script finished.")
print(f"Check the MLflow UI to see your new experiment and run: http://127.0.0.1:5000")