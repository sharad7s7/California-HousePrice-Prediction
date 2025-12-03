import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("1. Loading data...")
housing = pd.read_csv("housing.csv")
housing['income_cat'] = pd.cut(housing["median_income"],
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

housing_train = strat_train_set.copy()
train_labels = housing_train["median_house_value"].copy()
train_features = housing_train.drop("median_house_value", axis=1)

cat_attribs = ["ocean_proximity"]
num_attribs = train_features.drop(columns=cat_attribs).columns.tolist()

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

print("2. Training RandomForest model...")
housing_prepared = full_pipeline.fit_transform(train_features)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_reg.fit(housing_prepared, train_labels)

print("3. Generating Advanced Graph Data...")

cat_encoder = full_pipeline.named_transformers_["cat"]["onehot"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + cat_one_hot_attribs
importances = rf_reg.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': attributes, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

test_features = strat_test_set.drop("median_house_value", axis=1)
test_labels = strat_test_set["median_house_value"].copy()
test_prepared = full_pipeline.transform(test_features)
test_predictions = rf_reg.predict(test_prepared)

test_results_df = pd.DataFrame({
    "Actual": test_labels,
    "Predicted": test_predictions
}).sample(300)

print("4. Saving artifacts...")
joblib.dump(full_pipeline, "pipeline.pkl")
joblib.dump(rf_reg, "model.pkl")
joblib.dump(housing_train, "training_data.pkl")
joblib.dump(feature_importance_df, "feature_importance.pkl") 
joblib.dump(test_results_df, "test_results.pkl")       

print("Done! New files: feature_importance.pkl, test_results.pkl")