import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. SYNTHETIC AGRICULTURAL DATASET
# ==========================================
np.random.seed(42)
data = pd.DataFrame({
    'crop_type': ['Wheat', 'Rice', 'Maize'] * 40,
    'rainfall': np.random.uniform(200, 1000, 120),
    'temperature': np.random.uniform(15, 35, 120),
    'humidity': np.random.uniform(40, 80, 120),
    'soil_ph': np.random.uniform(5.5, 7.5, 120),
    'crop_yield': np.random.uniform(1.5, 5.0, 120),
})

X = data.drop(columns=['crop_yield'])
y = data['crop_yield']

# ==========================================
# 2. TRAIN-TEST SPLIT (80/20)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================
preprocessor = ColumnTransformer([
    (
        'num',
        StandardScaler(),
        ['rainfall', 'temperature', 'humidity', 'soil_ph'],
    ),
    ('cat', OneHotEncoder(drop='first'), ['crop_type']),
])

# ==========================================
# 4. BASELINE EVALUATION (MAE, RMSE, R2)
# ==========================================
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge(alpha=1.0)),
    ('Random Forest (Default)', RandomForestRegressor(random_state=42)),
]

print('=' * 65)
print('WEEK 5 MODEL EVALUATION RESULTS')
print('=' * 65)

for name, model in models:
  pipe = Pipeline([('prep', preprocessor), ('model', model)])
  pipe.fit(X_train, y_train)
  preds = pipe.predict(X_test)

  mae = mean_absolute_error(y_test, preds)
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  r2 = r2_score(y_test, preds)

  print(
      f'{name:24} | MAE: {mae:.2f} t/ha | RMSE: {rmse:.2f} t/ha | R2:'
      f' {r2:.2f}'
  )

# ==========================================
# 5. CROSS-VALIDATION (5-FOLD)
# ==========================================
print('\n' + '=' * 65)
print('5-FOLD CROSS-VALIDATION (MAE)')
print('=' * 65)

for name, model in models:
  pipe = Pipeline([('prep', preprocessor), ('model', model)])
  cv_scores = -cross_val_score(
      pipe, X_train, y_train, cv=5, scoring='neg_mean_absolute_error'
  )
  print(f'{name:24} | 5-Fold Mean MAE: {cv_scores.mean():.2f} t/ha')

# ==========================================
# 6. OPTIMIZATION: HYPERPARAMETER TUNING
# ==========================================
print('\n' + '=' * 65)
print('GRID SEARCH HYPERPARAMETER TUNING (RANDOM FOREST)')
print('=' * 65)

rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor(random_state=42)),
])

param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5, 10],
    'model__min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
opt_preds = best_model.predict(X_test)

opt_mae = mean_absolute_error(y_test, opt_preds)
opt_rmse = np.sqrt(mean_squared_error(y_test, opt_preds))
opt_r2 = r2_score(y_test, opt_preds)

print(f'Best Parameters: {grid_search.best_params_}')
print(
    f'Optimized Random Forest  | MAE: {opt_mae:.2f} t/ha | RMSE:'
    f' {opt_rmse:.2f} t/ha | R2: {opt_r2:.2f}'
)
