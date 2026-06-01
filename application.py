from flask import Flask, render_template, request
import pandas as pd
import os
import pickle

# try to load trained model once
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'LinearRegressionModel.pkl')
MODEL = None
MODEL_LOAD_ERROR = None
if os.path.exists(MODEL_FILE):
    try:
        with open(MODEL_FILE, 'rb') as f:
            MODEL = pickle.load(f)
            print(f"[debug] Loaded model from {MODEL_FILE}")
    except Exception as e:
        MODEL_LOAD_ERROR = str(e)
        print(f"[debug] Failed to load model: {MODEL_LOAD_ERROR}")

app = Flask(__name__)

# Load dataset robustly
DATA_FILE = os.path.join(os.path.dirname(__file__), 'Cleaned car.csv')
if os.path.exists(DATA_FILE):
    car = pd.read_csv(DATA_FILE)
    car.columns = car.columns.str.strip()
else:
    car = pd.DataFrame()


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].dropna().unique()) if not car.empty and 'company' in car.columns else []
    car_models = sorted(car['name'].dropna().unique()) if not car.empty and 'name' in car.columns else []
    years = sorted(car['year'].dropna().unique(), reverse=True) if not car.empty and 'year' in car.columns else []
    fuel_type = sorted(car['fuel_type'].dropna().unique()) if not car.empty and 'fuel_type' in car.columns else []

    # build mapping from company -> models for client-side filtering
    models_by_company = {}
    if not car.empty and 'company' in car.columns and 'name' in car.columns:
        for comp, grp in car.groupby('company'):
            models_by_company[comp] = sorted(grp['name'].dropna().unique().tolist())

    predicted_price = None
    error = None
    fallback_info = None

    if request.method == 'POST':
        # collect form inputs
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = request.form.get('year')
        fuel = request.form.get('fuel_type')
        kilo = request.form.get('kilo_driven') or 0

        # coerce numeric fields
        try:
            year_val = int(year) if year else None
        except Exception:
            year_val = None
        try:
            kms_val = int(kilo)
        except Exception:
            try:
                kms_val = int(float(kilo))
            except Exception:
                kms_val = 0

        # Require at least one of company/car_model/year to attempt prediction
        if not (company or car_model or year):
            error = 'Please select at least a company, model, or year before predicting.'
        else:
            if MODEL is None:
                # fallback: try to estimate using historical median price from CSV using best available match
                if not car.empty and 'Price' in car.columns:
                    subset = car.copy()
                    # prefer strongest matching criteria in order
                    # 1) model + year + fuel
                    if car_model and year and fuel:
                        s = subset[(subset['name'] == car_model) & (subset['year'] == int(year)) & (subset['fuel_type'] == fuel)]
                        method = 'model+year+fuel'
                    else:
                        s = pd.DataFrame()
                        method = None

                    # 2) model + year
                    if s.empty and car_model and year:
                        s = subset[(subset['name'] == car_model) & (subset['year'] == int(year))]
                        method = 'model+year'

                    # 3) model only
                    if s.empty and car_model:
                        s = subset[subset['name'] == car_model]
                        method = 'model'

                    # 4) company + model
                    if s.empty and company and car_model:
                        s = subset[(subset['company'] == company) & (subset['name'] == car_model)]
                        method = 'company+model'

                    # 5) company only
                    if s.empty and company:
                        s = subset[subset['company'] == company]
                        method = 'company'

                    if not s.empty:
                        try:
                            predicted_price = float(s['Price'].median())
                            fallback_info = {'method': method, 'rows': int(len(s))}
                        except Exception as e:
                            error = f'No model and fallback estimate failed: {e}'
                    else:
                        error = 'Prediction model not available and no historical data to estimate price.'
                else:
                    # surface model load error if present
                    if MODEL_LOAD_ERROR:
                        error = f'Model failed to load: {MODEL_LOAD_ERROR}'
                    else:
                        error = 'Prediction model not found on server.'
            else:
                # build a single-row DataFrame for the model
                input_row = {}
                input_row['name'] = car_model
                input_row['company'] = company
                input_row['year'] = year_val
                # CSV uses 'kms_driven' column name
                input_row['kms_driven'] = kms_val
                input_row['fuel_type'] = fuel

                X = pd.DataFrame([input_row])
                try:
                    # If model expects named features
                    if hasattr(MODEL, 'feature_names_in_'):
                        cols = list(MODEL.feature_names_in_)
                        # reindex to expected columns (fill missing with defaults)
                        Xp = X.reindex(columns=cols, fill_value=0)
                        ypred = MODEL.predict(Xp)
                    else:
                        # try predicting directly with DataFrame, otherwise numpy array
                        try:
                            ypred = MODEL.predict(X)
                        except Exception:
                            ypred = MODEL.predict(X.values)

                    predicted_price = float(ypred[0])
                except Exception as e:
                    error = f'Prediction failed: {e}'

    print(f"[debug] companies_count={len(companies)}")
    # only show results after a POST attempt
    show_result = (request.method == 'POST') and (predicted_price is not None or error is not None)
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_type=fuel_type, models_by_company=models_by_company, predicted_price=predicted_price, error=error, fallback_info=fallback_info, show_result=show_result)


if __name__ == '__main__':
    app.run(debug=True)