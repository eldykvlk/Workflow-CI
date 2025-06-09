import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():



    # Nama eksperimen MLflow
    experiment_name = "Walmart Sales Prediction Basic"
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow experiment set to: {experiment_name}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID: {run_id}")

        # Mengaktifkan autologging untuk Scikit-learn
        mlflow.sklearn.autolog()
        logging.info("MLflow autologging for scikit-learn enabled.")

        try:
            # Memuat dataset yang telah dipreprocessing
            df = pd.read_csv('Walmart_Sales_preprocessing.csv')
            logging.info("Dataset 'Walmart_Sales_preprocessing.csv' loaded successfully.")
        except FileNotFoundError:
            logging.error("Error: 'Walmart_Sales_preprocessing.csv' not found. Make sure it's in the same directory as modelling.py.")
            return
        except Exception as e:
            logging.error(f"An error occurred while loading the dataset: {e}")
            return

        # Feature Engineering (ulangi seperti di preprocessing Anda)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df = df.drop('Date', axis=1) 
            logging.info("Date column processed for Month and Year features.")
        else:
            logging.warning("Date column not found. Ensure preprocessing output includes it if needed for feature engineering.")

        # Create lagged features - Perlu grouping by 'Store'
        if 'Store' in df.columns and 'Weekly_Sales' in df.columns:
            df['Weekly_Sales_Lag1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
            df['Weekly_Sales_Lag2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
            df['Weekly_Sales_Lag4'] = df.groupby('Store')['Weekly_Sales'].shift(4)

            # Fill NaN values with backward fill per group
            for col in ['Weekly_Sales_Lag1', 'Weekly_Sales_Lag2', 'Weekly_Sales_Lag4']:
                df[col] = df.groupby('Store')[col].bfill()
            logging.info("Lagged sales features created and NaNs filled.")
        else:
            logging.warning("Store or Weekly_Sales column not found for lagged feature creation. Skipping this step.")

        # Create interaction terms
        if 'Temperature' in df.columns and 'Fuel_Price' in df.columns:
            df['Temperature_Fuel_Interaction'] = df['Temperature'] * df['Fuel_Price']
            logging.info("Temperature_Fuel_Interaction feature created.")
        else:
            logging.warning("Temperature or Fuel_Price column not found for interaction feature creation. Skipping this step.")

        # Drop rows with any remaining NaN values (e.g., from lagged features at the beginning of series)
        df.dropna(inplace=True)
        logging.info(f"NaN values dropped. Dataset shape after dropping NaNs: {df.shape}")

        # Definisikan fitur (X) dan variabel target (y)
        features = [col for col in df.columns if col not in ['Weekly_Sales']]
        X = df[features]
        y = df['Weekly_Sales']
        logging.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        logging.info(f"Features used: {features}")

        # Bagi data menjadi training dan testing sets
        if 'Store' in df.columns:
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Store'])
             logging.info("Data split into training and testing sets with stratification on 'Store' column.")
        else:
             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
             logging.warning("Store column not found for stratification. Data split without stratification.")

        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Inisialisasi dan latih model DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
        logging.info("DecisionTreeRegressor model initialized.")
        model.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Buat prediksi pada data uji
        y_pred = model.predict(X_test)
        logging.info("Predictions made on test data.")

        # Evaluasi model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # Tambahkan RMSE
        r2 = r2_score(y_test, y_pred)

        # Log metrik menggunakan MLflow (autolog akan melakukannya, tapi ini untuk eksplisit)
        mlflow.log_metrics({"mae": mae, "mse": mse, "rmse": rmse, "r2_score": r2})
        logging.info(f"Model Metrics: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, R2 Score={r2:.2f}")

        # Log model (autolog sudah melakukannya, tapi ini untuk eksplisit)
        # mlflow.sklearn.log_model(model, "decision_tree_model")
        logging.info("Model and metrics logged to MLflow.")

        logging.info("Training process finished.")

if __name__ == "__main__":
    import numpy as np
    train_model()
