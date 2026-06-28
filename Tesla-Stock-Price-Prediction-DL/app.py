import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="TSLA",
    layout="wide",
)


HORIZONS = [1, 5, 10]
DEFAULT_LOOKBACK = 60
REQUIRED_DATE_COLUMN = "Date"


@dataclass
class TrainingResult:
    model_name: str
    model: object
    history: object
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: pd.DataFrame


def load_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam

        return {
            "tf": tf,
            "Sequential": Sequential,
            "SimpleRNN": SimpleRNN,
            "LSTM": LSTM,
            "Dense": Dense,
            "Dropout": Dropout,
            "Adam": Adam,
            "EarlyStopping": EarlyStopping,
            "ReduceLROnPlateau": ReduceLROnPlateau,
        }
    except Exception as exc:
        st.error(
            "TensorFlow is required to train SimpleRNN and LSTM models. "
            "Install it with: pip install tensorflow"
        )
        st.exception(exc)
        st.stop()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    return pd.read_csv(io.BytesIO(raw))


def prepare_stock_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = clean_columns(df)
    if REQUIRED_DATE_COLUMN not in df.columns:
        raise ValueError("CSV must contain a Date column.")
    if target_col not in df.columns:
        raise ValueError(f"CSV must contain the selected target column: {target_col}")

    prepared = df.copy()
    prepared[REQUIRED_DATE_COLUMN] = pd.to_datetime(prepared[REQUIRED_DATE_COLUMN], errors="coerce")
    prepared[target_col] = pd.to_numeric(prepared[target_col], errors="coerce")
    prepared = prepared.dropna(subset=[REQUIRED_DATE_COLUMN])
    prepared = prepared.sort_values(REQUIRED_DATE_COLUMN)
    prepared = prepared.drop_duplicates(subset=[REQUIRED_DATE_COLUMN], keep="last")
    prepared = prepared.set_index(REQUIRED_DATE_COLUMN)

    missing_before = int(prepared[target_col].isna().sum())
    prepared[target_col] = prepared[target_col].ffill().bfill()
    if prepared[target_col].isna().any():
        raise ValueError("Target column still has missing values after forward/backward fill.")

    prepared.attrs["missing_target_values_filled"] = missing_before
    return prepared


def create_multi_horizon_sequences(
    scaled_values: np.ndarray,
    lookback: int,
    horizons: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_horizon = max(horizons)
    x_values, y_values, target_positions = [], [], []

    for end_index in range(lookback, len(scaled_values) - max_horizon + 1):
        x_values.append(scaled_values[end_index - lookback : end_index])
        y_values.append([scaled_values[end_index + horizon - 1, 0] for horizon in horizons])
        target_positions.append(end_index)

    return np.array(x_values), np.array(y_values), np.array(target_positions)


def chronological_train_test_split(
    x_values: np.ndarray,
    y_values: np.ndarray,
    target_positions: np.ndarray,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_index = max(1, int(len(x_values) * (1 - test_size)))
    split_index = min(split_index, len(x_values) - 1)

    return (
        x_values[:split_index],
        x_values[split_index:],
        y_values[:split_index],
        y_values[split_index:],
        target_positions[split_index:],
    )


def build_model(
    model_type: str,
    lookback: int,
    units: int,
    dropout_rate: float,
    learning_rate: float,
    tf_parts: Dict[str, object],
):
    Sequential = tf_parts["Sequential"]
    SimpleRNN = tf_parts["SimpleRNN"]
    LSTM = tf_parts["LSTM"]
    Dense = tf_parts["Dense"]
    Dropout = tf_parts["Dropout"]
    Adam = tf_parts["Adam"]

    recurrent_layer = SimpleRNN if model_type == "SimpleRNN" else LSTM

    model = Sequential(
        [
            recurrent_layer(units, input_shape=(lookback, 1), return_sequences=False),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dense(len(HORIZONS), name="close_price_forecast"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def build_scikeras_model(
    model_type: str,
    lookback: int,
    units: int = 64,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
):
    tf_parts = load_tensorflow()
    return build_model(
        model_type=model_type,
        lookback=lookback,
        units=units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        tf_parts=tf_parts,
    )


def inverse_transform_matrix(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    flat = values.reshape(-1, 1)
    inverted = scaler.inverse_transform(flat)
    return inverted.reshape(values.shape)


def metric_table(actuals: np.ndarray, predictions: np.ndarray, model_name: str) -> pd.DataFrame:
    rows = []
    for idx, horizon in enumerate(HORIZONS):
        y_true = actuals[:, idx]
        y_pred = predictions[:, idx]
        mse = mean_squared_error(y_true, y_pred)
        rows.append(
            {
                "Model": model_name,
                "Horizon": f"{horizon} Day",
                "MAE": mean_absolute_error(y_true, y_pred),
                "MSE": mse,
                "RMSE": math.sqrt(mse),
                "R2 Score": r2_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def train_single_model(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    lookback: int,
    units: int,
    dropout_rate: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    validation_split: float,
):
    tf_parts = load_tensorflow()
    tf_parts["tf"].random.set_seed(42)
    np.random.seed(42)

    model = build_model(
        model_type=model_type,
        lookback=lookback,
        units=units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        tf_parts=tf_parts,
    )
    callbacks = [
        tf_parts["EarlyStopping"](
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
        ),
        tf_parts["ReduceLROnPlateau"](
            monitor="val_loss",
            patience=4,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )
    predictions = model.predict(x_test, verbose=0)
    return model, history.history, predictions


def run_light_grid_search(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int,
    unit_grid: List[int],
    dropout_grid: List[float],
    lr_grid: List[float],
    epochs: int,
    batch_size: int,
) -> Dict[str, float]:
    tf_parts = load_tensorflow()
    best_score = float("inf")
    best_params = {
        "units": unit_grid[0],
        "dropout_rate": dropout_grid[0],
        "learning_rate": lr_grid[0],
    }

    split_index = max(1, int(len(x_train) * 0.85))
    x_inner_train, x_val = x_train[:split_index], x_train[split_index:]
    y_inner_train, y_val = y_train[:split_index], y_train[split_index:]

    progress = st.progress(0)
    combinations = [
        (units, dropout_rate, learning_rate)
        for units in unit_grid
        for dropout_rate in dropout_grid
        for learning_rate in lr_grid
    ]

    for index, (units, dropout_rate, learning_rate) in enumerate(combinations, start=1):
        model = build_model(
            model_type=model_type,
            lookback=lookback,
            units=units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            tf_parts=tf_parts,
        )
        history = model.fit(
            x_inner_train,
            y_inner_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
            callbacks=[
                tf_parts["EarlyStopping"](
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                )
            ],
        )
        score = min(history.history["val_loss"])
        if score < best_score:
            best_score = score
            best_params = {
                "units": units,
                "dropout_rate": dropout_rate,
                "learning_rate": learning_rate,
            }
        progress.progress(index / len(combinations))

    progress.empty()
    return best_params


def run_grid_search_cv(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    lookback: int,
    epochs: int,
    batch_size: int,
) -> Dict[str, float]:
    try:
        from scikeras.wrappers import KerasRegressor
    except Exception as exc:
        raise ImportError(
            "SciKeras is required for sklearn GridSearchCV. "
            "Install it with: pip install scikeras"
        ) from exc

    estimator = KerasRegressor(
        model=build_scikeras_model,
        model_type=model_type,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        shuffle=False,
        verbose=0,
    )
    param_grid = {
        "model__units": [32, 64],
        "model__dropout_rate": [0.1, 0.2],
        "model__learning_rate": [0.001, 0.002],
    }
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1,
    )
    search.fit(x_train, y_train)
    return {
        "units": int(search.best_params_["model__units"]),
        "dropout_rate": float(search.best_params_["model__dropout_rate"]),
        "learning_rate": float(search.best_params_["model__learning_rate"]),
    }


def future_forecast(
    model,
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
    lookback: int,
) -> pd.DataFrame:
    latest_window = scaled_values[-lookback:].reshape(1, lookback, 1)
    scaled_forecast = model.predict(latest_window, verbose=0)
    forecast = inverse_transform_matrix(scaled_forecast, scaler)[0]
    return pd.DataFrame(
        {
            "Forecast Horizon": [f"{horizon} Day" for horizon in HORIZONS],
            "Predicted Close Price": forecast,
        }
    )


def plot_price_history(df: pd.DataFrame, target_col: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df[target_col], color="#0f766e", linewidth=1.8)
    ax.set_title(f"Tesla Historical {target_col} Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def plot_actual_vs_predicted(
    dates: pd.DatetimeIndex,
    actuals: np.ndarray,
    predictions: np.ndarray,
    horizon_index: int,
    model_name: str,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    horizon = HORIZONS[horizon_index]
    ax.plot(dates, actuals[:, horizon_index], label="Actual", color="#111827", linewidth=1.8)
    ax.plot(dates, predictions[:, horizon_index], label="Predicted", color="#dc2626", linewidth=1.8)
    ax.set_title(f"{model_name}: Actual vs Predicted Close Price ({horizon} Day Horizon)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def plot_training_history(histories: Dict[str, Dict[str, List[float]]]):
    fig, ax = plt.subplots(figsize=(12, 4))
    for model_name, history in histories.items():
        ax.plot(history["loss"], label=f"{model_name} Train Loss")
        ax.plot(history["val_loss"], label=f"{model_name} Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def app_header():
    st.title("Tesla Stock Price Prediction with SimpleRNN and LSTM")
    st.caption(
        "Deep learning time-series app for forecasting Tesla closing price behavior "
        "over 1-day, 5-day, and 10-day horizons."
    )


def sidebar_controls(df: pd.DataFrame):
    numeric_columns = []
    for column in df.columns:
        if column == REQUIRED_DATE_COLUMN:
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_columns.append(column)

    if not numeric_columns:
        st.error("No numeric stock-price columns were found in the uploaded CSV.")
        st.stop()
    preferred_target = "Close" if "Close" in numeric_columns else numeric_columns[0]

    st.sidebar.header("Model Setup")
    target_col = st.sidebar.selectbox("Target price column", numeric_columns, index=numeric_columns.index(preferred_target))
    lookback = st.sidebar.slider("Lookback window", min_value=20, max_value=120, value=DEFAULT_LOOKBACK, step=5)
    test_size = st.sidebar.slider("Test data size", min_value=0.10, max_value=0.35, value=0.20, step=0.05)
    epochs = st.sidebar.slider("Training epochs", min_value=5, max_value=100, value=25, step=5)
    batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64, 128], index=1)

    st.sidebar.header("Hyperparameters")
    units = st.sidebar.selectbox("Recurrent units", [32, 50, 64, 100], index=2)
    dropout_rate = st.sidebar.selectbox("Dropout rate", [0.0, 0.1, 0.2, 0.3, 0.4], index=2)
    learning_rate = st.sidebar.selectbox("Learning rate", [0.0005, 0.001, 0.002, 0.005], index=1)
    use_grid_search = st.sidebar.checkbox("Run GridSearchCV tuning", value=False)

    return {
        "target_col": target_col,
        "lookback": lookback,
        "test_size": test_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "units": units,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "use_grid_search": use_grid_search,
    }


def render_dataset_summary(df: pd.DataFrame, prepared: pd.DataFrame, target_col: str):
    left, middle, right, fourth = st.columns(4)
    left.metric("Rows", f"{len(prepared):,}")
    middle.metric("Start Date", prepared.index.min().strftime("%Y-%m-%d"))
    right.metric("End Date", prepared.index.max().strftime("%Y-%m-%d"))
    fourth.metric("Missing Target Filled", prepared.attrs.get("missing_target_values_filled", 0))

    with st.expander("Preview cleaned dataset", expanded=False):
        st.dataframe(prepared.tail(20), use_container_width=True)

    plot_price_history(prepared, target_col)


def main():
    app_header()

    uploaded_file = st.file_uploader("Upload Tesla stock CSV file", type=["csv"])
    if uploaded_file is None:
        st.info(
            "Upload the Tesla stock dataset as a CSV file. Expected columns include "
            "Date, Open, High, Low, Close, Adj Close, and Volume."
        )
        st.stop()

    try:
        raw_df = read_csv(uploaded_file)
        raw_df = clean_columns(raw_df)
    except Exception as exc:
        st.error("Could not read the uploaded CSV file.")
        st.exception(exc)
        st.stop()

    if REQUIRED_DATE_COLUMN not in raw_df.columns:
        st.error("The uploaded CSV must include a Date column.")
        st.stop()

    controls = sidebar_controls(raw_df)

    try:
        prepared_df = prepare_stock_data(raw_df, controls["target_col"])
    except Exception as exc:
        st.error("Data preparation failed.")
        st.exception(exc)
        st.stop()

    minimum_rows = controls["lookback"] + max(HORIZONS) + 20
    if len(prepared_df) < minimum_rows:
        st.error(
            f"Need at least {minimum_rows} rows for a {controls['lookback']}-day lookback "
            f"and {max(HORIZONS)}-day horizon. Uploaded data has {len(prepared_df)} rows."
        )
        st.stop()

    render_dataset_summary(raw_df, prepared_df, controls["target_col"])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(prepared_df[[controls["target_col"]]].values)

    x_values, y_values, target_positions = create_multi_horizon_sequences(
        scaled_values=scaled_values,
        lookback=controls["lookback"],
        horizons=HORIZONS,
    )
    x_train, x_test, y_train, y_test, test_positions = chronological_train_test_split(
        x_values=x_values,
        y_values=y_values,
        target_positions=target_positions,
        test_size=controls["test_size"],
    )

    st.subheader("Training Configuration")
    config_cols = st.columns(4)
    config_cols[0].metric("Training Sequences", f"{len(x_train):,}")
    config_cols[1].metric("Testing Sequences", f"{len(x_test):,}")
    config_cols[2].metric("Lookback Days", controls["lookback"])
    config_cols[3].metric("Forecast Horizons", "1, 5, 10 Days")

    run_training = st.button("Train SimpleRNN and LSTM Models", type="primary")
    if not run_training:
        st.stop()

    selected_params = {
        "SimpleRNN": {
            "units": controls["units"],
            "dropout_rate": controls["dropout_rate"],
            "learning_rate": controls["learning_rate"],
        },
        "LSTM": {
            "units": controls["units"],
            "dropout_rate": controls["dropout_rate"],
            "learning_rate": controls["learning_rate"],
        },
    }

    if controls["use_grid_search"]:
        st.subheader("GridSearchCV Hyperparameter Tuning")
        for model_type in ["SimpleRNN", "LSTM"]:
            with st.spinner(f"Searching best parameters for {model_type}..."):
                try:
                    selected_params[model_type] = run_grid_search_cv(
                        model_type=model_type,
                        x_train=x_train,
                        y_train=y_train,
                        lookback=controls["lookback"],
                        epochs=min(controls["epochs"], 20),
                        batch_size=controls["batch_size"],
                    )
                except ImportError as exc:
                    st.warning(f"{exc} Falling back to the built-in time-series grid search.")
                    selected_params[model_type] = run_light_grid_search(
                        model_type=model_type,
                        x_train=x_train,
                        y_train=y_train,
                        lookback=controls["lookback"],
                        unit_grid=[32, 64],
                        dropout_grid=[0.1, 0.2],
                        lr_grid=[0.001, 0.002],
                        epochs=min(controls["epochs"], 20),
                        batch_size=controls["batch_size"],
                    )
        st.dataframe(pd.DataFrame(selected_params).T, use_container_width=True)

    results: List[TrainingResult] = []
    histories = {}

    for model_type in ["SimpleRNN", "LSTM"]:
        params = selected_params[model_type]
        with st.spinner(f"Training {model_type} model..."):
            model, history, scaled_predictions = train_single_model(
                model_type=model_type,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                lookback=controls["lookback"],
                units=int(params["units"]),
                dropout_rate=float(params["dropout_rate"]),
                learning_rate=float(params["learning_rate"]),
                epochs=controls["epochs"],
                batch_size=controls["batch_size"],
                validation_split=0.15,
            )
            predictions = inverse_transform_matrix(scaled_predictions, scaler)
            actuals = inverse_transform_matrix(y_test, scaler)
            metrics = metric_table(actuals, predictions, model_type)
            histories[model_type] = history
            results.append(
                TrainingResult(
                    model_name=model_type,
                    model=model,
                    history=history,
                    predictions=predictions,
                    actuals=actuals,
                    metrics=metrics,
                )
            )

    all_metrics = pd.concat([result.metrics for result in results], ignore_index=True)
    st.subheader("Model Performance Comparison")
    st.dataframe(
        all_metrics.style.format(
            {
                "MAE": "{:.4f}",
                "MSE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R2 Score": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    best_row = all_metrics.sort_values("RMSE").iloc[0]
    st.success(
        f"Best overall result: {best_row['Model']} on {best_row['Horizon']} horizon "
        f"with RMSE {best_row['RMSE']:.4f}."
    )

    plot_training_history(histories)

    st.subheader("Actual vs Predicted Closing Price")
    horizon_label = st.selectbox("Select horizon to visualize", [f"{horizon} Day" for horizon in HORIZONS])
    horizon_index = [f"{horizon} Day" for horizon in HORIZONS].index(horizon_label)
    plot_dates = prepared_df.index[test_positions + HORIZONS[horizon_index] - 1]

    left, right = st.columns(2)
    for column, result in zip([left, right], results):
        with column:
            plot_actual_vs_predicted(
                dates=plot_dates,
                actuals=result.actuals,
                predictions=result.predictions,
                horizon_index=horizon_index,
                model_name=result.model_name,
            )

    st.subheader("Future Price Forecast")
    forecast_cols = st.columns(2)
    for column, result in zip(forecast_cols, results):
        with column:
            st.markdown(f"**{result.model_name} Forecast**")
            forecast_df = future_forecast(
                model=result.model,
                scaled_values=scaled_values,
                scaler=scaler,
                lookback=controls["lookback"],
            )
            st.dataframe(
                forecast_df.style.format({"Predicted Close Price": "{:.2f}"}),
                use_container_width=True,
            )

    st.subheader("Project Notes")
    st.write(
        "Missing stock prices are handled with forward fill followed by backward fill, "
        "which preserves chronological continuity for time-series modeling. "
        "The train-test split is chronological, so future observations are not leaked into training."
    )
    st.write(
        "This Streamlit app trains both SimpleRNN and LSTM models on the selected closing-price "
        "series, compares MAE, MSE, RMSE, and R2, and presents 1-day, 5-day, and 10-day predictions."
    )


if __name__ == "__main__":
    main()