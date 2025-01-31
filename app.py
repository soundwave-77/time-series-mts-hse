import streamlit as st
from forecaster import ModelType, Forecaster
import warnings

warnings.filterwarnings("ignore")


def cast_horizon_to_str(horizon: int) -> str:
    if horizon == 7:
        return "Неделя"
    elif horizon == 30:
        return "Месяц"
    elif horizon == 90:
        return "Квартал"
    else:
        raise ValueError("неподдерживаемое значение горизонта планирования")


st.title("Time Series Forecasting :chart_with_upwards_trend:")

placeholder = st.empty()

with placeholder.container():
    model_type = st.selectbox(
        label="Выберите тип модели",
        options=["MA", "ARIMA", "CATBOOST"],
    )

    horizon = st.selectbox(
        label="Выберите горизонт прогнозирования",
        options=[7, 30, 90],
        format_func=lambda x: cast_horizon_to_str(x),
    )

    store_id = st.selectbox(
        label="Выберите магазин", options=["STORE_2"], disabled=True
    )

    run_button = st.button("Запустить инференс", type="primary")

if run_button:
    placeholder.empty()
    st.markdown(
        "##### :green[Если хотите заново запустить обучение модели, обновите страницу!]"
    )

    with st.spinner("Идет процесс обучения модели"):
        model = Forecaster(model_name=model_type, horizon=horizon, use_fitted=True)
        model.load_models()
        model.predict()

        st.pyplot(model.plot_predictions())
