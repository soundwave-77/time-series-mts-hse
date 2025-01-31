import streamlit as st
from forecaster import Forecaster
import warnings
import matplotlib.pyplot as plt

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
        "##### :green[Если хотите заново запустить инференс модели, обновите страницу!]"
    )

    with st.spinner("Идет процесс инференса модели"):
        model = Forecaster(model_name=model_type, horizon=horizon, use_fitted=True)
        model.load_models()
        model.predict()

        model.plot_predictions()

        for i in plt.get_fignums():
            fig = plt.figure(i)  # Получаем фигуру по номеру
            st.pyplot(fig)
