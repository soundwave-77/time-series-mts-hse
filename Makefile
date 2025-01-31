init:
	poetry shell && poetry install

lint:
	ruff format && ruff check --fix

run-app:
	streamlit run app.py