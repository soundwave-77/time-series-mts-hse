init:
	poetry install

lint:
	ruff format && ruff check --fix

run-app:
	poetry run streamlit run app.py