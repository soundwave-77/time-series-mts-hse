# Задания 1-2  
Эти задания объединены в одном ноутбуке, разделены разными секциями внутри него.  



# Задание 3  
Это задание реализовано в виде одностраничного streamlit-приложения для инференса.  
Для запуска обязательно нужен пакетный менеджер poetry.  
Запустить приложение можно двумя способами:
- С помощью утилиты make. Для этого сначала нужно установить зависимости командой
```
make init
```
После этого можно запустить приложение командой
```
make run-app
```
- Если утилиты make нет, то нужно запустить следующую команду:
```
poetry install; poetry run streamlit run app.py
```