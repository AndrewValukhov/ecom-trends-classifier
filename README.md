# ecom-trends-classifier

Ноутбуки для соревнования https://ods.ai/competitions/dls_ecomtech

Для код-ревью их нужно запускать в порядке:

- 1. eda_final - исследование данных и сохранение датасетов для обучения
- 2. training_final - обучение модели и сохранение на HF
- 3. evaluation_final - загрузка обученной модели с HF и предсказание на тестовой выборке (сравнение с лучшим сабмитом)
 
Монолит в streamlit cloud

https://trends-ecom.streamlit.app/

Используется менее качественная модель, основанная на rubert-tiny-2, потому что лучшая модель из соревнования не укладывается в ресуры streamlit cloud