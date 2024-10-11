import pickle
import pathlib
import json
import streamlit as st
import numpy as np
import pandas as pd

# from backend.src.core.data import preprocess, predict, explain_prediction
# from backend.src.core.definitions import DATA_DIR, MODEL_DIR

from transformers import pipeline

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.parent.absolute()
DATA_DIR = ROOT_DIR / "backend" / "data"

st.set_page_config(
    page_title="DLS | Trends Indicator",
    page_icon="💬"
)


def preprocess(data)-> str:
    return str(data)


def predict(data: str, model: pipeline):

    prediction = model(data)[0]
    preds = np.array([d['score'] for d in prediction]) > 0.55
    prob_dict = {d['label']: d['score'] for d in prediction}
    return preds, prob_dict

def explain_prediction(mapping, prediction):
    result = []
    try:
        indices = np.where(prediction == 1)
        for idx in indices[0].tolist():
            result.append(mapping[str(idx)])
    except IndexError:
        pass
    return result

def schema_state_init()-> None:
    if "model" not in st.session_state:
        st.session_state["model"] = pipeline("text-classification",
                                             'Maldopast/bge-ecom-trends-classifier',
                                             device='cpu',
                                             batch_size=16,
                                             return_all_scores=True,
                                             )
    if "mapping" not in st.session_state:
        with open('mapping_backend.json', 'r') as f:
            st.session_state["mapping"] = json.load(f)

def clear_text() -> None:
    st.session_state["text_input_area"] = ""

def main()-> None:
    
    schema_state_init()

    st.header(f"Классификация Пользовательского контента", divider='rainbow')

    st.markdown("<b>Описание:</b><br>\
                В рамках данного примера, мы решаем задачу множественной классификации для определения всех классов, к которым можно отнести отзыв Пользователя о Сервисе.\
                Множественная классификация отличается от многоклассовой тем, что экземпляр данных может одновременно относиться сразу к нескольким классам.\
                На вход модели мы подаём только комментарий Пользователя и пытаемся определить для него 50 различных меток классов, к которым он может относиться.", unsafe_allow_html = True)

    review = st.text_area(label = "Введите текст отзыва:", key = "text_input_area", value = "")

    col_1, col_2 = st.columns(2)

    col_1.button("Очистить", use_container_width = True, on_click=clear_text)
    if col_2.button("Отправить", type = "primary", use_container_width = True):
        if len(review) == 0:
            st.write(":red[Длина сообщения не может быть нулевой!]")
        else:
            data = preprocess(review)
            prediction, probs_dict = predict(data, st.session_state["model"])
            trends_list = explain_prediction(st.session_state["mapping"], prediction)
            if len(trends_list) > 0:
                st.write("<br><br>".join([f"<b>Name:</b> {trend[0]}<br><b>Description:</b><br>{trend[1]}" for trend in trends_list]), unsafe_allow_html = True)
                # st.write(probs_dict)
                df = pd.DataFrame.from_dict(probs_dict, orient='index').reset_index()
                df.columns = ['Тренд', 'Вероятность']
                df = df.sort_values('Вероятность', ascending=False)
                st.dataframe(df)
                st.bar_chart(data=df[:5], y='Вероятность', x='Тренд')
            else:
                st.write(":red[Не удалось определить тренды:(]")
    
    return None

if __name__ == "__main__":
    main()