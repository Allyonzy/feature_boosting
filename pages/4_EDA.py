import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

st.title("Решение задачи бустинга на датасете Диабет")

st.write("Подготовка к работе с датасетом, чтение данных")

df = pd.read_csv('data/diabetes1.csv')

st.write("Датасет Диабет")

st.table(df.head())

st.markdown("""
            ## Параметры датасета
            
            * Pregnancies - беременность
            * Glucose - глюкоза
            * BloodPressure - артериальное давление
            * SkinThickness - толщина кожи
            * Insulin - инсулин
            * BMI - индекс массы тела
            * DiabetesPedigreeFunction - диабет родословная функция
            * Age - возраст
            * Outcome - исход (целевая переменная)
    """
    )

st.write(df.info(memory_usage='deep'))
st.write(f"В датасете {df.shape[0]} строк")

st.write(f"\n В датасете {df.shape[1]} столбцов, из них :")
st.write(df.dtypes.value_counts())

st.write('определяем минимальные, максимальные, средние значения, медиана, персентили 25 и 75')
st.table(df.describe().T)

st.plotly_chart(px.box(df))


st.markdown("""Вывод:

1. Датасет "Диабет" не большой и содержит 768 строк и 9 колонок (столбцов).
2. В данных 2 столбца типа float64 и 7 столбцов типа int64
3. Объем памяти около 54,1 КВ.
4. Выбросы есть.
5. Пропусков нет.

#Предварительная обработка данных (Препроцессинг)
""")
