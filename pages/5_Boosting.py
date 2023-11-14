import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, auc, accuracy_score, f1_score, max_error, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.tree import export_text, export_graphviz
from sklearn.metrics import (accuracy_score, mean_squared_error, r2_score,classification_report, f1_score, precision_score, recall_score)
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
# from xgboost import XGBClassifier

plt.style.use("fivethirtyeight")

st.title("Решение задачи бустинга на датасете Диабет")

st.write("Подготовка к работе с датасетом, чтение данных")

df = pd.read_csv('data/diabetes1.csv')

st.write("Датасет Диабет")

st.table(df.head())


st.markdown("# Предварительная обработка данных (Препроцессинг)")

st.write(df.columns)

st.write("cколько пропущенных значений отсутствует в каждом элементе")
feature_columns = df.columns

for column in feature_columns:
    st.write(f"{column} ==> Пропущенные значения: {df.isna().mean()}")
    st.write("============================================")
    st.write(f"{column} ==> Значения  равные 0: {len(df.loc[df[column] == 0])}")

st.write("нормирование нулей")
fill_values = SimpleImputer(missing_values=0, strategy="mean", copy=False)
df[feature_columns] = fill_values.fit_transform(df[feature_columns])

for column in feature_columns:
    st.write("============================================")
    st.write(f"{column} ==> Значения равные 0 : {len(df.loc[df[column] == 0])}")

X = df[feature_columns]
y = df.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    st.write("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ (ОБУЧАЮЩАЯ ВЫБОРКА): \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    st.write(f"МАТРИЦА ОШИБОК (CONFUSION MATRIX):\n{confusion_matrix(y_train, y_train_pred)}")
    st.write(f"ACCURACY ПАРАМЕТР:\n{accuracy_score(y_train, y_train_pred):.4f}")
    st.write(f"PRECISION ПАРАМЕТР:\n{precision_score(y_train, y_train_pred):.4f}")
    st.write(f"RECALL ПАРАМЕТР:\n{recall_score(y_train, y_train_pred):.4f}")
    st.write(f"F1 МЕРА:\n{f1_score(y_train, y_train_pred):.4f}")
    st.write(f"ОТЧЕТ О КЛАССИФИКАЦИИ:\n{clf_report}")

    st.write("РЕЗУЛЬТАТЫ ТЕСТОВОЙ ВЫБОРКИ: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    st.write(f"МАТРИЦА ОШИБОК (CONFUSION MATRIX):\n{confusion_matrix(y_test, y_test_pred)}")
    st.write(f"ACCURACY ПАРАМЕТР:\n{accuracy_score(y_test, y_test_pred):.4f}")
    st.write(f"PRECISION ПАРАМЕТР:\n{precision_score(y_test, y_test_pred):.4f}")
    st.write(f"RECALL ПАРАМЕТР:\n{recall_score(y_test, y_test_pred):.4f}")
    st.write(f"F1 МЕРА:\n{f1_score(y_test, y_test_pred):.4f}")
    st.write(f"ОТЧЕТ О КЛАССИФИКАЦИИ:\n{clf_report}")

"""#Bagged Decision Trees"""

tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(estimator=tree, n_estimators=1500, random_state=1234)
bagging_clf.fit(X_train, y_train)

evaluate(bagging_clf, X_train, X_test, y_train, y_test)

scores = {}
scores_f1 = {}

scores['Bagging Classifier'] = {
        'Train': accuracy_score(y_train, bagging_clf.predict(X_train)),
        'Test': accuracy_score(y_test, bagging_clf.predict(X_test)),
}

scores_f1['Bagging Classifier'] = {
        'Train': f1_score(y_train, bagging_clf.predict(X_train)),
        'Test': f1_score(y_test, bagging_clf.predict(X_test))
}

st.write(scores_f1)

"""#Random Forest"""

rf_clf = RandomForestClassifier(random_state=1234, n_estimators=1000)
rf_clf.fit(X_train, y_train)
evaluate(rf_clf, X_train, X_test, y_train, y_test)

scores['Random Forest'] = {
        'Train': accuracy_score(y_train, rf_clf.predict(X_train)),
        'Test': accuracy_score(y_test, rf_clf.predict(X_test)),
    }

scores_f1['Random Forest'] = {
        'Train': f1_score(y_train, rf_clf.predict(X_train)),
        'Test': f1_score(y_test, rf_clf.predict(X_test)),
}

st.write(scores_f1)

"""#AdaBoost"""

ada_boost_clf = AdaBoostClassifier(n_estimators=100)
ada_boost_clf.fit(X_train, y_train) #тренировка на данных
evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)

scores['AdaBoost'] = {
        'Train': accuracy_score(y_train, ada_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, ada_boost_clf.predict(X_test)),
    }

scores_f1['AdaBoost'] = {
        'Train': f1_score(y_train, ada_boost_clf.predict(X_train)),
        'Test': f1_score(y_test, ada_boost_clf.predict(X_test)),
}

st.write(scores_f1)

"""#Catboost"""

cat_clf = CatBoostClassifier(n_estimators=100)
cat_clf.fit(X_train, y_train) #тренировка на данных
evaluate(cat_clf, X_train, X_test, y_train, y_test)

scores['CatBoostClassifier Boosting'] = {
        'Train': accuracy_score(y_train, cat_clf.predict(X_train)),
        'Test': accuracy_score(y_test, cat_clf.predict(X_test)),
    }

scores_f1['CatBoostClassifier Boosting'] = {
        'Train': f1_score(y_train, cat_clf.predict(X_train)),
        'Test': f1_score(y_test, cat_clf.predict(X_test)),
}

st.write(scores_f1)

# """#XGBoost"""

# xgboost_clf = XGBClassifier(n_estimators=100)
# xgboost_clf.fit(X_train, y_train) #тренировка на данных
# evaluate(xgboost_clf, X_train, X_test, y_train, y_test)

# scores['XGBClassifier Boosting'] = {
#         'Train': accuracy_score(y_train, xgboost_clf.predict(X_train)),
#         'Test': accuracy_score(y_test, xgboost_clf.predict(X_test)),
#     }

# scores_f1['XGBClassifier Boosting'] = {
#         'Train': f1_score(y_train, xgboost_clf.predict(X_train)),
#         'Test': f1_score(y_test, xgboost_clf.predict(X_test)),
# }

# st.write(scores_f1)

st.write("""#Сравнение моделей""")

# Сравнение моделей по Accuracy

scores_df = pd.DataFrame(scores)

st.table(scores_df)

scores_df.plot(kind='barh', figsize=(15, 8))

# Сравнение моделей по F1 мере

scores_f1_df = pd.DataFrame(scores_f1)

st.plotly_chart(scores_f1_df.plot(kind='barh', figsize=(15, 8)))

st.markdown("""##Выводы:
По результатам анализа результат на обучаемой выборке лучше, чем на тестовой.
Лучшие показатели у:

*   Bagging Classifier
*   Random Forest
*   XGBClassifier Boosting

На тестовой выборке лучший результат показал CatBoostClassifier Boosting по accuracy и XGBClassifier Boosting по f1 мере.

# Fearture engineering
""")