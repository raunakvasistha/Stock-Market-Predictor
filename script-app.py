
import streamlit as st
import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from pathlib import Path
import base64


# plt.style.use('fivethirtyeight')
def app():
    st.set_page_config(layout="wide")

    def img_to_bytes(img_path):
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded

    header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
        img_to_bytes("kinal.png")
    )
    st.markdown(
        header_html, unsafe_allow_html=True,
    )

    activities = ["Predict Price", "About"]

    choice = st.sidebar.selectbox("Select Your Activity", activities)

    if choice == "Predict Price":
        pdc = []
        pda = []
        lpda = []
        st.subheader("Predict the price of Relience Share ")

        st.subheader("This tool performs the following tasks :")

        st.write("1. Runs A Machine learning Model ")
        st.write(
            "2. Imports the dataset from the year 2017-2019 to learn about relience ")
        st.write(
            "3. Performs a logistical Analysis on the dataset and predict the price ")

        pd1 = st.number_input("Enter the previous day close", 1000, 10000)

        pd2 = st.number_input("Enter the previous day High", 900, 100000)

        pd3 = st.number_input(
            "Enter the previous day Volume", 100000, 100000000)

        pd4 = st.number_input(
            "Enter the previous day percentage change ", -10, 10)

        ##tcount=int(st.selectbox("The number of tweets you want to be analysed",("50","100","200"),1))

        st.markdown(
            "<--------     Also Do checkout the another cool tool from the sidebar")

        Analyzer_choice = st.selectbox("Select Regression Model from best three",  [
                                       "Linear", "Multiple", "Polynomial"])
        classiication_choice = st.selectbox("Select Classification Model ",  [
                                            "KNN", "Random_Forest", "Naive_Bayes"])

        if st.button("Analyze"):
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as p
            from sklearn import datasets, linear_model, metrics
            from sklearn.model_selection import train_test_split
            data = pd.read_csv("MLdataset.csv")
            dataset = data

            st.success("Running Model ")

            def Linear():

                X = data.iloc[:, 2].values
                Y = data.iloc[:, 3].values

                X = X.reshape(-1, 1)
                Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                    X, Y, test_size=0.4, random_state=2)
                reg = linear_model.LinearRegression()
                reg.fit(Xtrain, Ytrain)
                Ypred = reg.predict([[pd1]])
                return(Ypred)

                # classification start

            def Random_Forest():
                X = data.iloc[:, [2, 4, 8, 9]].values
                y = data.iloc[:, -1].values

                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.35, random_state=0)

                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)

                from sklearn.ensemble import RandomForestClassifier
                nb = RandomForestClassifier(
                    n_estimators=7, criterion='entropy', random_state=0)
                nb.fit(X_train, y_train)

                y_pred = nb.predict(sc.transform([[pd1, pd2, pd3, pd4]]))

                if y_pred == 1:
                    return("YES")
                else:
                    return("NO")

            def Multiple():
                X = data.iloc[:, [2, 4, 8, 9]].values
                Y = data.iloc[:, 3].values

                from sklearn.model_selection import train_test_split
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=0.01, random_state=0)

                from sklearn.linear_model import LinearRegression
                regressor = LinearRegression()
                regressor.fit(X_train, Y_train)
                Y_predict = regressor.predict([[pd1, pd2, pd3, pd4]])
                return(Y_predict)

            def Polynomial():
                X = dataset.iloc[:, 2].values
                Y = dataset.iloc[:, 3].values
                X = X.reshape(-1, 1)

                from sklearn.preprocessing import PolynomialFeatures
                poly_reg = PolynomialFeatures(degree=5)
                X_poly = poly_reg.fit_transform(X)
                from sklearn.linear_model import LinearRegression
                lin = LinearRegression()
                lin.fit(X_poly, Y)
                from sklearn.linear_model import LinearRegression
                lin_reg = LinearRegression()
                lin_reg.fit(X, Y)
                return(lin.predict(poly_reg.fit_transform([[pd1]])))

            def KNN():
                X = dataset.iloc[:, [2, 4, 8, 9]].values
                y = dataset.iloc[:, -1].values

                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.01, random_state=0)

                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform([[pd1, pd2, pd3, pd4]])

                from sklearn.neighbors import KNeighborsClassifier
                regressor = KNeighborsClassifier(
                    n_neighbors=5, metric='minkowski', p=2)
                regressor.fit(X_train, y_train)

                y_pred = regressor.predict(X_test)
                if y_pred == 1:
                    return("YES")
                else:
                    return("NO")

            def Naive_Bayes():
                X = dataset.iloc[:, [2, 4, 8, 9]].values
                y = dataset.iloc[:, -1].values

                from sklearn.compose import ColumnTransformer
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)

                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                    test_size=0.01, random_state=0)

                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform([[pd1, pd2, pd3, pd4]])

                from sklearn.naive_bayes import GaussianNB
                nb = GaussianNB()
                nb.fit(X_train, y_train)

                y_pred = nb.predict(X_test)
                if y_pred == 1:
                    return("YES")
                else:
                    return("NO")

            st.write(eval(Analyzer_choice+"()"))
            st.write(eval(classiication_choice+"()"))

    elif (choice == "About"):

        st.markdown(
            "This Project  was designed to see  various machine learning models adapt to  a simple stock model and how accurate can they get .")
        st.header(
            "Built  by [This Website is built by RAUNAK SHARMA](https://www.linkedin.com/in/raunak-sharma-6524181b4/)")


if __name__ == "__main__":
    app()
