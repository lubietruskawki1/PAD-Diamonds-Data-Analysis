import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import statsmodels.api as sm

def model_visualization(data: pd.DataFrame, model: smf.ols):
    st.header('Model visualization')

    st.subheader('Linearity of the relationship between the response and the explanatory variables')
    data["fitted"] = model.fittedvalues
    fig_linearity = px.scatter(data, "price", "fitted", title="Price vs predicted values")
    fig_linearity.add_trace(go.Scatter(x=[data["price"].min(), data["price"].max()],
                            y=[data["price"].min(), data["price"].max()],
                            mode='lines', showlegend=False))
    st.plotly_chart(fig_linearity)
        
    st.subheader('Statistical independence of residuals')
    data["residuals"] = model.resid
    fig_independence = px.scatter(data, "price", "residuals", title="Price vs model residuals")
    st.plotly_chart(fig_independence)

    st.subheader('Homoscedasticity')
    fig_homoscedasticity = px.scatter(data, "fitted", "residuals", title="Fitted values vs model residuals")
    st.plotly_chart(fig_homoscedasticity)

    st.subheader('Normality of error distribution')
    col1, _, _ = st.columns([2, 1, 1])
    with col1:
        plt.figure(figsize=(10, 10))
        sm.qqplot(data["residuals"], fit=True, line='45')
        plt.title("QQ Plot of residuals")
        st.pyplot(plt)

@st.cache_data
def create_model(formula: str, df: pd.DataFrame):
    model = smf.ols(formula=formula, data=df).fit()
    return model

def create_formula(selected_variables: list, encoded: bool):
    formula = 'price ~ ' + ' + '.join(selected_variables)
    if not encoded:
        formula = formula.replace('clarity', 'C(clarity)')
        formula = formula.replace('cut', 'C(cut)')
        formula = formula.replace('color', 'C(color)')
    return formula

def regression_model(df_normalized: pd.DataFrame, df_encoded_normalized: pd.DataFrame):
    # Preprocessing
    columns_without_price = df_normalized.columns.copy().to_list()
    columns_without_price.remove('price')

    # Actual dashboard page
    st.header('Regression model')
    st.write('Building a regression model to predict the price of diamonds.')

    use_encoded_data = st.checkbox('Use encoded data')
    if use_encoded_data:
        df = df_encoded_normalized
    else:
        df = df_normalized

    use_the_best_model = st.checkbox('Use the best model')

    if use_the_best_model:
        selected_variables = ['carat', 'clarity', 'cut', 'x_dimension', 'depth', 'table']
        formula = create_formula(selected_variables, use_encoded_data)
        st.markdown(f'The best model uses the following formula: `{formula}`')
    else:
        selected_variables = st.multiselect(
            'Select variables for the regression model:',
            columns_without_price
        )
        if len(selected_variables) > 0:
            formula = create_formula(selected_variables, use_encoded_data)
            st.markdown(f'The formula for this regression model is: `{formula}`')

    if len(selected_variables) > 0:
        model = create_model(formula, df)

        st.write(model.summary())

        X = df.drop('price', axis=1)
        y = df['price']

        st.header('Model evaluation')

        pred = model.predict(X)

        st.write('R^2 score:', r2_score(y, pred))
        st.write('Mean squared error:', mean_squared_error(y, pred))
        st.write('Root mean squared error:', np.sqrt(mean_squared_error(y, pred)))

        model_visualization(df, model)