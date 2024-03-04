import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

def variable_distributions(
        df: pd.DataFrame,
        numerical_columns: list,
        color_mapping: dict
    ):
    st.header('Variable distributions')
    st.write('Visualizing the distribution of numerical variables.')

    variable = st.selectbox('Select a variable:', numerical_columns)

    sns.histplot(df[variable], kde=True, color=color_mapping[variable])
    plt.title(f'Distribution of {variable}')
    st.pyplot(plt, use_container_width=True)

def price_dependencies_on_other_variables(
        df: pd.DataFrame,
        numerical_columns_without_price: list,
        color_mapping: dict
    ):
    st.header('Price dependencies on other variables')
    st.write('Visualizing the relationship between price and other variables.')

    variable = st.selectbox('Select a variable:', numerical_columns_without_price)

    sns.scatterplot(x=df[variable], y=df['price'], color=color_mapping[variable])
    plt.title(f'Relationship between {variable} and price')
    st.pyplot(plt, use_container_width=True)

def correlation_matrix(
        df: pd.DataFrame,
        df_encoded: pd.DataFrame,
        numerical_columns: list
    ):
    st.header('Correlation matrix')
    st.write('Visualizing the correlation between variables.')

    add_encoded_categorical = st.checkbox('Add encoded categorical variables')

    if add_encoded_categorical:
        correlation_matrix = df_encoded.corr()
    else:
        correlation_matrix = df[numerical_columns].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='rocket')
    plt.title('Correlation matrix')
    st.pyplot(plt, use_container_width=True)

def category_frequencies(df: pd.DataFrame, categorical_columns: list):
    st.header('Category frequencies')
    st.write('Visualizing the frequency of categorical variables.')

    variable = st.selectbox('Select a variable:', categorical_columns)

    sns.countplot(x=df[variable], palette='pastel', hue=df[variable])
    plt.title(f'Frequency of {variable}')
    st.pyplot(plt, use_container_width=True)


def data_overview(df: pd.DataFrame, df_encoded: pd.DataFrame):
    # Preprocessing
    numerical_columns = df.select_dtypes(include=[np.number]).columns.to_list()
    numerical_columns_without_price = numerical_columns.copy()
    numerical_columns_without_price.remove('price')
    categorical_columns = ['clarity', 'color', 'cut']

    # Set a pastel color palette
    pastel_palette = sns.color_palette("pastel")

    # Create a dictionary to map each numerical column to a different pastel color
    color_mapping = {variable: pastel_palette[i] for i, variable in enumerate(numerical_columns)}

    # Actual data overview page
    with st.sidebar:
        selected_data_overview = option_menu(
            menu_title='Data visualizations',
            options = [
                'Variable distributions',
                'Price dependencies on other variables',
                'Correlation matrix',
                'Category frequencies'
            ],
            icons=[
                'bar-chart',
                'currency-dollar',
                'diagram-3',
                'list-task'
            ],
            menu_icon='bar-chart-line-fill',
            default_index=0
        )

    if selected_data_overview == 'Variable distributions':
        variable_distributions(df, numerical_columns, color_mapping)

    elif selected_data_overview == 'Price dependencies on other variables':
        price_dependencies_on_other_variables(df, numerical_columns_without_price, color_mapping)

    elif selected_data_overview == 'Correlation matrix':
        correlation_matrix(df, df_encoded, numerical_columns)

    elif selected_data_overview == 'Category frequencies':
        category_frequencies(df, categorical_columns)