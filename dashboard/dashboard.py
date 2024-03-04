import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

from data_overview import data_overview
from regression_model import regression_model

# Prepare the data
path = 'data'
df = pd.read_csv(f'{path}/clean_data.csv')
df_encoded = pd.read_csv(f'{path}/clean_encoded_data.csv')
df_normalized = pd.read_csv(f'{path}/clean_data_for_regression.csv')
df_normalized_encoded = pd.read_csv(f'{path}/clean_encoded_data_for_regression.csv')

# Categorical variables are of type 'object' and need to be converted to 'category'
# and ordered according to the logical order of the categories
categorical_columns = ['clarity', 'color', 'cut']
for column in categorical_columns:
    df[column] = df[column].astype('category')
    df_normalized[column] = df_normalized[column].astype('category')

clarity_order = ['IF', 'VVS1', 'VVS2', 'SI1', 'SI2', 'I1']
df['clarity'] = pd.Categorical(df['clarity'], categories=clarity_order, ordered=True)
df_normalized['clarity'] = pd.Categorical(df_normalized['clarity'], categories=clarity_order, ordered=True)

color_order = ['Colorless', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
df['color'] = pd.Categorical(df['color'], categories=color_order, ordered=True)
df_normalized['color'] = pd.Categorical(df_normalized['color'], categories=color_order, ordered=True)

cut_order = ['Ideal', 'Premium', 'Very good', 'Good', 'Fair']
df['cut'] = pd.Categorical(df['cut'], categories=cut_order, ordered=True)
df_normalized['cut'] = pd.Categorical(df_normalized['cut'], categories=cut_order, ordered=True)

# Actual dashboard
st.set_page_config(layout = "wide")

st.title('Diamonds Data Analysis Dashboard')
st.subheader('By Zofia Ogonek (s31625)')

selected = option_menu(
    menu_title=None,
    options=[
        'Data sample',
        'Data overview',
        'Regression model'
    ],
    icons=['table', 'bar-chart-line', 'graph-up-arrow'],
    menu_icon='cast',
    default_index=0,
    orientation='horizontal'
)

if selected == 'Data sample':
    st.header('Data sample')
    st.write('This is a sample of the diamonds dataset.')
    st.dataframe(df)

elif selected == 'Data overview':
    data_overview(df, df_encoded)

elif selected == 'Regression model':
    regression_model(df_normalized, df_normalized_encoded)