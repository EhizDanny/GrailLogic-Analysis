import pandas as pd 
import streamlit as st 
import numpy as np 
import plotly.express as px 
import seaborn as sns
import matplotlib.pyplot as plt 
import streamlit_antd_components as std
# import warnings
# warnings.filterwarnings('ignore')

st.set_page_config(layout='wide')
head1, head2, head3 = st.columns([1,1,1])
head2.markdown("<h1 style=color: #2c3e50; font-size: 30px;text-align; center>Analysis Platform</h1>", unsafe_allow_html=True)
# head2.markdown("<h3 style=color: #34495e; font-size: 32px; text-align: center;>Grail Analytics</h2>", unsafe_allow_html=True)
st.sidebar.subheader('Data Import')
data1 = st.sidebar.file_uploader('Choose Data 1 File' )
data2 = st.sidebar.file_uploader('Choose Data 2 File')

data1Yes = True if data1 else False 
data2Yes = True if data2 else False 

def cleanCol(data,col, itemToRemove):
    if col in data.columns.to_list():
        data[col]= data[col].astype(str)
        data[col] = data[col].str.replace(itemToRemove,'')
        data[col] = data[col].astype(float)
        return data[col]
    else:
        st.error(f'{col} was not part of the columns used in creating the application. Dont be stupid')
        st.toast('E never reach make you still get sense? I said use the same columns used in the first data you gave me. Why you no dey hear word')

if data1Yes:
    data1 = pd.read_csv(data1)
    data1['Date/Time'] = pd.to_datetime(data1['Date/Time'])
    cleanCol(data1, 'Price USD', ',')
    cleanCol(data1, 'P&L %', '%')
    cleanCol(data1, 'Drawdown %', '%')
    cleanCol(data1, 'Cumulative P&L %', '%')
    data1['Year'] = data1['Date/Time'].dt.year
    data1['Month'] = data1['Date/Time'].dt.month_name()

if data2Yes:
    data2 = pd.read_csv(data2)
    data2['Date/Time'] = pd.to_datetime(data2['Date/Time'])
    cleanCol(data2, 'Price USD', ',')
    cleanCol(data2, 'P&L %', '%')
    cleanCol(data2, 'Drawdown %', '%')
    cleanCol(data2, 'Cumulative P&L %', '%')
    data2['Year'] = data2['Date/Time'].dt.year
    data2['Month'] = data2['Date/Time'].dt.month_name()

data1View, data2View = st.columns([1,1])
if data1Yes:
    data1View.subheader('Data1')
    data1View.dataframe(data1,use_container_width=True)
else:
    data1View.info('Waiting for Data1')
if data2Yes:
    data2View.subheader('Data2')
    data2View.dataframe(data2, use_container_width=True)
else:
    data2View.info('Waiting for Data2')
std.divider(label='Chart', align='center')


view1, view2 = st.columns([1,1])
if data1Yes:
    with view1:
        view1Data = st.selectbox('Choose Data', options=['Data1', 'Data2'], key='view1Data')
        if view1Data == 'Data1':
            view1_data = data1
        else:
            view1_data = data2 if data2 else None

        respCol1 = st.selectbox('Response Column', options= ['Select Column']+view1_data.columns.tolist())
        if data1Yes and respCol1 != 'Select Column':
            view1_data[respCol1] = view1_data[respCol1].astype(float)
            fig = px.line(data_frame = view1_data, x = 'Date/Time', y = respCol1, title=f'{respCol1} By Time')
            st.plotly_chart(fig, theme='streamlit')

if data2Yes:
    with view2:
        view2Data = st.selectbox('Choose Data', options=['Data1', 'Data2'], key='view2Data', index=1)
        if view2Data == 'Data1':
            view2_data = data1
        else:
            view2_data = data2

        respCol2 = st.selectbox('Response Column', options= ['Select Column']+view2_data.columns.tolist(), key='ll')
        if data2Yes and respCol2 != 'Select Column':
            view2_data[respCol2] = view2_data[respCol2].astype(float)
            fig = px.line(data_frame = view2_data, x = 'Date/Time', y = respCol2, title=f'{respCol2} By Time')
            st.plotly_chart(fig, theme='streamlit' ,key='po')

std.divider('Joint Chart', align='center')
j1, j2, j3 = st.columns([1,2,1])
j21, j22 = j2.columns([1,1])
if data1Yes and data2Yes:
    cols = j21.selectbox('Select Column', options = ['Select One']+data1.columns.tolist(), key='j1')
    aggre = j22.selectbox('Select Aggregation', options=['Day', 'Month', 'Year'])
    combined =pd.concat([data1, data2], axis=0).reset_index(drop=True)
    combined.dropna(inplace = True)
    if cols != 'Select One':
        if combined[cols].dtype =='O':
            j2.error('Wrong data type selected. Pls select numerical column')
        else :
            combined = combined[['Date/Time', cols]]
            combined['Date/Time'] = pd.to_datetime(combined['Date/Time'])
            combined.set_index('Date/Time', inplace=True)
            aggregation = 'D' if aggre == 'Day' else 'M' if aggre == 'Month' else 'Y'  if aggre == 'Year' else None
            combined = combined.resample(aggregation).sum()

            fig = px.line(combined, x=combined.index, y=cols, title=f'Joint {cols} For Data1 and Data2')
            st.plotly_chart(fig, key='joint')
