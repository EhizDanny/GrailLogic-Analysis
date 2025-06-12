import pandas as pd 
import streamlit as st 
import numpy as np 
import plotly.express as px 
import matplotlib.pyplot as plt 
import streamlit_antd_components as antd
from streamlit import session_state as ss
from streamlit_extras.stylable_container import stylable_container
import warnings
warnings.filterwarnings('ignore')
# import warnings
# warnings.filterwarnings('ignore')


st.set_page_config(layout='wide', page_icon=':chart_with_upwards_trend:', page_title='Grail Analytics')
head1, head2, head3 = st.columns([1,1,1])
head2.markdown(
    "<h1 style='color: #5409DA; font-size: 38px; text-align: center;'>ANALYSIS PLATFORM</h1>",unsafe_allow_html=True)
antd.divider('Trade Chart', align='center')

# Initialize session state counter if not exists
if 'counter' not in st.session_state:
    st.session_state['counter'] = 1

def importer():
    st.session_state['counter'] += 1

def cleanCol(data,col, itemToRemove):
    data[col] = data[col].str.replace(itemToRemove,'')
    data[col] = data[col].astype(float)
    return data[col]

def clean_price_columns(df):
    # Find all columns with 'price' in their name (case-insensitive)
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    for col in price_cols:
        # Remove commas and any non-numeric characters except dot and minus
        df[col] = df[col].astype(str).str.replace(',', '')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

st.sidebar.image('pngwing.com.png', caption='Grail Analytics')
# st.sidebar.markdown("<h3 style=color: #2c3e50; font-size: 24px; text-align: center>Import Data</h3>", unsafe_allow_html=True)
st.sidebar.button('Import More Data', on_click=importer)

uploaded_a_file = False
if 'upF' not in ss:
    ss['upF'] = {} # Keep the imported data

# Display one file_uploader per counter value
for i in range(1, st.session_state['counter'] + 1):
    uploaded = st.sidebar.file_uploader(f"Choose Data {i} File", key=f"data_{i}_up", type='csv')

    # Preprocess the data upon upload and save it to df holder
    if uploaded: 
        df = pd.read_csv(uploaded)
        cleanCol(df, 'P&L %', '%')
        cleanCol(df, 'Cumulative P&L %', '%')
        cleanCol(df, 'Drawdown %', '%')
        df = clean_price_columns(df)
        df['Date/Time'] = pd.to_datetime(df['Date/Time'])
        df['Year'] = df['Date/Time'].dt.year
        df['Month'] = df['Date/Time'].dt.month_name()
        ss.upF[f'data_{i}'] = df
        uploaded_a_file=True
        # st.dataframe(df.head())
 
col1, col2 = st.columns([4,4], gap='large')
for index, value in enumerate(ss.upF.keys()):
        if index % 2 == 0:
            with col1:
                if not index in [0,1]:
                    st.divider()
                with stylable_container(
                        key=f"container_with_borders{index}",
                        css_styles="""{
                                # background: #7474A3;
                                box-shadow: rgba(0, 0, 0, 0.15) 0px 5px 15px 0px;
                                border-radius: 0.3rem;
                                padding-left: 15px;
                                padding-right: 15px;
                                padding-bottom: 15px;
                                margin-top: 10px
                            }"""):
                    
                    index+=1
                    data = ss.upF[value]
                    data_ = data.copy()
                    st.subheader(f'Data {index}')
                    colss =['Select Column']+data_.columns.tolist()
                    colss.remove('Date/Time')
                    colss.remove('Year')
                    colss.remove('Month')
                    resp = st.selectbox('Choose The Response Column', options= colss, key=f'{index}_resp')
                    aggre = st.selectbox('Aggregate The Data', options= ['Daily', 'Monthly', 'Yearly'], key=f'{index}_aggre')
                    aggregation = 'D' if aggre == 'Daily' else 'M' if aggre == 'Monthly' else 'Y'  if aggre == 'Yearly' else None
                    data_.set_index('Date/Time', inplace=True)
                    data_ = data_.resample(aggregation).sum()
                    numeric_cols = data_.select_dtypes(include=[np.number]).columns.tolist()
                    if resp != 'Select Column':
                        if data_[resp].dtype == 'O':
                            st.info(f'Wrong data type selected. Please select any of the numerical column {numeric_cols}')
                        else:
                            data_[resp] = data_[resp].astype(float)
                            fig = px.line(data_frame = data_, x = data_.index, y = resp, title=f'{resp} By Time')
                            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                
        else:
            with col2:
                if not index in [0,1,2]:
                    st.divider()
                with stylable_container(
                        key=f"container_with_borders{index}",
                        css_styles="""{
                                # background: #7474A3;
                                box-shadow: rgba(0, 0, 0, 0.15) 0px 5px 15px 0px;
                                border-radius: 0.3rem;
                                padding-left: 15px;
                                padding-right: 15px;
                                padding-bottom: 15px;
                                margin-top: 10px
                            }"""):
                    
                    index+=1
                    data = ss.upF[value]
                    data_ = data.copy()
                    st.subheader(f'Data {index}')
                    colss =['Select Column']+data_.columns.tolist()
                    colss.remove('Date/Time')
                    colss.remove('Year')
                    colss.remove('Month')
                    resp = st.selectbox('Choose The Response Column', options= colss, key=f'{index}_resp')
                    aggre = st.selectbox('Aggregate The Data', options= ['Daily', 'Monthly', 'Yearly'], key=f'{index}_aggre')
                    aggregation = 'D' if aggre == 'Daily' else 'M' if aggre == 'Monthly' else 'Y'  if aggre == 'Yearly' else None
                    data_.set_index('Date/Time', inplace=True)
                    data_ = data_.resample(aggregation).sum()
                    numeric_cols = data_.select_dtypes(include=[np.number]).columns.tolist()
                    if resp != 'Select Column':
                        if data_[resp].dtype == 'O':
                            st.info(f'Wrong data type selected. Please select any of the numerical column {numeric_cols}')
                        else:
                            data_[resp] = data_[resp].astype(float)
                            fig = px.line(data_frame = data_, x = data_.index, y = resp, title=f'{resp} By Time')
                            st.plotly_chart(fig, theme='streamlit', use_container_width=True)

st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown(
    "<h2 style='color: #5409DA; font-size: 36px; text-align: center;'>Joint Analysis</h2>", unsafe_allow_html=True)
antd.divider('Joint Chart', align='center')

if uploaded_a_file:
    cols1, col2, col3 = st.columns([1,1.5,1], gap='large')
    with col2:
        with stylable_container(
            key=f"container_with_borders12",
            css_styles="""{
                    # background: #7474A3;
                    box-shadow: rgba(0, 0, 0, 0.15) 0px 5px 15px 0px;
                    border-radius: 0.3rem;
                    padding: 25px;
                    margin-top: 10px
                }"""):
            selData = st.multiselect(
                'Select Datasets to Compare',
                options=list(ss.upF.keys()),
                key='data_select',
                default=list(ss.upF.keys())
            )
            colu = ['Select One']+data.columns.tolist()
            colu.remove('Date/Time')
            colu.remove('Year')
            colu.remove('Month')
            resp2 = st.selectbox('Select Column', options = colu, key='j1')
            aggre2 = st.selectbox('Aggregate The Data', options= ['Daily', 'Monthly', 'Yearly'], key='aggre')

            if len(selData) >1:
                # st.write(selData)
                combined = pd.concat([ss.upF[i] for i in selData], axis=0).reset_index(drop=True)
                combined.dropna(inplace=True)

                if resp2 != 'Select One':
                    if combined[resp2].dtype == 'O':
                        st.info(f'Wrong data type selected. Please select any of the numerical column {numeric_cols}')
                        figss = None
                    else:
                        combined = combined[['Date/Time', resp2]]
                        combined['Date/Time'] = pd.to_datetime(combined['Date/Time'])
                        combined.set_index('Date/Time', inplace=True)
                        aggregation = 'D' if aggre2 == 'Daily' else 'M' if aggre2 == 'Monthly' else 'Y'  if aggre2 == 'Yearly' else None
                        combined = combined.resample(aggregation).sum()
                        figss = px.line(combined, x=combined.index, y=resp2, title=f'Joint {resp2} For Selected Datasets')
                else:
                    figss = None
                    st.info('Please select a column to plot.')
            else:
                figss = None
                st.info('Please select more than one dataset to compare.')

    j11, j22,j33 = st.columns([1,8,1], gap='large')       
    if figss:
        with j22:
            with stylable_container(
                key=f"container_with_borders13",
                css_styles="""{
                        # background: #7474A3;
                        box-shadow: rgba(0, 0, 0, 0.15) 0px 5px 15px 0px;
                        border-radius: 0.3rem;
                        padding-left: 45px;
                        padding-right: 45px;
                        padding-top: 5px;
                        padding-bottom: 25px;
                        margin-top: 10px
                    }"""):
                st.plotly_chart(figss, key='joint')
    else:
        pass

    antd.divider('View Data', align='center')
    j1, j2 = st.columns([1,3], gap='large')
    j1.selectbox('Select Any Data To View', options=['Select One']+list(ss.upF.keys()), key='view_data', on_change=lambda: st.session_state.update({'view_data': st.session_state['view_data']}))
    if 'view_data' in st.session_state and st.session_state['view_data'] != 'Select One':
        data_to_view = ss.upF[st.session_state['view_data']]
        st.dataframe(data_to_view)

