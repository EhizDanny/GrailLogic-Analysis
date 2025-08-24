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
# antd.divider('Trade Chart', align='center')

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

def analysis():
    uploaded_a_file = False
    if 'upF' not in ss:
        ss['upF'] = {} # Keep the imported data

    # Display one file_uploader per counter value
    for i in range(1, st.session_state['counter'] + 1):
        uploaded = st.sidebar.file_uploader(f"Choose Data {i} File", key=f"data_{i}_up", type='csv')

        # Preprocess the data upon upload and save it to df holder
        if uploaded: 
            df = pd.read_csv(uploaded)
            # cleanCol(df, 'Net P&L %', '%')
            # cleanCol(df, 'Cumulative P&L %', '%')
            # cleanCol(df, 'Drawdown %', '%')
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
                        aggre = st.selectbox('Aggregate The Data', options= ['Daily', 'Weekly',  'Monthly', 'Yearly'], key=f'{index}_aggre')
                        aggregation = 'D' if aggre == 'Daily' else 'M' if aggre == 'Monthly' else 'Y'  if aggre == 'Yearly' else 'W' if aggre == 'Weekly'else None
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
                        aggre = st.selectbox('Aggregate The Data', options= ['Daily', 'Weekly', 'Monthly', 'Yearly'], key=f'{index}_aggre')
                        aggregation = 'D' if aggre == 'Daily' else 'M' if aggre == 'Monthly' else 'Y'  if aggre == 'Yearly' else 'W' if aggre == 'Weekly' else None
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
                aggre2 = st.selectbox('Aggregate The Data', options= ['Daily', 'Weekly', 'Monthly', 'Yearly'], key='aggre')

                if len(selData) >1:
                    # st.write(selData)
                    combined = pd.DataFrame()
                    for _ , data in enumerate(selData):
                        combined = pd.concat([combined, ss.upF[data]], axis=0)

                    # combined = pd.concat([ss.upF[i] for i in selData], axis=0).reset_index(drop=True)
                    # combined.dropna(inplace=True)

                    if resp2 != 'Select One':
                        if combined[resp2].dtype == 'O':
                            st.info(f'Wrong data type selected. Please select any of the numerical column {numeric_cols}')
                            figss = None
                        else:
                            combined = combined[['Date/Time', resp2]]
                            combined['Date/Time'] = pd.to_datetime(combined['Date/Time'])
                            combined.set_index('Date/Time', inplace=True)
                            aggregation = 'D' if aggre2 == 'Daily' else 'M' if aggre2 == 'Monthly' else 'Y'  if aggre2 == 'Yearly' else 'W' if aggre2 == 'Weekly' else None
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


@st.fragment
def MonthAnalysis():
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
    allData = [i for i in ss.upF.keys()]
    with stylable_container(
        key=f"container_with_borders20",
        css_styles="""{
                # background: #7474A3;
                box-shadow:rgba(20, 20, 20, 0.8) 0px 5px 15px 0px;
                border-radius: 0.3rem;
                padding-left: 15px;
                padding-right: 15px;
                padding-bottom: 15px;
                margin-top: 10px
            }"""):
        selectData, respCol,aggregation, AggregData = st.columns([1,1,1,2], gap='large')
        data = selectData.selectbox('Select Dataset', options=['Select One']+allData, key='month_data')
        if data != 'Select One':
            data = ss.upF[data]
            data_ = data.copy()
            colss =['Select Column']+data_.columns.tolist()
            colss.remove('Date/Time')
            colss.remove('Year')
            colss.remove('Month')

            resp4 = respCol.selectbox('Choose The Response Column', options= colss, key='resp4', index = 7)
            aggregate = aggregation.selectbox('Aggregate The Data', options= ['Weekly', 'Quarterly', 'Monthly'], key='aggregated', index=2)
            if resp4 == 'Select Column':
                st.info('Please select a column to plot.')
                return
            groupbyType = 'Month' if aggregate == 'Monthly' else 'Quarter' if aggregate == 'Quarterly' else 'WeekNumber' if aggregate == 'Weekly' else None

            numeric_cols = data_.select_dtypes(include=[np.number]).columns.tolist()
            data_.set_index('Date/Time', inplace=True)
            data_['WeekNumber'] = (data_.index.isocalendar().week).astype(str)
            data_['Quarter'] = (data_.index.quarter).astype(str)
            quarter_map = {'1': '1st Quarter', '2': '2nd Quarter', '3': '3rd Quarter', '4': '4th Quarter'}
            if groupbyType == 'Quarter':
                data_['Quarter'] = data_['Quarter'].map(quarter_map)
            data_.reset_index(inplace=True)
            data_['Year'] = pd.to_datetime(data_['Date/Time']).apply(lambda x: x.strftime('%Y'))
            data_['Month'] = data_['Date/Time'].dt.month_name()
            data_ = data_.groupby(['Year', f'{groupbyType}'])[[i for i in numeric_cols if i not in ['Year', f'{groupbyType}']]].sum().reset_index()

            # ------------------------ Display Year By Month For Each Year ------------------------
            colYear1, colYear2 = st.columns([1,1], gap='large')
            years = data_['Year'].unique().tolist()
            for index, year in enumerate(years):
                year_data = data_[data_['Year'] == year]
                if aggregate == 'Monthly':
                    year_data[f'{groupbyType}'] = pd.Categorical(year_data[f'{groupbyType}'], categories=month_order, ordered=True)
                year_data = year_data.sort_values(f'{groupbyType}')
                custom_scale = [
                    (0, '#FF3F33'),      # Start color
                    (0.5, '#77BEF0'),   # Middle color
                    (1, '#00FFDE')        # End color
                ]
                
                if resp4 != 'Select Column':
                    if data_[resp4].dtype == 'O':
                        st.info(f'Wrong data type selected. Please select any of the numerical column {numeric_cols}')
                    else:
                        if index % 2 == 0:
                            with colYear1:
                                st.divider()
                                st.subheader(f'Data for {year}')
                                fig = px.bar(year_data, x=f'{groupbyType}', y=resp4, title=f'{resp4} By Month for {year}', labels={resp4: resp4, f'{groupbyType}': f'{groupbyType}'}, color = resp4, color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                        else:
                            with colYear2:
                                st.divider()
                                st.subheader(f'Data for {year}')
                                fig = px.bar(year_data, x=f'{groupbyType}', y=resp4, title=f'{resp4} By Month for {year}', labels={resp4: resp4, f'{groupbyType}': f'{groupbyType}'}, color = resp4, color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)


@st.fragment
def jointMonthAnalysis():
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
    allData = [i for i in ss.upF.keys()]
    colYear11_, colYear22_, colYear33_ = st.columns([1,1,1], gap='large')
    combineData = colYear11_.multiselect('Select Datasets to Compare', options=list(ss.upF.keys()), key='joint_month_data', default=list(ss.upF.keys()))
    if len(combineData) > 1:
        combined = pd.DataFrame()
        for _ , data in enumerate(combineData):
            combined = pd.concat([combined, ss.upF[data]], axis=0)

        numeric_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
        combined['Month'] = pd.Categorical(combined['Month'], categories=month_order, ordered=True)
        resp5 = colYear22_.selectbox('Select Column', options = ['Select One'] + [i for i in combined.columns if i not in ['Year', 'Month']], key='joint_month_col', index=5)
        aggregate2 = colYear33_.selectbox('Aggregate The Data', options= ['Weekly', 'Quarterly', 'Monthly'], key='joint_month_agg', index=0)
        groupbyType = 'Month' if aggregate2 == 'Monthly' else 'Quarter' if aggregate2 == 'Quarterly' else 'WeekNumber' if aggregate2 == 'Weekly' else None

        combined['Year'] = pd.to_datetime(combined['Date/Time']).apply(lambda x: x.strftime('%Y'))
        combined['Month'] = combined['Date/Time'].dt.month_name()
        combined.set_index('Date/Time', inplace=True)
        combined['WeekNumber'] = pd.to_datetime(combined.index).isocalendar().week.astype(str)
        combined['Quarter'] = pd.to_datetime(combined.index).quarter.astype(str)
        quarter_map = {'1': '1st Quarter', '2': '2nd Quarter', '3': '3rd Quarter', '4': '4th Quarter'}
        if groupbyType == 'Quarter':
            combined['Quarter'] = combined['Quarter'].map(quarter_map)
        combined = combined.groupby(['Year', f'{groupbyType}'])[[i for i in combined.select_dtypes(include=[np.number]).columns if i not in ['Year', f'{groupbyType}']]].sum().reset_index()

        # ------------------------ Display Year By Month For Each Year ------------------------
        colYear11, colYear22 = st.columns([1,1], gap='large')
        if resp5 != 'Select One':
            if combined[resp5].dtype == 'O':
                st.info(f'Wrong data type selected. Please select any of the numerical column {numeric_cols}')
                figss = None
            else:
                custom_scale = [
                    (0, '#FF3F33'),      # Start color
                    (0.5, '#77BEF0'),   # Middle color
                    (1, '#00FFDE')        # End color
                ]
                years = combined['Year'].unique().tolist()
                for index, year in enumerate(years):
                    year_data = combined[combined['Year'] == year]
                    if aggregate2 == 'Monthly':
                        year_data['Month'] = pd.Categorical(year_data['Month'], categories=month_order, ordered=True)
                    year_data = year_data.sort_values(f'{groupbyType}')
                    if index % 2 == 0:
                        with colYear11:
                            st.divider()
                            st.subheader(f'Combined Data for {year}')
                            fig = px.bar(year_data, x=f'{groupbyType}', y=resp5, labels={resp5: resp5, f'{groupbyType}': f'{groupbyType}'}, color = resp5, color_continuous_scale=custom_scale, text=resp5)
                            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                    else:
                        with colYear22:
                            st.divider()
                            st.subheader(f'Combined Data for {year}')
                            fig = px.bar(year_data, x=f'{groupbyType}', y=resp5, labels={resp5: resp5, f'{groupbyType}': f'{groupbyType}'}, color = resp5, color_continuous_scale=custom_scale, text=resp5)
                            st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    # del combineData,
            


@st.fragment
def yearByYear():
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
    custom_scale = [
                    (0, '#FF3F33'),      # Start color
                    (0.5, '#77BEF0'),   # Middle color
                    (1, '#00FFDE')        # End color
                ]
    # col1, col2 = st.columns([1,1], gap='large')
    combineData = pd.DataFrame()
    for index, value in enumerate(ss.upF.keys()):
        combineData = pd.concat([combineData, ss.upF[value]], axis=0)
    combineData['Year'] = pd.to_datetime(combineData['Date/Time']).apply(lambda x: x.strftime('%Y'))
    combineData['Month'] = combineData['Date/Time'].dt.month_name() 
    combineData = combineData[combineData['Type'].str.contains('Entry')]
    combineData['Month'] = pd.Categorical(combineData['Month'], categories=month_order, ordered=True)
    combineData = combineData.sort_values(['Year', 'Month'])
    usage = combineData[['Year', 'Month']]
    transaction_count = combineData.groupby(['Year', 'Month'])[['Trade #']].count().reset_index().rename(columns={'Trade #': 'Transaction Count'})

    e, a, b, c, d = st.columns([2,1,2,2.5,1])
    analysisType = e.selectbox('Select Analysis Type', options=['Single View Analysis', 'Multi View Analysis'], key='analysis_type')
    data = a.selectbox('Select Dataset', options=[i for i in ss.upF.keys()], key='year_data', disabled=True if analysisType != 'Single View Analysis' else False)

    filteredData = ss.upF[data]

    if analysisType == 'Single View Analysis':  
        filterYear = b.selectbox('Select Year', options=filteredData['Year'].unique().tolist(), key='year_by_year')
    else:
        filterYear = b.multiselect('Select Years', options=['All Year'] + combineData['Year'].unique().tolist(), key='year_by_year', default=['All Year'])

    filterMonth = c.multiselect('Select Month', options=['All Months'] + month_order, key='month_by_month', default=['All Months']) 
    filterType = d.selectbox('Select Visual Type', options=['Chart', 'Table'], key='visual_type', disabled=True if analysisType != 'Multi View Analysis' else False)

    if analysisType == 'Single View Analysis':
        # ------ set up the data for single analysis ------
        filteredData['Year'] = pd.to_datetime(filteredData['Date/Time']).apply(lambda x: x.strftime('%Y'))
        filteredData['Month'] = filteredData['Date/Time'].dt.month_name()
        filteredData = filteredData[filteredData['Type'].str.contains('Entry')]
        filteredData['Month'] = pd.Categorical(filteredData['Month'], categories=month_order, ordered=True)
        filteredData = filteredData.sort_values(['Year', 'Month'])
        transaction_count_single = filteredData.groupby(['Year', 'Month'])[['Trade #']].count().reset_index().rename(columns={'Trade #': 'Transaction Count'})
        # ------ end of setting up data for single analysis ------
        if 'All Months' not in filterMonth:
            transData = transaction_count_single.copy()
            transData = transaction_count[(transaction_count['Year'] == str(filterYear)) & (transaction_count['Month'].isin(filterMonth))]
            total = transData['Transaction Count'].sum()
        else:
            transData = transaction_count_single[transaction_count_single['Year'] == str(filterYear)]
            total = transaction_count_single['Transaction Count'].sum()
        col1, col2 = st.columns([1,1], gap='large')
        with col1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader('Transaction Count Chart View')
            fig = px.bar(transData, x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
            st.plotly_chart(fig, theme='streamlit', use_container_width=True)  
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader('Transaction Count Table View')
            st.dataframe(transData, use_container_width=True)
            st.info(f'Total Transactions for {filterYear} in the selected months is {total}')
    
    else:
        if 'All Year' in filterYear:
            if 'All Months' in filterMonth:
                transData = transaction_count.copy()
                uniqueYears = transData['Year'].unique().tolist()

                one, two = st.columns([1,1], gap='large')
                for index, year in enumerate(uniqueYears):
                    if index % 2 == 0:
                        with one:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='Blues'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                    else:
                        with two:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='Blues'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                st.warning(f'Total Transactions for all years is {transData["Transaction Count"].sum()}')

            else:
                transData = transaction_count[transaction_count['Month'].isin(filterMonth)]
                uniqueYears = transData['Year'].unique().tolist()
                one, two = st.columns([1,1], gap='large')
                for index, year in enumerate(uniqueYears):
                    if index % 2 == 0:
                        with one:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='viridis'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                    else:
                        with two:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='Blues'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                st.warning(f'Total Transactions for all years is {transData["Transaction Count"].sum()}')

        else:
            if 'All Months' in filterMonth:
                transData = transaction_count[transaction_count['Year'].isin(filterYear)]
                uniqueYears = transData['Year'].unique().tolist()
                one, two = st.columns([1,1], gap='large')
                for index, year in enumerate(uniqueYears):
                    if index % 2 == 0:
                        with one:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='Blues'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                    else:
                        with two:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='Blues'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                st.error(f'Total Transactions for all years is {transData["Transaction Count"].sum()}')
            
            else:
                transData = transaction_count[(transaction_count['Year'].isin(filterYear)) & (transaction_count['Month'].isin(filterMonth))]
                uniqueYears = transData['Year'].unique().tolist()
                one, two = st.columns([1,1], gap='large')
                for index, year in enumerate(uniqueYears):
                    if index % 2 == 0:
                        with one:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='Blues'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                    else:
                        with two:
                            st.divider()
                            if filterType == 'Chart':
                                st.subheader(f'Transaction Count for {year}')
                                fig = px.bar(transData[transData['Year'] == year], x='Month', y='Transaction Count', text='Transaction Count', color='Transaction Count', color_continuous_scale=custom_scale)
                                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                            else:
                                st.subheader(f'Transaction Count for {year}')
                                st.dataframe(transData[transData['Year'] == year].style.background_gradient(cmap='Blues'), use_container_width=True)
                                st.info(f'Total Transactions for {year} is {transData[transData["Year"] == year]["Transaction Count"].sum()}')
                st.error(f'Total Transactions for all years is {transData["Transaction Count"].sum()}')



# ----------------------------------------------------------------------------------------------------


# Function to calculate max consecutive wins or losses
def max_consecutive(series, value):
    from itertools import groupby
    return max((sum(1 for _ in group) for key, group in groupby(series) if key == value), default=0)
        
def get_streaks(data, types='win'):
    counter = 0
    indexer = []
    streak = []
    runTimer = 0
    streak_counts = {}  # dictionary to count occurrences of each streak length

    for index, value in enumerate(data.Result):
        if value == types:
            if runTimer == 0:
                indexer.append(index)
                runTimer += 1
            counter += 1
            continue

        runTimer = 0
        if counter > 0:
            streak.append(counter)
            streak_counts[counter] = streak_counts.get(counter, 0) + 1
        counter = 0

    # If streak ended at last row, append it
    if counter > 0:
        streak.append(counter)
        streak_counts[counter] = streak_counts.get(counter, 0) + 1

    if not streak:
        return 0, None, {}

    return max(streak), indexer[streak.index(max(streak))], streak_counts


@st.fragment
def lossAndProfit():
    col1, col2 = st.columns([1,1], gap='large')
    # """Get maximum winning streaks and maximum losing streaks"""
    for index, value in enumerate(ss.upF.keys()):
        index += 1 
        ss.upF[value]['Net P&L USD'] = ss.upF[value]['Net P&L USD'].astype(str).str.replace(',', '')
        ss.upF[value]['Net P&L USD'] = pd.to_numeric(ss.upF[value]['Net P&L USD'], errors='coerce')

        # Keep only rows where there's a closing trade (typically 'Exit long' or 'Exit short')
        exit_mask = ss.upF[value]['Type'].str.contains("Exit", na=False)
        df_trades = ss.upF[value][exit_mask].copy()

        # Create a column for win/loss label
        df_trades['Result'] = df_trades['Net P&L USD'].apply(lambda x: 'win' if x > 0 else 'loss')

        # Calculate maximum consecutive wins and losses
        max_win_streak, win_index, winCounts = get_streaks(df_trades, 'win')
        max_loss_streak, loss_index, lossCounts = get_streaks(df_trades, 'loss')

        # Display the results
        if index % 2 !=0:
            with col1:
                with stylable_container(
                    key=f"containerizer_with_borders{index}",
                    css_styles="""{
                            # background: #7474A3;
                            box-shadow: rgba(0, 0, 0, 0.15) 0px 5px 15px 0px;
                            border-radius: 0.3rem;
                            padding-left: 45px;
                            padding-right: 45px;
                            padding-top: 35px;
                            padding-bottom: 25px;
                            margin-top: 10px
                        }"""):
                    st.subheader(f'Data {index} Streak Analysis')
                    col11, col12 = st.columns([1,1], gap='small')
                    col11.info(f'Max Win Streak: {max_win_streak}' if win_index is not None else "No Winning Streak Found")
                    col11.info(f'Trade Started At: {df_trades.iloc[win_index]["Date/Time"]}' if win_index is not None else 'No Winning Streak Found')

                    col12.error(f'Max Loss Streak: {max_loss_streak}' if loss_index is not None else "No Losing Streak Found")
                    col12.error(f'Trade Started At: {df_trades.iloc[loss_index]["Date/Time"]}' if loss_index is not None else 'No Losing Streak Found')
                    
                    # col13.warning(f'Win Counts: {winCounts}' if win_index is not None else 'No Winning Streak Found')
                    # col13.warning(f'Loss Counts: {lossCounts}' if loss_index is not None else 'No Losing Streak Found')


                    col11.markdown('<br>', unsafe_allow_html=True)
                    show = col11.checkbox('Show Data', key=f'showData{index}')
                    if show:
                        result = col11.selectbox('Result Type', options=['win', 'loss'], key=f'resultType{index}')
                        if result == 'win':
                            streak_rows = df_trades.iloc[win_index:win_index+max_win_streak]
                        else:
                            streak_rows = df_trades.iloc[loss_index:loss_index+max_loss_streak]
                        st.dataframe(
                            streak_rows[['Date/Time', 'Result', 'Type', 'Net P&L USD', 'Cumulative P&L USD', 'Net P&L %', 'Run-up USD', 'Run-up %', 'Drawdown USD', 'Drawdown %']].reset_index(drop=True),
                            use_container_width=True
                        )
                    st.divider()
                    if win_index is not None:
                        win_df = pd.DataFrame(winCounts.items(), columns=['Streak', 'Count'])
                        win_fig = px.bar(win_df, x='Count', y='Streak', orientation='h', title='Win Streak Distribution')
                        st.plotly_chart(win_fig, use_container_width=True)
                    st.divider()
                    if loss_index is not None:
                        loss_df = pd.DataFrame(lossCounts.items(), columns=['Streak', 'Count'])
                        loss_fig = px.bar(loss_df, x='Count', y='Streak', orientation='h', title='Loss Streak Distribution', color_discrete_sequence=['red'])
                        st.plotly_chart(loss_fig, use_container_width=True)
        else:
            with col2:
                with stylable_container(
                    key=f"containerizer_with_borders{index}",
                    css_styles="""{
                            # background: #7474A3;
                            box-shadow: rgba(0, 0, 0, 0.15) 0px 5px 15px 0px;
                            border-radius: 0.3rem;
                            padding-left: 45px;
                            padding-right: 45px;
                            padding-top: 35px;
                            padding-bottom: 25px;
                            margin-top: 10px
                        }"""):
                    st.subheader(f'Data {index} Streak Analysis')
                    col21, col22 = st.columns([1,1], gap='small')
                    
                    col21.info(f'Max Win Streak: {max_win_streak}' if win_index is not None else "No Winning Streak Found")
                    col21.info(f'Trade Started At: {df_trades.iloc[win_index]["Date/Time"]}' if win_index is not None else 'No Winning Streak Found')

                    col22.error(f'Max Loss Streak: {max_loss_streak}' if loss_index is not None else "No Losing Streak Found")
                    col22.error(f'Trade Started At: {df_trades.iloc[loss_index]["Date/Time"]}' if loss_index is not None else 'No Losing Streak Found')

                    # col23.warning(f'Win Counts: {winCounts}' if win_index is not None else 'No Winning Streak Found')
                    # col23.warning(f'Loss Counts: {lossCounts}' if loss_index is not None else 'No Losing Streak Found')

                    col21.markdown('<br>', unsafe_allow_html=True)
                    show = col21.checkbox('Show Data', key=f'showData{index}')
                    if show:
                        result = col21.selectbox('Result Type', options=['win', 'loss'], key=f'resultType{index}')
                        if result == 'win':
                            streak_rows = df_trades.iloc[win_index:win_index+max_win_streak]
                        else:
                            streak_rows = df_trades.iloc[loss_index:loss_index+max_loss_streak]
                        st.dataframe(
                            streak_rows[['Date/Time', 'Result', 'Type', 'Price USD', 'Net P&L USD', 'Cumulative P&L USD', 'Net P&L %', 'Run-up USD', 'Run-up %', 'Drawdown USD', 'Drawdown %']].reset_index(drop=True),
                            use_container_width=True
                        )
                    st.divider()
                    if win_index is not None:
                        win_df = pd.DataFrame(winCounts.items(), columns=['Streak', 'Count'])
                        win_fig = px.bar(win_df, x='Count', y='Streak', orientation='h', title='Win Streak Distribution')
                        st.plotly_chart(win_fig, use_container_width=True)
                    st.divider()
                    if loss_index is not None:
                        loss_df = pd.DataFrame(lossCounts.items(), columns=['Streak', 'Count'])
                        loss_fig = px.bar(loss_df, x='Count', y='Streak', orientation='h', title='Loss Streak Distribution', color_discrete_sequence=['red'])
                        st.plotly_chart(loss_fig, use_container_width=True)



tab1, tab23, tab24, tab25 = st.tabs(['Visual Analysis', 'Balance Of Trade', 'Month By Year Analysis', 'Transaction Count Analysis'])
with tab1:
    analysis()
with tab23:
    if 'upF' in ss and ss.upF:
        lossAndProfit()
with tab24:
    if 'upF' in ss and ss.upF:
        # MonthAnalysis()
        tab11, tab22 = st.tabs(['Single Data Analysis', 'Multiple Data Analysis'])
        with tab11:
            MonthAnalysis()
        with tab22:
            jointMonthAnalysis()
with tab25:
    if 'upF' in ss and ss.upF:
            yearByYear()
st.markdown('<br><br>', unsafe_allow_html=True)
if st.button('Manual Rerun'):
    st.rerun()
