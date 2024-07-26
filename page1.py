import streamlit as st
st.set_page_config(page_title="Financial Analysis Tool", page_icon=":moneybag:", layout="wide", initial_sidebar_state="collapsed")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
from main import data, df_24
from summary_report import get_outlier_count, trend_seasonality
from llm_insights import llm_data
from forecast import forecast_df, forecast_combinations, best_model
import matplotlib.pyplot as plt

###############################################################################################################################################

# nbeats_count = len(best_model[best_model['MODEL']=="N-BEATS"])
# gru_count = len(best_model[best_model['MODEL']=='GRU'])
# lstm_count = len(best_model[best_model['MODEL'] == 'LSTM'])

### Tab that shows to eliminate <24 months ###
tab = st.columns(4)
c1, c2, c3, c4 = st.columns(4)

st._config.set_option(f'theme.backgroundColor' ,"white" )
st._config.set_option(f'theme.base' ,"light" )
st._config.set_option(f'theme.primaryColor' ,"#5591f5" )
st._config.set_option(f'theme.secondaryBackgroundColor' ,"#82E1D7" )
st._config.set_option(f'theme.textColor' ,"#0a1464")

with tab[3]:
    toggled = st.toggle('Eliminate < 24 months')
if toggled is True:
    with c1:
        market = st.multiselect("Select MARKET", df_24['MARKET'].unique(), default=[8191], max_selections=1)
    with c2:
        account = st.multiselect("Select ACCOUNT_ID", df_24['ACCOUNT_ID'].unique(), max_selections=1)
    with c3:
        channel = st.multiselect("Select CHANNEL_ID", df_24['CHANNEL_ID'].unique(), max_selections=1)
    with c4:
        mpg = st.multiselect("Select MPG_ID", df_24['MPG_ID'].unique(), max_selections=1)

    # data greater than 24 months
    dt = df_24.copy()       
    
else:
    with c1:
        market = st.multiselect("Select MARKET", data['MARKET'].unique(), default=[8191], max_selections=1)
    with c2:
        account = st.multiselect("Select ACCOUNT_ID", data['ACCOUNT_ID'].unique(), max_selections=1)
    with c3:
        channel = st.multiselect("Select CHANNEL_ID", data['CHANNEL_ID'].unique(), max_selections=1)
    with c4:
        mpg = st.multiselect("Select MPG_ID", data['MPG_ID'].unique(), max_selections=1)

    # data with all combinations
    dt = data.copy()

with st.container():
    apply_button = st.button("Apply", type="primary", use_container_width=True)

# Leaderboard that displays best performing and least performing MPG
if not apply_button:
    leader_board = st.tabs(["LEADERBOARD"])

    # if account_id is not input by the user
    if len(account)!=0:
        with leader_board[0]:
            dataframe = data.copy()
            if len(channel)==0:
                dataframe = dataframe[dataframe['ACCOUNT_ID'].isin(account)]
            else:
                dataframe = dataframe[dataframe['ACCOUNT_ID'].isin(account) & dataframe['CHANNEL_ID'].isin(channel)]
            total_amount = dataframe[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'AMOUNT']].groupby(['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['AMOUNT'].sum().reset_index().sort_values(by='AMOUNT', ascending=False)
            if len(total_amount)<10:
                top_5_mpg = total_amount.iloc[:round(len(total_amount)/2)]
                bottom_5_mpg = total_amount.iloc[-round(len(total_amount)/2):]
            else:
                top_5_mpg = total_amount.iloc[:5]
                bottom_5_mpg = total_amount.iloc[-5:]
            top_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = top_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype(str)
            bottom_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = bottom_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype(str)
            col1, col2 = st.columns(2)
            with col1.container(border=True):
                st.header("Best performing products")
                st.dataframe(top_5_mpg, hide_index=True, use_container_width=True)
            with col2.container(border=True):
                st.header("Least performing products")
                st.dataframe(bottom_5_mpg, hide_index=True, use_container_width=True)
    else:
        with leader_board[0]:
            total_amount = data[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'AMOUNT']].groupby(['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID'])['AMOUNT'].sum().reset_index().sort_values(by='AMOUNT', ascending=False)
            top_5_mpg = total_amount.iloc[:5]
            bottom_5_mpg = total_amount.iloc[-5:]
            top_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = top_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype(str)
            bottom_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = bottom_5_mpg[['MARKET','ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype(str)
            col1, col2 = st.columns(2)
            with col1.container(border=True):
                st.header("Best performing products")
                st.dataframe(top_5_mpg, hide_index=True, use_container_width=True)
            with col2.container(border=True):
                st.header("Least performing products")
                st.dataframe(bottom_5_mpg, hide_index=True, use_container_width=True)
            
    

    with st.container(border=True):
        st.header("Best Performing Model out of N-BEATS, GRU, LSTM")
        st.subheader("Gated Recurrent Unit - GRU")

####################################################################################################
# Analysis after apply#
####################################################################################################
if apply_button:
    try:
        if toggled is True:
            tab1, tab2, tab3 = st.tabs(['ANALYSIS', 'INSIGHTS', 'FORECAST'])

            #filtered dataframe after applying the combinations in UI
            new_df = dt[dt['MARKET'].isin(market) & dt['ACCOUNT_ID'].isin(account) & dt['CHANNEL_ID'].isin(channel) & dt['MPG_ID'].isin(mpg)]
            # new_df = dt[(dt['MARKET']==int(market[0])) &
            #             (dt['ACCOUNT_ID']==int(account[0])) &
            #             (dt['CHANNEL_ID']==int(channel[0])) &
            #             (dt['MPG_ID']==int(mpg[0]))]
            new_df = new_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATES_UPD', 'AMOUNT']]
            new_df.rename(columns={"DATES_UPD": "PERIOD_DATES"}, inplace=True)
            new_df['DATES'] = pd.to_datetime(new_df['PERIOD_DATES'])
            new_df.sort_values(by="DATES")
            new_df['YEAR'] = new_df['DATES'].dt.year
            new_df['MONTH'] = new_df['DATES'].dt.month_name()
            new_df['YEAR'] = new_df['YEAR'].astype(str)
            
            filtered_combination = [int(market[0]), int(account[0]),int(channel[0]), int(mpg[0])]
            

            ##### Analysis Tab  #########
            with tab1:
                column1, column2, column3 = st.columns((0.4, 0.4, 0.4), vertical_alignment='top')
                col1 = st.columns(1, gap = 'small', vertical_alignment='center')
                col4, col5 = st.columns((0.5, 0.5), gap = 'small')
                
                # Outliers, Negatives and Zero
                try:
                    def highlight_outliers(row):
                        return ['background-color: red; width: 0%;' if row['AMOUNT'] in outlier_val else '' for _ in row]
                   
                    outliers, outlier_val, smooth_data = get_outlier_count(new_df['AMOUNT'])
                    outliers_df = pd.DataFrame()
                    if outliers!=0:
                        outliers_df = new_df[new_df['AMOUNT'].isin(outlier_val)]
                        styled_df = new_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATES', 'AMOUNT']].style.apply(highlight_outliers, axis=1)
                        #outlier_df = new_df.style.apply(color_cells, subset="AMOUNT")
                    #     for i in outlier_val:
                    #         outliers_df = pd.concat([outliers_df, new_df[new_df['AMOUNT']==i]], ignore_index=True)
                        with column1:
                            container = st.container()
                            # container.write(f"Outlier Count - {outliers}")
                            
                            with st.container():
                                outliers_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = outliers_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype("str")
                                outliers_df = outliers_df.sort_values(by="DATES")
                            #st.data_editor(outliers_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', "DATES", "AMOUNT"]], hide_index=True)
                            #st.write(styled_df.hide(axis="index").to_html(), unsafe_allow_html=True )
                            

                                with container.popover(f"{outliers} Outlier found", use_container_width=True):
                                    fig = px.line(new_df.sort_values(by="DATES"), x="DATES", y="AMOUNT", width=2500)
                                    fig.add_scatter(x=outliers_df['DATES'], y=outliers_df['AMOUNT'], name="Outliers", mode="markers", marker=dict(color='red', size=10, symbol='circle'))
                                    print(outliers_df['AMOUNT'])
                                    print(smooth_data.data)
                                    st.plotly_chart(fig, use_container_width=True)
                               
                    else:
                        with column1:
                            # container = st.container(border = True)
                            # container.write("No outliers found") 
                            container = st.container(border = False)
                            container.popover(f"{outliers} Outlier found", use_container_width=True)

                except Exception as e:
                    print(f"{e} in finding outlier")
                
                try:
                    negative = (new_df['AMOUNT'] < 0).sum()
                    negative_df = new_df[new_df['AMOUNT']<0]
                    if len(negative_df)!=0:
                        with column2:
                            container = st.container()
                            with container.popover(f"{negative} Negative Amount found", use_container_width=True):
                                negative_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = negative_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype("str")
                                st.data_editor(negative_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATES', 'AMOUNT']], hide_index=True)
                    else:
                        with column2:
                            container = st.container(border = True)
                            container.write("No negative amount found")
                except Exception as e:
                    print(e)
                        

                try:
                    zero = (new_df['AMOUNT'] == 0).sum()
                    zero_df = new_df[new_df['AMOUNT'] == 0]
                    if len(zero_df)!=0:
                        with column3:
                            container = st.container()
                            with container.popover(f"{zero} Zero Amount found", use_container_width=True):
                                zero_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']] = zero_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID']].astype("str")
                                st.data_editor(zero_df[['MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID', 'DATES', 'AMOUNT']], hide_index=True)
                            
                    else:
                        with column3:
                            container = st.container(border = True)
                            container.write("No zero amount found")
                except Exception as e:
                    print(e)

                # Top containers that gives KPI - pie chart and bar chart
                # with col1.container():
                #     with st.container(border=True):
                #         total_per_year = new_df.groupby('YEAR')['AMOUNT'].sum().reset_index()
                #         fig = px.pie(total_per_year, names="YEAR", values='AMOUNT', color = "YEAR", height=300, title="Yearly Revenue")
                #         st.plotly_chart(fig)

                @st.experimental_fragment
                def year_dropdown():
                    year = st.selectbox("Choose year:",sorted(new_df['YEAR'].unique()))
                    yearly_data = new_df[new_df['YEAR']==year]
                    total_per_month = yearly_data.groupby(['YEAR', 'MONTH'])['AMOUNT'].sum().reset_index()
                    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                    total_per_month['MONTH'] = pd.Categorical(total_per_month['MONTH'], categories=months, ordered=True)
                    total_per_month = total_per_month.sort_values(by='MONTH')
                    fig = px.line(total_per_month, x="MONTH", y="AMOUNT", color="YEAR", height=300, title="Monthly Revenue")
                    st.plotly_chart(fig)


                with st.container(border=True):
                    year_dropdown()                    

                    
                
                
                #trend and seasonality of the product specified
                ts_obj = trend_seasonality(new_df)

                if ts_obj is not None:
                    # Second container that contains trend and seasonality
                    with col4.container(border=True):
                        fig = px.line(ts_obj, y='trend', title="Trend", height=300, color=None, labels={'DATES':'DATES', 'trend':'AMOUNT'})
                        st.plotly_chart(fig)
                    with col5.container(border=True):
                        fig = px.line(ts_obj, y='seasonality', title="Seasonality", height=300, color=None, labels={'DATES':'DATES', 'seasonality':'AMOUNT'})
                        st.plotly_chart(fig)
                else:
                    with st.container(border=True):
                        st.write("No trend and seasonality found")


            ##### Tab 2 for LLM Insights ########    
            with tab2:
                data_for_llm = new_df[['PERIOD_DATES', 'AMOUNT']]
                with st.container(border=True):
                    st.header("Insights from LLM")
                    st.write(llm_data(data_for_llm))

            ##### Tab 3 for prediction and forecasting ######
            with tab3:
                result_data = pd.DataFrame()  # Dataframe to store prediction and forecast

                f_combination = [c for c in forecast_combinations if c == filtered_combination]

                for i in f_combination:
                    result_data = forecast_df[(forecast_df['MARKET'] == i[0]) &         # Filter the data DataFrame where the 'MARKET', 'ACCOUNT_ID', 'CHANNEL_ID', 'MPG_ID' column matches the element of ids
                            (forecast_df['ACCOUNT_ID'] == i[1]) &
                            (forecast_df['CHANNEL_ID'] == i[2]) &
                            (forecast_df['MPG_ID'] == i[3])]
                    
                result_data.set_index("DATES", inplace=True)
                result_data_nbeats = result_data[result_data['MODEL']=='N-BEATS']
                result_data_gru = result_data[result_data['MODEL']=='GRU']
                result_data_lstm = result_data[result_data['MODEL']=='LSTM']
                
                best_model_data = best_model[(best_model['MARKET']==int(market[0])) &
                                            (best_model['ACCOUNT_ID']==int(account[0])) &
                                            (best_model['CHANNEL_ID']==int(channel[0])) &
                                            (best_model['MPG_ID']==int(mpg[0]))]
                best_model_data = best_model_data['MODEL'].iloc[0]

                
                with st.container():
                    new_df_copy = new_df.copy()
                    new_df_copy['PERIOD_DATES'] = pd.to_datetime(new_df_copy['PERIOD_DATES'])
                    new_df_copy.set_index("PERIOD_DATES", inplace=True)
                    new_df_copy = new_df_copy.sort_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=new_df_copy.index, y=new_df_copy['AMOUNT'], mode="lines",name="ACTUAL_AMOUNT", showlegend=True))
                    #fig = px.line(data_frame=new_df_copy, x=new_df_copy.index, y="AMOUNT", labels={"AMOUNT:ACTUAL_AMOUNT"})
                    fig.update_layout(xaxis=dict(title="PERIOD_DATE", dtick="M1",tickformat="%b %Y"), yaxis=dict(title="AMOUNT"))

                    if best_model_data == "N-BEATS":
                        residuals = abs(result_data_nbeats['ACTUAL_AMOUNT'][:9] - result_data_nbeats['AMOUNT'][:9])
                        error_mean = np.array(residuals.mean())
                        conf_percent = round(abs(1 - (error_mean/result_data_nbeats['ACTUAL_AMOUNT'][:9].mean()))*100, 2)
                        
                        fig.add_trace(go.Scatter(x=result_data_nbeats.index, y=result_data_nbeats['AMOUNT'], mode="lines", name=f"<b>N-BEATS - <i>Best Model</i></b>"))
                    else:
                        fig.add_scatter(x=result_data_nbeats.index, y=result_data_nbeats['AMOUNT'], mode="lines", name="N-BEATS")
                    
                    if best_model_data == "GRU":
                        residuals = abs(result_data_gru['ACTUAL_AMOUNT'][:9] - result_data_gru['AMOUNT'][:9])
                        error_mean = np.array(residuals.mean())
                        conf_percent = round(abs(1 - error_mean/result_data_gru['ACTUAL_AMOUNT'][:9].mean())*100, 2)
                        
                        fig.add_scatter(x=result_data_gru.index, y=result_data_gru['AMOUNT'], mode="lines", name=f"<b>GRU - <i>Best Model</i></b>")
                    else:
                        fig.add_scatter(x=result_data_gru.index, y=result_data_gru['AMOUNT'], mode="lines", name="GRU")
                    
                    if best_model_data == "LSTM":
                        residuals = abs(result_data_lstm['ACTUAL_AMOUNT'][:9] - result_data_lstm['AMOUNT'][:9])
                        error_mean = np.array(residuals.mean())
                        conf_percent = round(abs(1 - error_mean/result_data_lstm['ACTUAL_AMOUNT'][:9].mean())*100, 2)

                        fig.add_scatter(x=result_data_lstm.index, y=result_data_lstm['AMOUNT'], mode="lines", name=f"<b>LSTM - <i>Best Model</i></b>")
                    else:
                        fig.add_scatter(x=result_data_lstm.index, y=result_data_lstm['AMOUNT'], mode="lines", name="LSTM")

                    st.plotly_chart(fig, use_container_width=True)
                    with st.container():
                        st.info(f"Best Model is {best_model_data} with confidence percentage {conf_percent}%")
        else:
            st.error("Please eliminate <24 months data and choose the combinations.")
    except Exception as e:
        print(e)
        st.error("Something went wrong! Please try again. Make sure to choose all the IDs", icon="🚨")

        




st.write("")
st.write("")

st.markdown("---")
# if st.button("Back to home",type = "primary", use_container_width=True):
#     st.switch_page("app.py")