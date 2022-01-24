#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# HIDE CODE
from dateutil.relativedelta import relativedelta
from joblib import dump, load
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
matplotlib.rcParams.update({'font.size': 12})

import warnings
import sys
import os
sys.path.append('/home/server/gli-data-science/')
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import os
import ds_db
import helper_db
from helper import transform_to_rupiah, rupiah_format

import pickle
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
pd.options.mode.chained_assignment = None  # default='warn'
from IPython.display import display, HTML, display_html, IFrame
import ipywidgets as ipyw

def side_by_side_display(dfs:list, captions:list):
    output = ""
    combined = dict(zip(captions, dfs))
    styles = [
        dict(selector="caption", props=[("caption-side", "center"), ("font-size", "100%"), ("color", )])]
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline; font-size:85%' ").set_precision(2).set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0\xa0"

    display(HTML(output))

from sklearn.linear_model import LinearRegression, PoissonRegressor, Ridge, Lasso, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

import textwrap
def split_label(list_label):
    list_label = list(list_label)
    list_label = ["<br>".join(textwrap.wrap(t, width=12)) for t in list_label ]
    return list_label


# %%
li_df_pv = []
prefix = 'pv'


if prefix in ['pv', 'sla']:
    for end_date in pd.date_range('2021-08-01', '2022-01-01', freq='M'):

        start_date = end_date.replace(day=1)

        start_date_str = start_date.strftime('%d%b%y')
        end_date_str = end_date.strftime('%d%b%y')
        print(start_date_str, end_date_str)

        if prefix == 'sla':
            q = '''
                SELECT 
                    att.ATT_ORDER_ID, 
                    att.ATT_ORDER_DATE, 
                    att.ATT_DELIVERY_DATE, 
                    att.ATT_SEND_DATE_TOSTORE,  
                    ( att.ATT_DELIVERY_DATE - att.ATT_SEND_DATE_TOSTORE ) * 24 * 60 AS SLA,
                    CASE 
                        WHEN (TO_NUMBER(TO_CHAR( ATT_SEND_DATE_TOSTORE, 'HH24')) > 20) 
                            THEN (att.ATT_DELIVERY_DATE - (TRUNC(ATT_SEND_DATE_TOSTORE) + 1 + 8/24))  * 24 * 60
                        WHEN (TO_NUMBER(TO_CHAR( ATT_SEND_DATE_TOSTORE, 'HH24')) < 7) 
                            THEN (att.ATT_DELIVERY_DATE - (TRUNC(ATT_SEND_DATE_TOSTORE) + 8/24))  * 24 * 60
                        ELSE ( att.ATT_DELIVERY_DATE - att.ATT_SEND_DATE_TOSTORE ) * 24 * 60
                    END AS SLA_NORM
                FROM 
                    ALFAGIFT_TIME_TRX att 
                WHERE 
                    TRUNC(ATT_ORDER_DATE) BETWEEN '{}' AND '{}'
                ORDER BY 
                    TRUNC(ATT_ORDER_DATE) ASC


            '''.format(start_date_str, end_date_str)


        elif prefix == 'pv':
            q = '''
                SELECT 
                    tc.TRO_MEMBERS, 
                    TRUNC(ame.AME_EVENT_TIME, 'MONTH') AS EVENT_TIME,
                    count(ame.AME_CART_PRODUCT_ID) AS COUNT_VIEW_PRODUCT
                FROM 
                    TEMP_CHURN tc
                    LEFT JOIN PLMS_MEMBER_PROFILE pmp 
                    ON pmp.PMP_MEMBER_ID = tc.TRO_MEMBERS 
                    LEFT JOIN ALFAGIFT_MOE_EVENTS ame 
                    ON ame.AME_PONTA_ID = pmp.PMP_MEMBER_UNIQUE_ID 
                    LEFT JOIN ALFAGIFT_MASTER_PRODUCT amp 
                    ON amp.PRODUCT_ID = ame.AME_CART_PRODUCT_ID 
                WHERE 
                    TRUNC(ame.AME_EVENT_TIME) BETWEEN '{}' AND '{}'
                    AND ame.AME_EVENT_NAME = 'view_product'
                GROUP BY tc.TRO_MEMBERS, TRUNC(ame.AME_EVENT_TIME, 'MONTH')

            '''.format(start_date_str, end_date_str)

        else:
            continue

        con = ds_db.connect_alfabi()
        df_pv = pd.read_sql_query(q, con)
        con.close()
        li_df_pv.append(df_pv)

    df_pv = pd.concat(li_df_pv)

    df_pv.to_csv('/home/server/gli-data-science/akhiyar/churn/{}_{}.csv'.format(prefix, end_date_str), index=False)


# %%





# %%


# %%





# %%
if prefix == 'rating':

    li_re = []
    for end_date in pd.date_range('2021-08-01', '2022-01-01', freq='M'):

        start_date = end_date.replace(day=1)

        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(start_date_str, end_date_str)

        q = '''

        select 
            ttdr.tbtdr_created_date, 
            ttdr.tbtdr_ponta_user AS TRO_MEMBERS, 
            ttdr.tbtdr_tbto_id, 
            ttdr.tbtdr_rating AS rating
        from 
            tb_transaction_delivery_rating ttdr 
        where 
            ttdr.tbtdr_created_date between '{}' and '{}'
            and ttdr.tbtdr_rating > 0
        '''.format(start_date_str, end_date_str)

        con, cur = ds_db.connect_prd_order_4()
        df_re = pd.read_sql(q, con)

        li_re.append(df_re)
        con.close()

    df_re = pd.concat(li_re)

    df_re['created_date'] = df_re['tbtdr_created_date'].dt.strftime('%Y-%m')
    df_re = df_re[df_re['tro_members'] != '']

    ## groupby and get feature by rating below good and rating avg
    df_re_g = df_re.groupby(['tro_members', 'created_date']).agg(
                                    rating_below = ('rating', lambda x: len(x[x<3])),
                                    rating_avg = ('rating','mean')
                                )
    df_re_g_u = df_re_g.unstack(level=1).fillna(0).reset_index()
    df_re_g_u.columns = [' '.join(col).strip().upper() for col in df_re_g_u.columns.values]
    
    df_re_g_u.to_csv('/home/server/gli-data-science/akhiyar/churn/rating_{}.csv'.format(end_date_str), index=False)


# %%

# %%

# %%
