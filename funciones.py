import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np


def T_media(df,ciudad):
    ''' Función para rellenar la nos nulos interpolando y calcular la media de cada ciudad
    '''
    df['T_Max_'+ciudad].interpolate(method='linear', inplace=True, limit_direction="both")
    df['T_Min_'+ciudad].interpolate(method='linear', inplace=True, limit_direction="both")
    df['T_Media_'+ciudad]=(df.iloc[:,1]+df.iloc[:,2])/2
    return df

def media_pais(df):
    ''' Función para calcular la media dandole un peso a cada ciudad
    '''
    df['T_Max_España']=df.iloc[:,1]*0.369+df.iloc[:,4]*0.184+df.iloc[:,7]*0.038+df.iloc[:,10]*0.089+df.iloc[:,13]*0.076+df.iloc[:,16]*0.027+df.iloc[:,19]*0.076+df.iloc[:,22]*0.043+df.iloc[:,25]*0.033+df.iloc[:,28]*0.065
    df['T_Min_España']=df.iloc[:,2]*0.369+df.iloc[:,5]*0.184+df.iloc[:,8]*0.038+df.iloc[:,11]*0.089+df.iloc[:,14]*0.076+df.iloc[:,17]*0.027+df.iloc[:,20]*0.076+df.iloc[:,23]*0.043+df.iloc[:,26]*0.033+df.iloc[:,29]*0.065
    df['T_Media_España']=df.iloc[:,3]*0.369+df.iloc[:,6]*0.184+df.iloc[:,9]*0.038+df.iloc[:,12]*0.089+df.iloc[:,15]*0.076+df.iloc[:,18]*0.027+df.iloc[:,21]*0.076+df.iloc[:,24]*0.043+df.iloc[:,27]*0.033+df.iloc[:,30]*0.065
    return df

def hora_a_dia(lista):
    ''' Con esta función paso hago una media de los precios horarioa para ese día
    Es importante tener en cuenta los días con cambios de hora. Ese día solo hay 23 horas
    '''
    diario=[]
    for i in range(0,2064,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[2064:2087])/23) # 2016
                  
    for i in range(2087,10799,24):
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[10799:10822])/23) # 2017
                  
    for i in range(10822,19534,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[19534:19557])/23) # 2018
                  
    for i in range(19557,28437,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[28437:28460])/23) # 2019
                  
    for i in range(28460,37172,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[37172:37195])/23) # 2020
                  
    for i in range(37195,45907,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[45907:45930])/23) # 2021
                  
    for i in range(45930,54642,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[54642:54665])/23) # 2022
                  
    for i in range(54665,63377,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[63377:63400])/23) # 2023
                  
    for i in range(63400,72280,24): 
        diario.append(sum(lista[i:i+23])/24)
    diario.append(sum(lista[72280:72303])/23) # 2024
                  
    for i in range(72303,74487,24): 
        diario.append(sum(lista[i:i+23])/24)
    return diario

def juntar_todo():
    df_Precio_Luz_dia=pd.read_csv('Precio_Luz_dia.csv')
    df_Balance=pd.read_csv('Balance_diario.csv')
    df_gas=pd.read_csv('Precio_gas.csv')
    df_T=pd.read_csv('Temperaturas.csv')
    
    dfs=[df_Balance,df_T,df_gas,df_Precio_Luz_dia]
    
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on='Fecha', how='inner')
    df['Fecha']=pd.to_datetime(df['Fecha'])
    df.rename(columns={'Cierre': 'Precio_gas'}, inplace=True)
    return df


def modelos(df):
    ''' Una función que dado un dataframe calcula las metricas de 5 modelos de ML 
    '''
    X = df.drop(['Fecha','y'], axis=1)
    y = df['y']
    scaler = MinMaxScaler()
    X=scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model1 = LinearRegression()
    model2 = RandomForestRegressor()
    model3 =XGBRegressor()
    model4 = KNeighborsRegressor()
    model5 =GradientBoostingRegressor()
    
    model1.fit(X_train, y_train)
    y_pred = model1.predict(X_test)
    y_pred_train= model1.predict(X_train)
    
    rmse_train1 = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train1 = r2_score(y_train, y_pred_train)    
        
    rmse1 = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_1 = r2_score(y_test, y_pred)
    
    
    model2.fit(X_train, y_train)  
    y_pred = model2.predict(X_test)
    y_pred_train= model2.predict(X_train)
    
    rmse_train2 = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train2 = r2_score(y_train, y_pred_train)    
        
    rmse2 = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_2 = r2_score(y_test, y_pred)    
    
    model3.fit(X_train, y_train)  
    y_pred = model3.predict(X_test)
    y_pred_train= model3.predict(X_train)
    
    rmse_train3 = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train3 = r2_score(y_train, y_pred_train)    
        
    rmse3 = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_3 = r2_score(y_test, y_pred)

    model4.fit(X_train, y_train)  
    y_pred = model4.predict(X_test)
    y_pred_train= model4.predict(X_train)
    
    rmse_train4 = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train4 = r2_score(y_train, y_pred_train)    
        
    rmse4 = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_4 = r2_score(y_test, y_pred)    
    
    model5.fit(X_train, y_train)  
    y_pred = model5.predict(X_test)
    y_pred_train= model5.predict(X_train)
    
    rmse_train5 = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train5 = r2_score(y_train, y_pred_train)    
        
    rmse5 = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_5 = r2_score(y_test, y_pred)
    
    results = {'Model': ['Linear Regression','Random Forest','XGBoost','K-Nearest Neighbors','Gradient Boosting'],'RMSE train': [rmse_train1,rmse_train2,rmse_train3,rmse_train4,rmse_train5],'R2 train': [r2_train1,r2_train2,r2_train3,r2_train4,r2_train5], 'RMSE test':[rmse1,rmse2,rmse3,rmse4,rmse5],'R2 test': [r2_1,r2_2,r2_3,r2_4,r2_5] }

    return pd.DataFrame(results)
    
def crea_var_temporales(df):
    ''' Función que crea columnas. Para cada día pone el precio de i días antes
    '''
    for i in [1,2,3,7,14]:
        df['Precio_ES_'+str(i)]=df.precio_ES.shift(i)
        df['Diff_Precio_ES_'+str(i)]=df.precio_ES.diff(i)
        df['PCT_Precio_ES_'+str(i)]=df.precio_ES.pct_change(i)
    df=df.dropna()
    return df


def prepro_1_ML(df):
    ''' Elige ciertas columnas del dataset original, mete las variables temporales y lo que sería la columna del precio del día siguiente, es decir, 
        la varia objetivo o la 'y'.
    '''
    df=df[['Fecha','Generación renovable','Generación no renovable','Saldo I. internacionales','Demanda',
    'T_Max_España','T_Min_España','T_Media_España','Precio_gas','Precio_PT','precio_FR','precio_ES']]
    df=crea_var_temporales(df)
    df['y']=df.precio_ES.shift(-1)
    df=df.dropna()
    return df

def prepro_2_ML(df):
    ''' Elige ciertas columnas del dataset original, mete las variables temporales y lo que sería la columna del precio del día siguiente, es decir, 
        la varia objetivo o la 'y'.
    '''
    df_1=df[['Fecha','Saldo I. internacionales','Demanda',
    'T_Max_España','T_Min_España','T_Media_España','Precio_gas','Precio_PT','precio_FR','precio_ES']]
    df_1['Generación']=df['Generación renovable']+df['Generación no renovable']
    df_1['Bombeo']=df_1['Generación']-df['Demanda']-df['Saldo I. internacionales']
    df_1=crea_var_temporales(df_1)
    df_1['y']=df_1.precio_ES.shift(-1)
    df_1=df_1.dropna()
    return df_1

def prepro_3_ML(df):
    ''' Elige ciertas columnas del dataset original, mete las variables temporales y lo que sería la columna del precio del día siguiente, es decir, 
        la varia objetivo o la 'y'. Esta vez le añado una columna que dice si es fin de semana mañana
    '''
    df_1=df[['Fecha','Saldo I. internacionales','Demanda',
    'T_Media_España','Precio_gas','precio_FR','precio_ES']]
    df_1['Generación']=df['Generación renovable']+df['Generación no renovable']
    df_1['Bombeo']=df_1['Generación']-df['Demanda']-df['Saldo I. internacionales']
    df_1['Finde_mañana']=(df_1.index % 7 == 0) | ((df_1.index -1) % 7==0)
    df_1=crea_var_temporales(df_1)
    df_1['y']=df_1.precio_ES.shift(-1)
    df_1=df_1.dropna()
    return df_1    
    
    
    
    
    
    
    
    
    
    