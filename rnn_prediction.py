import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import date
import pandas_datareader.data as web
import math

from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

def linear_regr(x,y):
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y)[0]
    return {'k':k,'b':b}

def is_same_sign(group):
    sign = np.sign(group)
    return (sign == sign.flatten()[0]).all()

def find_first_bigger(group, x):
    value = group[group.shape[0] - 1]
    index = group.shape[0] - 1
    for i in np.arange(group.shape[0]):
        if group[i] > x:
            value = group[i]
            index = i
            break
    return{'value':value,'index':index}

def find_first_smaller(group, x):
    value = group[group.shape[0] - 1]
    index = group.shape[0] - 1
    for i in np.arange(group.shape[0]):
        if group[i] < x:
            value = group[i]
            index = i
            break
    return{'value':value,'index':index}
    
def integral_array(group):
    result = np.zeros([group.shape[0]])
    for i in np.arange(group.shape[0]):
        result[i] = np.sum(group[:i+1])
    return result
        
if False: #a standard way to implement KD
    #Stochastic oscillator %K  
    def STOK(df):  
        SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')  
        df = df.join(SOk)  
        return df

    #Stochastic oscillator %D  
    def STO(df, n):  
        SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')  
        SOd = pd.Series(pd.ewma(SOk, span = n, min_periods = n - 1), name = 'SO%d_' + str(n))  
        df = df.join(SOd)  
        return df

def moving_average(group, n = 9):
    sma = group.rolling(center = False, window = n).mean()
    return sma

def moving_average_convergence(group, nslow=26, nfast=12):
    emaslow = group.ewm(min_periods=1,ignore_na=False,adjust=True,span=nslow).mean()
    emafast = group.ewm(min_periods=1,ignore_na=False,adjust=True,span=nfast).mean()
    MACD = emafast-emaslow
    Signal = MACD.ewm(min_periods=1,ignore_na=False,adjust=True,span=9).mean()
    Hist = MACD-Signal
    result = pd.DataFrame({'MACD': MACD, 'Signal': Signal, 'Hist': Hist, 'MACD_emaslow': emaslow, 'MACD_emafast': emafast}, columns=['MACD', 'Signal', 'Hist', 'MACD_emaslow', 'MACD_emafast'])
    return result

def KD(close, high, low, nslow=9, nfast=3, nperiod=14):
    num_max=close.shape[0]
    K=np.zeros(num_max)
    D=np.zeros(num_max)
    RSV=np.zeros(num_max)
    
    for i in range(nperiod-1,num_max):
        RSV[i]=(close[i]-np.amin(low[i-nperiod+1:i+1]))/(np.amax(high[i-nperiod+1:i+1])-np.amin(low[i-nperiod+1:i+1]))

    #min14 = x.rolling(14).min() This is the best way
    #max14 = x.rolling(14).max()
    #rsv = (x - min14) / (max14 - min14)

    K=pd.DataFrame(RSV).ewm(min_periods=1,ignore_na=False,adjust=True,span=nfast).mean()
    D=K.ewm(min_periods=1,ignore_na=False,adjust=True,span=nslow).mean()
        
    result = pd.DataFrame({'RSV': RSV.reshape(num_max), 'K': K.values.reshape(num_max), 'D': D.values.reshape(num_max)}, columns=['RSV', 'K', 'D'])
    return result

def is_increased(group, n=1):
    temp=0.5*(np.sign(np.nan_to_num(group.shift(-n), False)-group)+np.ones(group.shape[0]))
    return np.nan_to_num(temp, False)

def is_incr_tomorrow_Close_Open(Close, Open):
    temp=0.5*(np.sign(Close.shift(-1)-Open.shift(-1))+np.ones(Close.shape[0]))
    return np.nan_to_num(temp, False)

def cross(group1, group2):
    num_max=group1.shape[0]
    temp=np.zeros(num_max)
    for i in range(1,num_max):
        if group1[i-1]<group2[i-1] and group1[i]>group2[i]:
            temp[i]=1
    return temp

def decross(group1, group2):
    num_max=group1.shape[0]
    temp=np.zeros(num_max)
    for i in range(1,num_max):
        if group1[i-1]>group2[i-1] and group1[i]<group2[i]:
            temp[i]=1
    return temp

def scale_haiyan(High,Low,n_day,n_history): #只返回今天和昨天的最大最小值4个量的缩放结果
    High_trunc=High[n_day-n_history+1:n_day+1]
    Low_trunc=Low[n_day-n_history+1:n_day+1]
    Highest=max(High_trunc)
    Lowest=min(Low_trunc)
    result=np.zeros((2,2))
    result[0,0]=(High_trunc[-2]-Lowest)/(Highest-Lowest)
    result[0,1]=(High_trunc[-1]-Lowest)/(Highest-Lowest)
    result[1,0]=(Low_trunc[-2]-Lowest)/(Highest-Lowest)
    result[1,1]=(Low_trunc[-1]-Lowest)/(Highest-Lowest)
    return result

def scale_haiyan_list(High,Low,n_history):
    num_max=High.shape[0]
    temp=np.zeros((num_max,2))
    for i in range(n_history-1,num_max):
        temp[i,0]=scale_haiyan(High,Low,i,n_history)[0,1]
        temp[i,1]=scale_haiyan(High,Low,i,n_history)[1,1]
    result = pd.DataFrame({'high_haiyan': temp[:,0], 'low_haiyan': temp[:,1]}, columns=['high_haiyan', 'low_haiyan'])
    return result
    
def cross_J_haiyan(High,Low,J,n_history):
    num_max=High.shape[0]
    temp=np.zeros(num_max)
    for i in range(n_history-1,num_max):
        result_temp=scale_haiyan(High,Low,i,n_history)
        if J[i-1]<result_temp[1,0] and J[i]>result_temp[1,1]:
            temp[i]=1
    return temp

def cross_J_haiyan_high(High,Low,J,n_history):
    num_max=High.shape[0]
    temp=np.zeros(num_max)
    for i in range(n_history-1,num_max):
        result_temp=scale_haiyan(High,Low,i,n_history)
        if J[i-1]<result_temp[0,0] and J[i]>result_temp[0,1]:
            temp[i]=1
    return temp

def cross_J_haiyan_pseudo(High,Low,J,n_history):
    num_max=High.shape[0]
    temp=np.zeros(num_max)
    for i in range(n_history-1,num_max):
        result_temp=scale_haiyan(High,Low,i,n_history)
        if J[i-1]<J[i] and J[i]>result_temp[0,1]:
            temp[i]=1
    return temp

def decross_J_haiyan(High,Low,J,n_history):
    num_max=High.shape[0]
    temp=np.zeros(num_max)
    for i in range(n_history-1,num_max):
        result_temp=scale_haiyan(High,Low,i,n_history)
        if J[i-1]>result_temp[0,0] and J[i]<result_temp[0,1]:
            temp[i]=1
    return temp

def decross_J_haiyan_low(High,Low,J,n_history):
    num_max=High.shape[0]
    temp=np.zeros(num_max)
    for i in range(n_history-1,num_max):
        result_temp=scale_haiyan(High,Low,i,n_history)
        if J[i-1]>result_temp[1,0] and J[i]<result_temp[1,1]:
            temp[i]=1
    return temp

def decross_J_haiyan_pseudo(High,Low,J,n_history):
    num_max=High.shape[0]
    temp=np.zeros(num_max)
    for i in range(n_history-1,num_max):
        result_temp=scale_haiyan(High,Low,i,n_history)
        if J[i-1]>J[i] and J[i]<result_temp[1,1]:
            temp[i]=1
    return temp

def low_vol_pass_prev_high(Open,High,Low,Close,Vol,idx_test_start,n_history,ratio1,ratio2):
    result=0
    if Close[idx_test_start]>Open[idx_test_start]:
        if Close[idx_test_start]>np.max(High[idx_test_start-n_history:idx_test_start]):
            argmax=np.argmax(High[idx_test_start-n_history:idx_test_start])
            if Vol[idx_test_start-n_history:idx_test_start][argmax]>ratio1*np.average(Vol[idx_test_start-n_history-20:idx_test_start][argmax+20-20:argmax+20]):
                if Close[idx_test_start-n_history:idx_test_start][argmax]<1.25*np.average(Close[idx_test_start-n_history-20:idx_test_start][argmax+20-20:argmax+20]):
                    if Close[idx_test_start-n_history:idx_test_start][argmax]>0.75*np.average(Close[idx_test_start-n_history-20:idx_test_start][argmax+20-20:argmax+20]):
                        if Close[idx_test_start-n_history:idx_test_start][argmax]<Open[idx_test_start-n_history:idx_test_start][argmax]:
                            if Vol[idx_test_start]<ratio2*np.average(Vol[idx_test_start-20:idx_test_start]):
                                result=1
    return result

#################################################################################################################

Load_online=0

if False:   #### 这些code作废了
    if Load_online==0:
        di = pd.read_excel('test_2_di_read.xlsx', sheetname='Sheet1')   #### WMT CF AAPL TSLA NFLX TGT WFC FB SOHU BABA XON 

    if Load_online==1:
    # Converting timestamp to date
        di = web.DataReader('TGT', 'yahoo', date(2014, 12, 1), date(2017, 8, 16))
        di['Date1'] = di.index.date
        di.set_index('Date1', drop=True, inplace=True)
    # save to excel and read from excel
        di.to_excel('test_2_di_read.xlsx','Sheet1')

di_foo = pd.read_excel('stock_price.xlsx', sheetname='X')

#starting_position=0 #从第x个数据开始。因为某些参数比如30天滑动平均 没有前30天的数据。
#profit_batch=profit_batch[starting_position:] ## 从第x个开始 因为starting point=x

#profit_by_action=np.zeros((di_foo.shape[0],4))
#profit_by_action[:,0]=[ii for ii in range(di_foo.shape[0])]

ticker='MLM' 

#for ticker in ['MLM','RAI','EXPE','SBUX','EA','ATVI','AMZN','TSLA']:
#for ticker in ['NFX','NBR','WMB','MU','AVGO','MNST','IRM','ESS','LUV','GD']:
#for ticker in ['NVDA','WDC','HSIC','STX','ORCL','ANSS','AYI','CCL','EFX','VZ']:
#for ticker in ['ETFC','CELG','BSX','GNW','FB','DAL','PBI','ACH','ANSS','CBPO','SOHU','SINA','SAVE','CPG']:
#for ticker in ['UAL','AAL','AMD','SHW','ALK','PAYX','GM','KR','LEN','CTL','EQT','IDXX','KLAC','RRC','COH','INCY','COP']:
for ticker in ['X','FANG','XLNX','MNK','ATI','ALSN','AMZN','TSLA','EA']:
#for ticker in ['UAL']:

    profit_by_action=np.zeros((di_foo.shape[0],4))
    profit_by_action[:,0]=[ii for ii in range(di_foo.shape[0])]

    di = pd.read_excel('stock_price.xlsx', sheetname=ticker)    

    Close = di.Close
    Open = di.Open
    High = di.High
    Low = di.Low
    Volume = di.Volume

    Volume_sma3=moving_average(Volume,3)
    Volume_sma5=moving_average(Volume,5)
    Volume_sma10=moving_average(Volume,10)
    Volume_sma15=moving_average(Volume,15)
    Volume_sma20=moving_average(Volume,20)

    SMA3=moving_average(Close,3)
    SMA5=moving_average(Close,5)
    SMA7=moving_average(Close,7)
    SMA10=moving_average(Close,10)
    SMA13=moving_average(Close,13)
    SMA15=moving_average(Close,15)
    SMA20=moving_average(Close,20)
    SMA30=moving_average(Close,30)
    SMA55=moving_average(Close,55)

    #MACD=moving_average_convergence(Close).MACD
    MACD_temp=moving_average_convergence(Close)
    MACD=MACD_temp.MACD
    Signal=MACD_temp.Signal
    Hist=MACD_temp.Hist
    MACD_emaslow=MACD_temp.MACD_emaslow
    MACD_emafast=MACD_temp.MACD_emafast
    
    KD_temp=KD(Close,High,Low,9,3,14)
    RSV=KD_temp.RSV
    K=KD_temp.K
    D=KD_temp.D
    J=3*K.values-2*D.values

    cross_J_haiyan_10=cross_J_haiyan(High.values,Low.values,J,10)
    decross_J_haiyan_10=decross_J_haiyan(High.values,Low.values,J,10)
    cross_J_haiyan_20=cross_J_haiyan(High.values,Low.values,J,20)
    decross_J_haiyan_20=decross_J_haiyan(High.values,Low.values,J,20)
    cross_J_haiyan_30=cross_J_haiyan(High.values,Low.values,J,30)
    decross_J_haiyan_30=decross_J_haiyan(High.values,Low.values,J,30)
    cross_J_haiyan_40=cross_J_haiyan(High.values,Low.values,J,40)
    decross_J_haiyan_40=decross_J_haiyan(High.values,Low.values,J,40)
    cross_J_haiyan_50=cross_J_haiyan(High.values,Low.values,J,50)
    decross_J_haiyan_50=decross_J_haiyan(High.values,Low.values,J,50)
    cross_J_haiyan_60=cross_J_haiyan(High.values,Low.values,J,60)
    decross_J_haiyan_60=decross_J_haiyan(High.values,Low.values,J,60)
    cross_J_haiyan_80=cross_J_haiyan(High.values,Low.values,J,80)
    decross_J_haiyan_80=decross_J_haiyan(High.values,Low.values,J,80)
    cross_J_haiyan_120=cross_J_haiyan(High.values,Low.values,J,120)
    decross_J_haiyan_120=decross_J_haiyan(High.values,Low.values,J,120)

    cross_J_haiyan_high_30=cross_J_haiyan_high(High.values,Low.values,J,30)
    decross_J_haiyan_low_30=decross_J_haiyan_low(High.values,Low.values,J,30)
    cross_J_haiyan_high_40=cross_J_haiyan_high(High.values,Low.values,J,40)
    decross_J_haiyan_low_40=decross_J_haiyan_low(High.values,Low.values,J,40)
    cross_J_haiyan_high_50=cross_J_haiyan_high(High.values,Low.values,J,50)
    decross_J_haiyan_low_50=decross_J_haiyan_low(High.values,Low.values,J,50)
    cross_J_haiyan_high_60=cross_J_haiyan_high(High.values,Low.values,J,60)
    decross_J_haiyan_low_60=decross_J_haiyan_low(High.values,Low.values,J,60)
    cross_J_haiyan_high_80=cross_J_haiyan_high(High.values,Low.values,J,80)
    decross_J_haiyan_low_80=decross_J_haiyan_low(High.values,Low.values,J,80)
    cross_J_haiyan_high_120=cross_J_haiyan_high(High.values,Low.values,J,120)
    decross_J_haiyan_low_120=decross_J_haiyan_low(High.values,Low.values,J,120)

    cross_J_haiyan_pseudo_30=cross_J_haiyan_pseudo(High.values,Low.values,J,30)
    decross_J_haiyan_pseudo_30=decross_J_haiyan_pseudo(High.values,Low.values,J,30)
    cross_J_haiyan_pseudo_40=cross_J_haiyan_pseudo(High.values,Low.values,J,40)
    decross_J_haiyan_pseudo_40=decross_J_haiyan_pseudo(High.values,Low.values,J,40)
    cross_J_haiyan_pseudo_50=cross_J_haiyan_pseudo(High.values,Low.values,J,50)
    decross_J_haiyan_pseudo_50=decross_J_haiyan_pseudo(High.values,Low.values,J,50)
    cross_J_haiyan_pseudo_60=cross_J_haiyan_pseudo(High.values,Low.values,J,60)
    decross_J_haiyan_pseudo_60=decross_J_haiyan_pseudo(High.values,Low.values,J,60)
    cross_J_haiyan_pseudo_80=cross_J_haiyan_pseudo(High.values,Low.values,J,80)
    decross_J_haiyan_pseudo_80=decross_J_haiyan_pseudo(High.values,Low.values,J,80)
    cross_J_haiyan_pseudo_120=cross_J_haiyan_pseudo(High.values,Low.values,J,120)
    decross_J_haiyan_pseudo_120=decross_J_haiyan_pseudo(High.values,Low.values,J,120)
    
    high_haiyan_80=scale_haiyan_list(High.values,Low.values,80)['high_haiyan'].values
    low_haiyan_80=scale_haiyan_list(High.values,Low.values,80)['low_haiyan'].values
    
    is_incr_10=is_increased(Close,10)
    is_incr_1=is_increased(Close,1)
    #is_incr_tomorrow_Close_Open=is_incr_tomorrow_Close_Open(Close, Open)

    Close = Close.values
    Open = Open.values
    High = High.values
    Low = Low.values
    Volume = Volume.values

    n_train = np.floor(Close.shape[0] * 0.8).astype(int);
    n_test = np.floor(Close.shape[0] * 0.2).astype(int);

    X = np.append(Close.reshape(-1,1)/np.max(Close[:n_train]),Open.reshape(-1,1)/np.max(Open[:n_train]),axis=1) 
    X = np.append(X, High.reshape(-1,1)/np.max(High[:n_train]),axis=1)
    X = np.append(X, Low.reshape(-1,1)/np.max(Low[:n_train]),axis=1)
    X = np.append(X, Volume.reshape(-1,1)/np.max(Volume[:n_train]),axis=1)
    X = np.append(X, MACD.reshape(-1,1)/np.max(MACD[:n_train]),axis=1)
    X = np.append(X, Signal.reshape(-1,1)/np.max(Signal[:n_train]),axis=1)
    X = np.append(X, Hist.reshape(-1,1)/np.max(Hist[:n_train]),axis=1) 
    #X = np.append(X, cross_J_haiyan_10.reshape(-1,1)/np.max(cross_J_haiyan_10[:n_train]),axis=1)
    #X = np.append(X, cross_J_haiyan_20.reshape(-1,1)/np.max(cross_J_haiyan_20[:n_train]),axis=1)
    #X = np.append(X, cross_J_haiyan_30.reshape(-1,1)/np.max(cross_J_haiyan_30[:n_train]),axis=1)
    #X = np.append(X, cross_J_haiyan_40.reshape(-1,1)/np.max(cross_J_haiyan_40[:n_train]),axis=1)

    Y = np.append(is_incr_1.reshape(-1,1),1-is_incr_1.reshape(-1,1),axis=1) 

    X_train_raw = X[:n_train]
    Y_train_raw = Y[:n_train]

    X_test_raw = X[n_train:]
    Y_test_raw = Y[n_train:]
    

    TIME_STEPS = 10     
    INPUT_SIZE = X.shape[1]     
    BATCH_SIZE = 50
    BATCH_INDEX = 0
    OUTPUT_SIZE = Y.shape[1]
    CELL_SIZE = 40
    LR = 0.001

    ## reformat X_train and Y_train
    X_train=X_train_raw[0:TIME_STEPS].reshape(-1,TIME_STEPS,INPUT_SIZE)
    Y_train=Y_train_raw[TIME_STEPS-1].reshape(-1,OUTPUT_SIZE)

    for iii in range(1,n_train-TIME_STEPS):
        X_to_add = X_train_raw[iii:iii+TIME_STEPS].reshape(-1,TIME_STEPS,INPUT_SIZE)
        Y_to_add = Y_train_raw[iii+TIME_STEPS-1].reshape(-1,OUTPUT_SIZE)
        X_train = np.append(X_train, X_to_add, axis=0)
        Y_train = np.append(Y_train, Y_to_add, axis=0)

    BATCH_INDEX_MAX = X_train.shape[0]

    ## reformat X_test and Y_test
    X_test=X_test_raw[0:TIME_STEPS].reshape(-1,TIME_STEPS,INPUT_SIZE)
    Y_test=Y_test_raw[TIME_STEPS-1].reshape(-1,OUTPUT_SIZE)

    for iii in range(1,n_test-TIME_STEPS):
        X_to_add = X_test_raw[iii:iii+TIME_STEPS].reshape(-1,TIME_STEPS,INPUT_SIZE)
        Y_to_add = Y_test_raw[iii+TIME_STEPS-1].reshape(-1,OUTPUT_SIZE)
        X_test = np.append(X_test, X_to_add, axis=0)
        Y_test = np.append(Y_test, Y_to_add, axis=0)

    ########
    model = Sequential()

    # RNN cell
    model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
    ))

    # output layer
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    # optimizer
    adam = Adam(LR)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # training
    for step in range(4001):
        
        BATCH_INDEX = step
        if BATCH_INDEX+BATCH_SIZE >= BATCH_INDEX_MAX:
            BATCH_INDEX = 0

        X_batch = X_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE]
        Y_batch = Y_train[BATCH_INDEX:BATCH_INDEX+BATCH_SIZE]
            
        cost = model.train_on_batch(X_batch, Y_batch)

        BATCH_INDEX += BATCH_SIZE

        if step % 500 == 0:
            cost, accuracy = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=False)
            print('test cost: ', cost, 'test accuracy: ', accuracy)

    print('Done, ticker name:', ticker)

    ###### Learn Loop finished ########
    
###### All Loop finished ########


