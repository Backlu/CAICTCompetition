# -*- coding: utf-8 -*-

train_data = pd.read_csv('data/ice1/train/15/15_data.csv')
train_normlabel = pd.read_csv('data/ice1/train/15/15_normalInfo.csv')
train_abnormlabel = pd.read_csv('data/ice1/train/15/15_failureInfo.csv')

valida_data = pd.read_csv('data/ice1/train/21/21_data.csv')
valida_normlabel = pd.read_csv('data/ice1/train/21/21_normalInfo.csv')
valida_abnormlabel = pd.read_csv('data/ice1/train/21/21_failureInfo.csv')

test_data = pd.read_csv('data/ice1/test/08/08_data.csv')

column_desc=pd.read_csv('doc/columndesc.csv')


#把time轉成datetime, 方便後續的時間比對操作
train_data['time']=pd.to_datetime(train_data['time'])
train_data = train_data.sort_values(by='time')
train_data['timestamp'] = train_data['time'].apply(lambda x: x.timestamp())

valida_data['time']=pd.to_datetime(valida_data['time'])
valida_data = valida_data.sort_values(by='time')
valida_data['timestamp'] = valida_data['time'].apply(lambda x: x.timestamp())

train_normlabel['startTime']=pd.to_datetime(train_normlabel['startTime'])
train_normlabel['endTime']=pd.to_datetime(train_normlabel['endTime'])
train_normlabel = train_normlabel.sort_values(by='startTime')

valida_normlabel['startTime']=pd.to_datetime(valida_normlabel['startTime'])
valida_normlabel['endTime']=pd.to_datetime(valida_normlabel['endTime'])
valida_normlabel = valida_normlabel.sort_values(by='startTime')

train_abnormlabel['startTime']=pd.to_datetime(train_abnormlabel['startTime'])
train_abnormlabel['endTime']=pd.to_datetime(train_abnormlabel['endTime'])
train_abnormlabel = train_abnormlabel.sort_values(by='startTime')

valida_abnormlabel['startTime']=pd.to_datetime(valida_abnormlabel['startTime'])
valida_abnormlabel['endTime']=pd.to_datetime(valida_abnormlabel['endTime'])
valida_abnormlabel = valida_abnormlabel.sort_values(by='startTime')


#風機參數與風機狀態的數據對應
for i in range(train_abnormlabel.shape[0]):
    startTime, endTime = train_abnormlabel.iloc[i]
    subset=train_data['time'].apply(lambda x: timerangeCheck(x,startTime,endTime))
    train_data.loc[subset, 'label']=1

for i in range(train_normlabel.shape[0]):
    startTime, endTime = train_normlabel.iloc[i]
    subset=train_data['time'].apply(lambda x: timerangeCheck(x,startTime,endTime))
    train_data.loc[subset, 'label']=0
    
for i in range(valida_abnormlabel.shape[0]):
    startTime, endTime = valida_abnormlabel.iloc[i]
    subset=valida_data['time'].apply(lambda x: timerangeCheck(x,startTime,endTime))
    valida_data.loc[subset, 'label']=1

for i in range(valida_normlabel.shape[0]):
    startTime, endTime = valida_normlabel.iloc[i]
    subset=valida_data['time'].apply(lambda x: timerangeCheck(x,startTime,endTime))
    valida_data.loc[subset, 'label']=0
    
    
#刪除無效數據
#风机正常时间区间和风机结冰时间区间均不覆盖的数据视为无效数据   
print('刪除前：','訓練數據：',train_data.shape, ',  驗證數據：',valida_data.shape)
train_data = train_data.dropna()
valida_data = valida_data.dropna()
print('刪除後：','訓練數據：',train_data.shape, ',  驗證數據：',valida_data.shape)