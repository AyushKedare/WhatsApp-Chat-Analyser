import re
import pandas as pd

def preprocess(data):
    pattern = "\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{1,2}\s\w+\s-\s"
    message = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    date = []
    times = []
    for i in dates:
        date.append(i.split(", ")[0])
        times.append(i.split(", ")[1])
        
    time =[]
    for i in times:
        time.append(i.split("\u202f")[0])
        
        
    df = pd.DataFrame({
        'user_msg': message,
        'date': date,
        'time': time
    })
    
    user = []
    msg = []
    for i in df['user_msg']:
        x = re.split( "([\w\W]+?):\s", i)
        if x[1:]:
            user.append(x[1])
            msg.append(x[2])
        else:
            user.append('Group Notification')
            msg.append(x[0])
            
            
    df['user'] = user
    df['msg'] = msg
    df.drop(columns=['user_msg'], inplace=True)
    
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day_name()
    return df