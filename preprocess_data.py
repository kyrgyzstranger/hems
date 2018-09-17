import csv
import time
from decimal import Decimal

env = []
incl_cols = [0, 2, 3, 4]

csv_rfile = open('C:/Users/Nazgul/Documents/GitHub/hems/data/Use_Car_Gen_Summer2016_3.csv',)
csv_reader = csv.reader(csv_rfile, delimiter=',')
next(csv_reader, None)
#headers = list(next(csv_reader)[i] for i in incl_cols)


for row in csv_reader:
    raw_state = list(row[i] for i in incl_cols)
    env.append(raw_state)
    
csv_rfile.close()

for i in range(0, len(env)):
    for j in range(1, 4):
        env[i][j] = round(float(env[i][j]), 1)
    env[i][1] = round(env[i][1] - env[i][2], 1)


i = 0
while i < len(env):
    agg = 0.0
    k = i
    while env[k][2] > 0.0:
        agg = agg + env[k][2]
        env[k][2] = 0.0
        k = k + 1
    env[i][2] = agg
    i = k + 1

for i in range(0, len(env)):
    env[i][0] = env[i][0][:-3]
    parsed_time = time.strptime(env[i][0], '%Y-%m-%d %H:%M:%S')
    day_of_week = parsed_time.tm_wday
    hour_of_day = parsed_time.tm_hour
    tou_price = 0.0
    if day_of_week in range(0, 5):
        if hour_of_day < 7 or hour_of_day >= 19:
            tou_price = 7.0
        elif hour_of_day in range(7, 11) or hour_of_day in range(17, 19):
            tou_price = 9.0
        else:
            tou_price = 13.0
    else:
        tou_price = 7.0
    env[i].append(tou_price)

print(env[1])  
csv_wfile = open('C:/Users/Nazgul/Documents/GitHub/hems/data/preprocessed_dataset_Summer2016_3.csv', 'w', newline='')
csv_writer = csv.writer(csv_wfile, delimiter=',')
for row in env:
    csv_writer.writerow(row)

csv_wfile.close()


        
    
    
