# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:08:15 2018

@author: HJUNG
"""

## Q Code Reading

import pandas as pd
import NOTAM_API



columns = ['loc', 'state', 'qcode', 'feature1', 'feature2', 'text']
df = pd.DataFrame(columns=columns)

#Statelist = ['USA', 'CAN', 'DEU', 'FRA', 'GBR', 'ESP', 'RUS', 'JPN', 'KOR', 'CHN', 'IND',
#             'THA', 'MYS', 'TUR', 'BRA', 'MEX', 'IDN', 'ARE', 'AUS', 'BEL', 'ARG', 'BGD',
#             'CMR', 'CHL', 'CUB', 'DNK', 'EGY', 'ETH', 'GRC', 'IRN', 'IRQ', 'IRL', 'KEN',
#             'LBY', 'MAR', 'NLD', 'NGA', 'PHL', 'SAU', 'SGP', 'ZAF', 'SWE', 'UKR', 'VNM',
#             'TUN', 'SYR', 'POL', 'PER', 'KAZ', 'HTI', 'CIV', 'GHA', 'GTM', 'PRK', 'ITA']

address_dir = './add.csv'
address = open(address_dir, 'r').read().split()
adlist = []

for i in range(round(len(address)/10)):

    adlist.append(','.join(address[(0 + i*10):min((10 + i*10),len(address))])) 

for a in adlist:
    try:
        Data, error = NOTAM_API.getNOTAM(api_key='f2f818c0-3d00-11e8-b03e-177c16a7d37a', locations=a)
        
        if error is not None:
            print("Error:", error)
        
        i = 0
        
        for line in Data:
            text = line['all'].replace('\n', ' ').replace('.', '').lower()
        
            if ' q)' in text: 
                qcode = text.split(' q)', 1)[1].split('/')[1][1:].upper()[0:4]
                
                df.loc[len(df)] = [line['location'], line['StateName'], line['Qcode'], line['entity'], line['status'], text]
            else:
                qcode = ''
            
            
    
            i = i + 1
            
    except KeyboardInterrupt:
        raise
        
    except:
        continue
        
    
df.to_csv('NOTAMs.csv')