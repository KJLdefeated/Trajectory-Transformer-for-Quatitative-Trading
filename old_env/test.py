import os
import pandas as pd
import csv
from datetime import datetime

def _load_csv_data(code):
    data = []
    with open("dataset/" + "stock_data_{}.csv".format(code), newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    instrument_dicts = []
    for row in data[1:]:
        instrument_dict = {}
        instrument_dict['date'] = row[0]
        instrument_dict['data'] = []
        flag = False
        for i in range(1, 9):
            if row[i] == '--':
                flag = True
                break
            instrument_dict['data'].append(float(row[i].replace(',', '').replace('X','')))
        if flag:
            continue
        instrument_dicts.append(instrument_dict)
    return instrument_dicts
"""
codes = ["2303"]
columns, dates_set = ['open', 'high', 'low', 'close', 'volume'], set()
for index, code in enumerate(codes):
    # Load instrument docs by code.
    instrument_docs = _load_csv_data(code)
    print(instrument_docs)
    # Init instrument dicts.
    #for instrument in instrument_docs:
    #    print(instrument)
    instrument_dicts = instrument_docs.to_dict()
    #[instrument.to_dic() for instrument in instrument_docs]
    # Split dates.
    a = instrument_dicts.items()
    dates = [instrument['0'] for instrument in instrument_dicts.items()]
    print(dates)
    # Split instruments.
    instruments = [] 
    # Update dates set.
    dates_set = dates_set.union(dates)
"""
date_obj = datetime.strptime('111/01/03', '%Y/%m/%d')
timestamp = date_obj.timestamp()
print(timestamp)