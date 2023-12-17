import pandas as pd
import os 

def tradition_merge(dirs = []):
    data = {}
    for i in range(11):
        data['client{}_origin'.format(i)] = []
        data['client{}_another'.format(i)] = []
    for dir in dirs:
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            print('path: {}'.format(path))
            df = pd.read_excel(path, header=None, sheet_name='origin', engine='openpyxl')
            for i, dp in enumerate(df.loc[1]):
                data['client{}_origin'.format(i)].append(dp)
            df = pd.read_excel(path, header=None, sheet_name='another', engine='openpyxl')
            for i, dp in enumerate(df.loc[1]):
                data['client{}_another'.format(i)].append(dp)
    excel_writer = pd.ExcelWriter('tradition.xlsx', engine='openpyxl', mode='w')
    data_frame = pd.DataFrame(data)
    data_frame.to_excel(excel_writer = excel_writer, index = False)
    excel_writer.close()

tradition_merge(['out\\new_trial\\eicu_DP_0-01_0-50_multi_another_seed_2\\results\\new_trial',
                 'out\\new_trial\\eicu_DP_0-01_0-50_multi_another_seed_3\\results\\new_trial',
                 'out\\new_trial\\eicu_DP_0-01_0-50_multi_another_seed_4\\results\\new_trial',])