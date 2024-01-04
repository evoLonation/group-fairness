import pandas as pd
import openpyxl
import os 

class TraditionAggregator:
    def __init__(self):
        self.data_origin = {}
        self.data_another = {}
    def handle_file(self, file_path):
        print(file_path)
        df = pd.read_excel(file_path, header=None, sheet_name='origin', engine='openpyxl')
        for j, dp in enumerate(df.loc[1]):
            self.data_origin.setdefault(f'client_{j}', []).append(dp)
        df = pd.read_excel(file_path, header=None, sheet_name='another', engine='openpyxl')
        for j, dp in enumerate(df.loc[1]):
            self.data_another.setdefault(f'client_{j}', []).append(dp)
    def save(self, file_path):
        excel_writer = pd.ExcelWriter(file_path, engine='openpyxl', mode='w')
        pd.DataFrame(self.data_origin).to_excel(excel_writer, index=False, sheet_name='origin')
        pd.DataFrame(self.data_another).to_excel(excel_writer, index=False, sheet_name='another')
        excel_writer.close()
aggregator = TraditionAggregator()
for seed in range(1, 11):
    input_dir = f'../FUEL/out/new_trial_old/eicu_DP_0-01_0-50_seed_{seed}/results/new_trial'
    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)
        aggregator.handle_file(input_file_path)
aggregator.save('total.xlsx')
