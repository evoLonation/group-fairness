import pandas as pd
import os 

output_dir = 'results'
origin_result_dir = '../FUEL/out/new_trial'
os.makedirs(output_dir)
tradition_dir = os.path.join(output_dir, 'tradition')
os.makedirs(tradition_dir)
bootstrap_dir = os.path.join(output_dir, 'bootstrap')
os.makedirs(bootstrap_dir)
bootstrap_i = 0
tradition_data_origins = []
tradition_data_anothers = []
for i in range(10):
    tradition_data_origins.append({})
    tradition_data_anothers.append({})
for unit_dir in os.listdir(origin_result_dir):
    if not 'eicu_DP_0-01_0-50_seed_' in unit_dir:
        continue
    for round_dir in os.listdir(os.path.join(origin_result_dir, unit_dir)):
        for i in range(10):
            rate = i + 1
            filename = f'{0.35:.3f}_{0.15*rate/10:.3f}_{0.5:.3f}.xlsx'
            file_path = os.path.join(origin_result_dir, unit_dir, round_dir, 'results', filename)
            if not os.path.exists(file_path):
                continue
            print(file_path)
            df = pd.read_excel(file_path, header=None, sheet_name='origin', engine='openpyxl')
            for j, dp in enumerate(df.loc[1]):
                data = tradition_data_origins[rate-1]
                data.setdefault(f'client_{j}', []).append(dp)
            df = pd.read_excel(file_path, header=None, sheet_name='another', engine='openpyxl')
            for j, dp in enumerate(df.loc[1]):
                data = tradition_data_anothers[rate-1]
                data.setdefault(f'client_{j}', []).append(dp)
            if rate == 10:
                excel_writer = pd.ExcelWriter(os.path.join(bootstrap_dir, f'{bootstrap_i}.xlsx'), engine='openpyxl', mode='w')
                df = pd.read_excel(file_path, header=0, sheet_name='origin_bootstrap', engine='openpyxl')
                df.to_excel(excel_writer, index=False, sheet_name='origin_bootstrap')
                df = pd.read_excel(file_path, header=0, sheet_name='another_bootstrap', engine='openpyxl')
                df.to_excel(excel_writer, index=False, sheet_name='another_bootstrap')
                excel_writer.close()
                bootstrap_i += 1
for i in range(10):
    rate = i + 1
    excel_writer = pd.ExcelWriter(os.path.join(tradition_dir, f'{int(int(rate)/10)}-{int(rate)%10}.xlsx'), engine='openpyxl', mode='w')
    pd.DataFrame(tradition_data_origins[i]).to_excel(excel_writer, index=False, sheet_name='origin')
    pd.DataFrame(tradition_data_anothers[i]).to_excel(excel_writer, index=False, sheet_name='another')
    excel_writer.close()
