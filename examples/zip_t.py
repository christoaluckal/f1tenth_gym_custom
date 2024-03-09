import zipfile

with zipfile.ZipFile('ppo_f1tenth_1000.zip', 'r') as zip_ref:
    zip_ref.extractall('logs/ppo_f1tenth_1000')

