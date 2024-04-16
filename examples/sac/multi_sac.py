import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

configs = [1,2]

for config in configs:
    exp = f"python main.py --config {config} --num_steps 1000000"
    run_exp(exp)
    