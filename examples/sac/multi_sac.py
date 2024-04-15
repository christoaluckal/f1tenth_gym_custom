import multiprocessing as mp
from subprocess import call

def run_exp(exp):
    call(exp, shell=True)

configs = [1,2,3]

for config in configs:
    exp = f"python main.py --config {config} --num_steps 500000"
    run_exp(exp)
    