import multiprocessing as mp
from subprocess import call

exp1 = "python3 test_sb1.py"
exp2 = "python3 test_sb2.py"
exp3 = "python3 test_sb3.py"

def run_exp(exp):
    call(exp, shell=True)

if __name__ == "__main__":
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=run_exp, args=(exp1,)),
                 mp.Process(target=run_exp, args=(exp2,)),
                 mp.Process(target=run_exp, args=(exp3,))]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()