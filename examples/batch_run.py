import multiprocessing as mp
from subprocess import call

exps = [1,2,3]
save_freqs = [25000,50000,100000]
modify_epochs = [50000,100000,200000]
total_epochs = [100000,200000,400000]

# exps = [1,2,3]
# save_freqs = [25,50,100]
# modify_epochs = [50,100,200]
# total_epochs = [100,200,400]

def run_exp(exp):
    call(exp, shell=True)

for i in range(3):
    e = exps[i]
    sf = save_freqs[i]
    me = modify_epochs[i]
    te = total_epochs[i]

    exp_str = " --exp "+ str(e) +" --save_freq " + str(sf) + " --modify_epoch " + str(me) + " --total_timesteps " + str(te)
    
    exp1 = "python3 test_sb.py --config 1 --car_idx 1 --base 0 --retain 1" + exp_str
    exp2 = "python3 test_sb.py --config 2 --car_idx 2 --base 0 --retain 1" + exp_str
    exp3 = "python3 test_sb.py --config 2 --car_idx 3 --base 0 --retain 0.95" + exp_str
    exp4 = "python3 test_sb.py --config 2 --car_idx 4 --base 0 --retain 0.75" + exp_str
    exp5 = "python3 test_sb.py --config 2 --car_idx 5 --base 0 --retain 0.5" + exp_str
    exp6 = "python3 test_sb.py --config 2 --car_idx 6 --base 0 --retain 0.25" + exp_str

    processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1, exp2, exp3, exp4, exp5, exp6]]

    for p in processes:
        p.start()
    
    for p in processes:
        p.join()



# if __name__ == "__main__":
#     # Setup a list of processes that we want to run
#     processes = [mp.Process(target=run_exp, args=(exp,)) for exp in [exp1, exp2, exp3, exp4, exp5, exp6]]

#     # Run processes
#     for p in processes:
#         p.start()

#     # Exit the completed processes
#     for p in processes:
#         p.join()