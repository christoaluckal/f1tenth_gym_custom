import numpy as np
import argparse
from scipy.interpolate import CubicSpline
import math

def averageKappa(waypoints, window_size=30):
        
    waypoints = np.array(waypoints)
    x = waypoints[:,0].flatten()
    y = waypoints[:,1].flatten()    
    kappa = []

    ts = np.arange(window_size,len(x),window_size)
    for t in ts:
        xs = x[t-window_size:t]
        ys = y[t-window_size:t]
        ts = np.linspace(0,window_size,len(xs))
        cx = CubicSpline(ts,xs)
        cy = CubicSpline(ts,ys)
        xdot = cx(ts,1)
        ydot = cy(ts,1)
        xddot = cx(ts,2)
        yddot = cy(ts,2)
        kappa.append(abs(xdot*yddot - ydot*xddot)/((xdot**2 + ydot**2)**1.5))


    return np.mean(kappa)

def totalKappa(waypoints):
    waypoints = np.array(waypoints)
    x = waypoints[:,0].flatten()
    y = waypoints[:,1].flatten()    
    kappa = []

    for i in range(1,len(x)-1):
        xdot = x[i+1] - x[i-1]
        ydot = y[i+1] - y[i-1]
        xddot = x[i+1] - 2*x[i] + x[i-1]
        yddot = y[i+1] - 2*y[i] + y[i-1]
        kappa.append((xdot*yddot - ydot*xddot)/(xdot**2 + ydot**2)**1.5)

    return np.sum(kappa)
    
    
def main():
    """
    main entry point
    """

    import os
    import matplotlib.pyplot as plt
    map_location = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','gym','f110_gym','unittest')
    gen_csv_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','gym','f110_gym','unittest','generated.csv')

    from f110_gym.unittest.collate import getConfigList
    
    while True:
        config_dict = getConfigList(csv_f=gen_csv_loc,scale_search=1.5)
        trs = config_dict['tr']
        scale = config_dict['scale']
        
        print(f"Available maps: {trs}")

        configs = []
        
        for t in trs:
            map_config = {
                'map_ext': '.png',
                'map': os.path.join(map_location,f'maps/map_{t}_{scale}') if t>=100 else os.path.join(map_location,f'maps/map_0{t}_{scale}'),
                'waypoints': os.path.join(map_location,f'centerline/map_{t}_{scale}.csv') if t>=100 else os.path.join(map_location,f'centerline/map_0{t}_{scale}.csv'),
                'reset_pose': [0.0,0.0,np.pi/2]
            }
            # plt.show()
            # yes = input("Is this the map you want to use? (y/n): ")
            # if yes == 'y':
            #     configs.append(map_config)
            configs.append(map_config)
            
        print(f"Total number of maps: {len(configs)}")
        y = input("Are you satisfied with the maps? (y/n): ")
        if y == 'y':
            break
        
    curvatures = []

    for i,config in enumerate(configs):
        print(config)
        avg_kappa = averageKappa(np.loadtxt(config['waypoints'],delimiter=','))
        curvatures.append([i,avg_kappa])
        
    # sort by curvature
    curvatures = sorted(curvatures,key=lambda x: x[1])
    
    print("Sorted maps by curvature: ")
    for i,curvature in enumerate(curvatures):
        print(f"Map {curvature[0]}: Kappa {curvature[1]}")
    
    n = int(math.ceil(len(curvatures)**0.5))
    fig, axs = plt.subplots(n,n,figsize=(15,15))
    for i,_ in enumerate(curvatures):
        ax = axs[i//n,i%n]
        ax.imshow(plt.imread(configs[curvatures[i][0]]['map']+'.png'))
        ax.set_title(f"Map {curvatures[i][0]}: Kappa {curvatures[i][1]}")
    
    plt.show()
    
    '''
    fig, axs = plt.subplots(n,n,figsize=(15,15))
    for i,_ in enumerate(curvatures):
        ax = axs[i//n,i%n]
        ax.imshow(plt.imread(configs[curvatures[i][0]]['map']+'.png'))
        ax.set_title(f"Map {curvatures[i][0]}: Kappa {curvatures[i][1]}")

        
    plt.show()
    
    selected_maps = [0,1,2,3]
    
    final_configs = [configs[i] for i in selected_maps]
    
    print("Selected maps: ")
    for i,config in enumerate(final_configs):
        print(f"Map {i}: {config['map']}")
        
    n = int(math.ceil(len(final_configs)**0.5))

    fig, axs = plt.subplots(n,n,figsize=(15,15))
    for i,config in enumerate(final_configs):
        ax = axs[i//n,i%n]
        ax.imshow(plt.imread(config['map']+'.png'))
        ax.set_title(f"Map {i}")
        
    plt.show()
    
    print(final_configs)
    '''

    eval_config = curvatures[0]
    easy_config = curvatures[1]
    medium_config = curvatures[len(curvatures)//2]
    hard_config = curvatures[-1]

    curvatures = [eval_config,easy_config,medium_config,hard_config]
    
    num_maps = 4
    selected_maps = [curvature[0] for curvature in curvatures[:num_maps]]
    final_configs = [configs[i] for i in selected_maps]
    
    print("Selected maps: ")
    for i,config in enumerate(final_configs):
        print(f"Map {i}: {config['map']}")
        
    n = int(math.ceil(len(final_configs)**0.5))
    fig, axs = plt.subplots(n,n,figsize=(15,15))
    
    for i,config in enumerate(final_configs):
        ax = axs[i//n,i%n]
        ax.imshow(plt.imread(config['map']+'.png'))
        ax.set_title(f"Map {i}")
        
    plt.show()

    for i,config in enumerate(final_configs):
        map_loc = config['map']
        map_loc = map_loc.split('f1tenth_gym_custom')[1]
        config['map'] = map_loc

        wpt_loc = config['waypoints']
        wpt_loc = wpt_loc.split('f1tenth_gym_custom')[1]
        config['waypoints'] = wpt_loc


    import pickle
    
    with open('maps.pkl','wb') as f:
        pickle.dump(final_configs,f)
    
    
    


        
    
    
    
    

    

if __name__ == '__main__':
    main()
