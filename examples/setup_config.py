import numpy as np
import argparse
from scipy.interpolate import CubicSpline

def averageKappa(waypoints, window_size=50):
        
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
        xdot = cx(1)
        ydot = cy(1)
        xddot = cx(2)
        yddot = cy(2)
        kappa.append((xdot*yddot - ydot*xddot)/(xdot**2 + ydot**2)**1.5)


    return np.mean(kappa)
    
    
def main():
    """
    main entry point
    """

    import os
    import matplotlib.pyplot as plt
    map_location = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','gym','f110_gym','unittest')

    from f110_gym.unittest.collate import getConfigList
    
    while True:
        config_dict = getConfigList(csv_f=os.path.join(map_location,'generated.csv'),scale_search=2.0)
        trs = config_dict['tr']
        scale = config_dict['scale']
        
        print(f"Available maps: {trs}")

        configs = []
        
        for t in trs:
            map_config = {
                'map_ext': '.png',
                'map': os.path.join(map_location,f'maps/map_{t}_{scale}'),
                'waypoints': os.path.join(map_location,f'centerline/map_{t}_{scale}.csv'),
                'reset_pose': [0.0,0.0,np.pi/2]
            }
            plt.imshow(plt.imread(map_config['map']+'.png'))
            # plt.show()
            # yes = input("Is this the map you want to use? (y/n): ")
            # if yes == 'y':
            #     configs.append(map_config)
            configs.append(map_config)
            plt.close()
            
        print(f"Total number of maps: {len(configs)}")
        # Arrange in nxn square grid
        n = int(np.ceil(np.sqrt(len(configs))))
        fig, axs = plt.subplots(n,n,figsize=(20,20))
        for i,config in enumerate(configs):
            ax = axs[i//n,i%n]
            ax.imshow(plt.imread(config['map']+'.png'))
            ax.set_title(f"Map {i}")
        
        plt.show()
        
        y = input("Are you satisfied with the maps? (y/n): ")
        if y == 'y':
            break
        
    curvatures = []

    for i,config in enumerate(configs):
        avg_kappa = averageKappa(np.loadtxt(config['waypoints'],delimiter=','))
        print(f"Map {i}: Kappa: {avg_kappa}")
        curvatures.append([i,avg_kappa])
        
    # sort by curvature
    curvatures = sorted(curvatures,key=lambda x: x[1])
    
    print("Sorted maps by curvature: ")
    for i,c in curvatures:
        print(f"Map {i}: Kappa: {c}")
    
    selected_maps = [2,7,6,5]
    
    final_configs = [configs[i] for i in selected_maps]
    
    print("Selected maps: ")
    for i,config in enumerate(final_configs):
        print(f"Map {i}: {config['map']}")
    
    n = int(np.ceil(np.sqrt(len(final_configs))))
    fig, axs = plt.subplots(n,n,figsize=(20,20))
    for i,config in enumerate(final_configs):
        ax = axs[i//n,i%n]
        ax.imshow(plt.imread(config['map']+'.png'))
        ax.set_title(f"Map {i}")
        
    plt.show()
            

        
    
    
    
    

    

if __name__ == '__main__':
    main()
