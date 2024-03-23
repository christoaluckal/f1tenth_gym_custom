from sklearn.cluster import KMeans
import math

def getConfigList(csv_f=None,scale_search=None):
    if csv_f is None:
        with open("generated.csv","r") as f:
            data = f.readlines()
    else:
        with open(csv_f,"r") as f:
            data = f.readlines()
        
    data = [d[:-2] for d in data]
    data = [d.split(",") for d in data]
    data = [[float(d) for d in dd] for dd in data]

    cols = ["turn_rate","scale","px_length","length"]

    import pandas as pd

    df = pd.DataFrame(data,columns=cols)
    
    print(df)

    if scale_search is None:
        scales = df["scale"].unique()
        yes = ""
        while yes != "y":
            scale_search = input(f"Enter scale to search from {scales}: ")
            scale_search = float(scale_search)
            if scale_search in scales:
                print(f"Searching for scale {scale_search}")
                print(df[df["scale"]==scale_search])
                yes = input("Is this the scale you want to search? (y/n): ")
            else:
                print("Invalid scale. Try again.")
    


    df = df[df["scale"]==scale_search]

    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(df[["length"]])

    df["cluster"] = kmeans.labels_
    df["centroid"] = kmeans.cluster_centers_[kmeans.labels_]


    configs = {}
    centroids = []

    for c in df["centroid"].unique():
        configs[c] = {'tr':[],'scale':0}
        centroids.append(c)

    for c in df["cluster"].unique():
        dfc = df[df["cluster"]==c]
        dfc.sort_values(by="turn_rate",inplace=True)
        for i in range(len(dfc)):
            tr = dfc.iloc[i]["turn_rate"]
            tr = math.ceil(tr*1000)
            scale = dfc.iloc[i]["scale"]
            scale = math.ceil(scale*10)
            configs[dfc.iloc[i]["centroid"]]['tr'].append(tr)
            configs[dfc.iloc[i]["centroid"]]['scale'] = scale
            
    print("Select index of centroid to save:")
    for i,c in enumerate(centroids):
        print(f"{i}: {c}")
    
    while True:    
        idx = int(input("Enter index: "))
        if idx < len(centroids):
            break
        else:
            print("Invalid index. Try again.")
    
    return configs[centroids[idx]]
        
    
if __name__ == "__main__":
    print(getConfigList())