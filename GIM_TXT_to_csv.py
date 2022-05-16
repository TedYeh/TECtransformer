import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import imageio, re

'''
# to move file to txt dir
zip_path = 'GIM_CODE/2019'
pathes = os.listdir(zip_path)

for path in pathes:    
    if 'Z' in path:continue
    os.replace(zip_path+'/'+path, 'GIM_CODE/txt/2019/'+path)
'''

def plot_init():
    plt.rcParams.update({"figure.figsize":[10, 6]})
    plt.rcParams.update({"figure.autolayout":False}) 
    plt.rcParams.update({'font.size': 15})   
    #plt.xlim(-180, 180)
    #plt.ylim(-87.5, 87.5)

def plot(np_data, points=None, datetime=[0,0,0,0], type_='GIM Code', use_model=''):
    
    if type_=='GIM-Pred':
        cmap = plt.cm.get_cmap('RdBu')
    else:cmap = plt.cm.get_cmap('jet')       

    date_format = f'{datetime[0]}-{datetime[1]:02d}-{datetime[2]:02d} {datetime[3]:02d}'
    plt.title(f"GLOBAL IONOSPHERE MAPS\n{date_format}:00 UT\n{use_model} {type_}")    
     
    if points:
        for i in range(len(points[0])):plt.scatter(points[0][i][0], points[0][i][1], marker='o', color='m', label='diff_max'if i==0 else'')
        for i in range(len(points[1])):plt.scatter(points[1][i][0], points[1][i][1], marker='o', color='w', label='diff_min'if i==0 else'')        
        plt.legend(loc='upper right')
    
    if type_ == 'GIM-Pred':
        #im = plt.imshow(np_data, cmap = cmap)
        im = plt.imshow(np_data, extent=[-180, 180, -87.5, 87.5], cmap = cmap)
        norm = mpl.colors.BoundaryNorm(list(range(-25,30,5)), cmap.N)#norm = mpl.colors.Normalize(vmin=-30, vmax=30)
    else:
        #im = plt.imshow(np_data, cmap = cmap)
        im = plt.imshow(np_data, extent=[-180, 180, -87.5, 87.5], cmap = cmap)
        norm = mpl.colors.BoundaryNorm(list(range(0,400,50)), cmap.N)#norm = mpl.colors.Normalize(vmin=0, vmax=350)   
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), shrink=0.65)
    #plt.colorbar(im, shrink=0.65)
    plt.xlabel('GEOGRAPHIC LONGITUDE')
    plt.ylabel('GEOGRAPHIC LATITUDE')
    if type_ == 'GIM Code':plt.savefig(f'img/2020_real/{date_format}.png')
    elif type_ == 'GIM-Pred':plt.savefig(f'img/2020_GIM-Pred/{date_format}.png')
    else:plt.savefig(f'img/2020_pred/{date_format}.png')
    
    plt.close()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]

def generate_gif(img_dir, datetime, use_model=''):
    filenames = os.listdir(img_dir)
    filenames.sort(key = natural_keys)
    with imageio.get_writer(f'gif/{datetime[0]}-{datetime[1]:02d}-{datetime[2]:02d}_{use_model}.gif', mode='I', fps=1) as writer:
        for filename in filenames:
            image = imageio.imread(img_dir+filename)
            writer.append_data(image)

def read_file(filename):
    with open(filename, 'r') as code:
        str_list = []
        tec_for_hour = []
        times = []
        LAT, latitudes = [], []
        n, nn = 1, 0
        while True:
            line = code.readline()
            n += 1
            if not line:break 
            if 'EPOCH OF CURRENT MAP' in line: #record date & time of TEC
                time_data = [int(d) for d in line.split()[:4]]
                times.append(time_data)
                
            if 'LAT/LON1/LON2/DLON/H' in line: #record TEC
                tec_data = []
                latitudes.append(eval(line.replace('-180.0', ' ').split()[0])) #add latitude trying to let model know the position

                for _ in range(5):
                    tec_data.append(code.readline().replace('"', '').replace('!', '').replace('\x1a', ''))
                    
                full_data = [int(d) for d in ' '.join(tec_data).split()]    
                tec_for_hour.append(full_data)
                nn += 1

            if nn == 71:
                nn = 0            
                str_list.append(tec_for_hour)
                LAT.append(latitudes)
                tec_for_hour, latitudes = [], []
        #print(np.array(str_list)[0])
        return str_list[:25], times[:25]#, LAT[:25]

if __name__ == "__main__":
    plot_init()
    full_data, datetime = read_file('txt/2020/CODG0500.20I')
    print(full_data, datetime)
    '''
    file_list = os.listdir('GIM_CODE/txt/2019')
    for file_name in file_list:
        full_data, datetime = read_file('GIM_CODE/txt/2019/'+file_name)
        full_data = np.array(full_data)
        
        for i in range(full_data.shape[0]):
            plot(full_data[i], datetime[i])
    
    full_data, datetime = read_file('txt/2020/CODG0500.20I')
    full_data = np.array(full_data)
        
    for i in range(full_data.shape[0]):
        plot(full_data[i], datetime[i], type_='Turth')
    generate_gif('img/2020_real/', datetime[0])
    '''