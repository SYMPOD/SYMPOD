import os
import glob
import warnings
from tqdm import tqdm
from Utils import extract_ID, get_information, append_crystal_info, save_powder_image, append_paths_info

warnings.filterwarnings("ignore")
os.mkdir(os.path.join('Data'))
os.mkdir(os.path.join('Data','Structures'))
os.mkdir(os.path.join('Data','Powder_images'))
os.mkdir(os.path.join('Data', 'Paths_info'))
ls = glob.glob(os.path.join('cod','cif','**','*.cif'), recursive = True)
for i in tqdm(range(len(ls))):
    ID = extract_ID(ls[i])
    try: 
        space_group_number, alpha, beta, gamma, a, b, c, intensities, atoms = get_information(ls[i])
        append_crystal_info(ID+'.json', ID, space_group_number, alpha, beta, gamma, a, b, c, intensities, atoms)
        save_powder_image(intensities, ID)
    except: 
         continue

ls2 = glob.glob(os.path.join('Data','Structures', '*.json'))
append_paths_info(ls2)