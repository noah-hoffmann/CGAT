import pickle
import gzip as gz
from tqdm import tqdm
from pymatgen.entries.computed_entries import ComputedStructureEntry
import os
from glob import glob
import matplotlib.pyplot as plt
from multiprocessing import Pool


def load(file):
    return pickle.load(gz.open(file))


def get_properties(data: list[ComputedStructureEntry], property_keys):
    properties = {key: [] for key in property_keys}
    for entry in data:
        for key in property_keys:
            properties[key].append(entry.data[key])
    return properties


def load_properties(file, property_keys=('e_above_hull_new', 'e-form', 'volume')):
    return get_properties(load(file), property_keys)


def main():
    path = os.path.join('unprepared_volume_data')
    files = sorted(glob(os.path.join(path, '*.pickle.gz')))

    properties = {'e_above_hull_new': [], 'e-form': [], 'volume': []}

    # load files in parallel to speed up the loading
    with Pool() as pool:
        for data in tqdm(pool.imap_unordered(load_properties, files), total=len(files)):
            for key, item in data.items():
                properties[key].extend(item)

    # for file in tqdm(files[:10]):
    #     data = get_properties(load(file), properties.keys())
    #     for key, item in data.items():
    #         properties[key].extend(item)

    save_path = os.path.join('histogram', 'active')
    os.makedirs(save_path, exist_ok=True)

    plt.figure()
    plt.hist(properties['e_above_hull_new'], bins=15)
    plt.xlabel('distance to the convex hull [eV atom$^{-1}$]', fontsize=17)
    plt.ylabel('count', fontsize=17)
    plt.yscale('log')
    plt.gca().tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'hull_histogram.pdf'))

    plt.figure()
    plt.hist(properties['e-form'], bins=15)
    plt.xlabel('formation energy [eV atom$^{-1}$]', fontsize=17)
    plt.ylabel('count', fontsize=17)
    plt.yscale('log')
    plt.gca().tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'formation_histogram.pdf'))

    plt.figure()
    plt.hist(properties['volume'], bins=15)
    plt.xlabel(r'volume [$\AA^3$ atom$^{-1}$]', fontsize=17)
    plt.ylabel('count', fontsize=17)
    plt.yscale('log')
    plt.gca().tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'volume_histogram.pdf'))

    plt.show()


if __name__ == '__main__':
    main()
