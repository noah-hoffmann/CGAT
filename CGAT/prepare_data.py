import gzip as gz
import os
import argparse

import numpy as np
import warnings
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .roost_message import LoadFeaturiser


def build_dataset_prepare(data,
                          target_property=["e_above_hull_new", 'e-form', 'volume'],
                          radius=18.0,
                          fea_path="../embeddings/matscholar-embedding.json",
                          max_neighbor_number=24):
    """Use to calculate features for lists of pickle and gzipped ComputedEntry pickles (either a path to the file or the file directly), returns dictionary with all necessary inputs. If the data has no target values the target values are set to -1e8
    Always enter list of target properties"""

    def tensor2numpy(l):
        """recursively convert torch Tensors into numpy arrays"""
        if isinstance(l, torch.Tensor):
            return l.numpy()
        elif isinstance(l, str) or isinstance(l, int) or isinstance(l, float):
            return l
        elif isinstance(l, list) or isinstance(l, tuple):
            return np.asarray([tensor2numpy(i) for i in l], dtype=object)
        elif isinstance(l, dict):
            npdict = {}
            for name, val in l.items():
                npdict[name] = tensor2numpy(val)
            return npdict
        else:
            return None  # this will give an error later on

    d = CompositionDataPrepare(data,
                               fea_path=fea_path,
                               target_property=target_property,
                               max_neighbor_number=max_neighbor_number,
                               radius=radius)

    loader = DataLoader(d, batch_size=1)

    input1_ = []
    input2_ = []
    input3_ = []
    comps_ = []
    batch_comp_ = []
    if type(target_property) == list:
        target_ = {}
        for name in target_property:
            target_[name] = []
    else:
        target_ = []
    batch_ids_ = []

    for input_, target, batch_comp, batch_ids in tqdm(loader):
        if len(input_) == 1:  # remove compounds with not enough neighbors
            continue
        input1_.append(input_[0])
        comps_.append(input_[1])
        input2_.append(input_[2])
        input3_.append(input_[3])
        if isinstance(target_property, list):
            for name in target_property:
                target_[name].append(target[name])
        else:
            target_.append(target)

        batch_comp_.append(batch_comp)
        batch_ids_.append(batch_ids)

    input1_ = tensor2numpy(input1_)
    input2_ = tensor2numpy(input2_)
    input3_ = tensor2numpy(input3_)

    n = input1_[0].shape[0]
    shape = input1_.shape
    if len(shape) > 2:
        i1 = np.empty(shape=(1, shape[0]), dtype=object)
        i2 = np.empty(shape=(1, shape[0]), dtype=object)
        i3 = np.empty(shape=(1, shape[0]), dtype=object)
        i1[:, :, ] = [[input1_[l] for l in range(shape[0])]]
        input1_ = i1
        i2[:, :, ] = [[input2_[l] for l in range(shape[0])]]
        input2_ = i2
        i3[:, :, ] = [[input3_[l] for l in range(shape[0])]]
        input3_ = i3

    inputs_ = np.vstack((input1_, input2_, input3_))

    return {'input': inputs_,
            'batch_ids': batch_ids_,
            'batch_comp': tensor2numpy(batch_comp_),
            'target': tensor2numpy(target_),
            'comps': tensor2numpy(comps_)}


class CompositionDataPrepare(Dataset):
    """
    The CompositionData dataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    """

    def __init__(self, data, fea_path, target_property=['e-form'], radius=18.0, max_neighbor_number=24):
        """
        """
        if isinstance(data, str):
            self.data = pickle.load(gz.open(data, 'rb'))
        else:
            self.data = data
        self.radius = radius
        self.max_num_nbr = max_neighbor_number
        self.target_property = target_property
        assert os.path.exists(fea_path), "{} does not exist!".format(fea_path)
        self.atom_features = LoadFeaturiser(fea_path)
        self.atom_fea_dim = self.atom_features.embedding_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            cry_id = self.data[idx].data['id']
        except KeyError:
            cry_id = 'unknown'
        composition = self.data[idx].composition.formula
        try:
            crystal = self.data[idx].structure
        except:
            crystal = self.data[idx]

        elements = [element.specie.symbol for element in crystal]
        try:
            target = {}
            for name in self.target_property:
                target[name] = self.data[idx].data[name] / len(crystal.sites)
        except KeyError:
            target = {}
            warnings.warn('no target property')
            for name in self.target_property:
                target[name] = -1e8

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1])[0:self.max_num_nbr] for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea, self_fea_idx = [], [], []
        for site, nbr in enumerate(all_nbrs):
            nbr_fea_idx_sub, nbr_fea_sub, self_fea_idx_sub = [], [], []
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} does not contain enough neighbors in the cutoff to build the full graph. '
                              'If it happens frequently, consider increase '
                              'radius. Compound is not added to the feature set'.format(cry_id))
                return (torch.ones(1)), torch.ones(1), torch.ones(1), torch.ones(
                    1)  # fake input will be removed in build_dataset_prepare
            else:
                for n in range(self.max_num_nbr):
                    self_fea_idx_sub.append(site)
                for j in range(self.max_num_nbr):
                    nbr_fea_idx_sub.append(nbr[j][2])
                index = 1
                dist = nbr[0][1]
                for j in range(self.max_num_nbr):
                    if (nbr[j][1] > dist + 1e-8):
                        dist = nbr[j][1]
                        index += 1
                    nbr_fea_sub.append(index)
            nbr_fea_idx.append(nbr_fea_idx_sub)
            nbr_fea.append(nbr_fea_sub)
            self_fea_idx.append(self_fea_idx_sub)
        return (nbr_fea, elements, self_fea_idx, nbr_fea_idx), \
               target, composition, cry_id

    def get_targets(self, idx1, idx2):
        target = []
        l = []
        for el in idx2:
            l.append(self.data[el][self.target_property])
        for el in idx1:
            target.append(l[el])
        del l
        return torch.tensor(target).reshape(len(idx1), 1)


def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom"s M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_target = []
    batch_comp = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, ((atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx, _),
            target, comp, cry_id) in enumerate(dataset_list):
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]
        # batch the features together
        # batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_target.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

        # increment the id counter
        cry_base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0), torch.cat(batch_nbr_fea, dim=0), torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0), torch.cat(crystal_atom_idx)), \
           torch.cat(batch_target, dim=0), \
           batch_comp, \
           batch_cry_ids


def collate_batch2(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
        Bond features of each atom"s M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
        Target value for prediction
    batch_cif_ids: list
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_nbr_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_target = []
    batch_comp = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, ((nbr_fea, atom_fea, self_fea_idx, nbr_fea_idx),
            target, comp, cry_id) in enumerate(dataset_list):
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]
        # batch the features together
        # batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_target.append(target)
        batch_comp.append(comp)
        batch_cry_ids.append(cry_id)

        # increment the id counter
        cry_base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0), torch.cat(batch_nbr_fea, dim=0), torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0), torch.cat(crystal_atom_idx)), \
           torch.cat(batch_target, dim=0), \
           batch_comp, \
           batch_cry_ids


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, log=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.tensor((0))
        self.std = torch.tensor((1))

    def fit(self, tensor, dim=0, keepdim=False):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor, dim, keepdim)
        self.std = torch.std(tensor, dim, keepdim)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean,
                "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].cpu()
        self.std = state_dict["std"].cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='data_0_10000.pickle.gz')
    parser.add_argument('--source-dir', default='unprepared_volume_data')
    parser.add_argument('--target-dir', default='data')
    parser.add_argument('--target-file', default=None)
    args = parser.parse_args()
    test = build_dataset_prepare(os.path.join(args.source_dir, args.file))
    if args.target_file is None:
        pickle.dump(test, gz.open(os.path.join(args.target_dir, os.path.basename(args.file)), 'wb'))
    else:
        pickle.dump(test, gz.open(os.path.join(args.target_dir, args.target_file), 'wb'))


if __name__ == '__main__':
    main()
