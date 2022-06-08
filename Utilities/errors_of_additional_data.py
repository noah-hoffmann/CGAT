from CGAT.lightning_module import LightningModel, collate_fn
from CGAT.data import CompositionData
from torch.utils.data import DataLoader
import pandas as pd
import os
import glob
from get_additional_data import get_composition
from tqdm import tqdm
import numpy as np
import re


def get_seed(path):
    pattern = re.compile(r'f-(\d+)_')
    return int(pattern.search(path).group(1))


def main():
    data_paths = glob.glob(os.path.join("additional_data", "*", "*.pickle.gz"))
    assert len(data_paths) > 0
    print(f"Found {len(data_paths)} datasets")
    sizes = [200_000, 250_000]
    runs = ["f-0_t-2022-03-02_16-24-35",
            "f-0_t-2022-03-03_22-20-11",
            ]
    model_paths = [glob.glob(os.path.join("tb_logs",
                                          "runs",
                                          "{run}",
                                          "*.ckpt").format(run=run))[0] for run in runs]
    # model_paths = sorted(glob.glob(os.path.join('new_active_learning', 'checkpoints', '*', '*.ckpt')), key=get_seed)
    # seeds = list(map(get_seed, model_paths))
    df = pd.DataFrame(columns=['comp', 'training set size', 'mae'])
    for i, model_path in zip(sizes, tqdm(model_paths)):
        model = LightningModel.load_from_checkpoint(model_path, train=False)
        model = model.cuda()

        for path in tqdm(data_paths):
            dataset = CompositionData(
                data=path,
                fea_path="embeddings/matscholar-embedding.json",
                max_neighbor_number=model.hparams.max_nbr,
                target=model.hparams.target
            )
            loader = DataLoader(dataset, batch_size=500, shuffle=False, collate_fn=collate_fn)
            comp = get_composition(path)
            errors = []
            for batch in loader:
                _, _, pred, target, _ = model.evaluate(batch)
                errors.append(np.reshape(np.abs(pred - target), (-1,)))
            df.loc[len(df)] = [comp, i, np.mean(np.concatenate(errors))]
        savepath = 'additional_data/random.csv'
        if not os.path.isfile(savepath):
            df.to_csv(savepath, index=False)
        else:
            old = pd.read_csv(savepath)
            merged = pd.concat((old, df), ignore_index=True)
            merged.to_csv(savepath, index=False)


if __name__ == '__main__':
    main()
