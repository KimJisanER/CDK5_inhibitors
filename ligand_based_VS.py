dir_data = '/home/kimjisan/PBI/data' # train.csv, dev.csv, test.csv가 위치한 폴더를 적어주세요
dir_output = '/home/kimjisan/PBI/outputs/rgcn'  # 각각의 tgm.nn.data.Data 가 저장될 폴더를 적어주세요.

import os
import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

import rdkit
import rdkit.Chem as Chem

from collections import defaultdict

# dir_output 폴더가 없으면 폴더를 만들고 각 폴더 안에 train, dev, test 폴더를 만듭니다.
if not os.path.exists(dir_output):
        os.makedirs(f'{dir_output}/CHEMBL')
        os.makedirs(f'{dir_output}/ZINC')

train = pd.read_csv(f'{dir_data}/train.csv')
test = pd.read_csv(f'{dir_data}/test_kinase.csv')
#
full = pd.concat([train, test], axis=0, ignore_index=True)
full['y'] = full["pChEMBL Value"]
full['folder'] = full['uid'].apply(lambda x: ''.join(list(filter(str.isalpha, x))))


def create_encoders(df):
    """
    데이터프레임 (df)의 각 row를 방문하여, 원자, 연결 타입, 연결 스테레오에 대한 LabelEncoder를 만들어주는 함수
    """
    encoder_atom = defaultdict(lambda: len(encoder_atom))
    encoder_bond_type = defaultdict(lambda: len(encoder_bond_type))
    encoder_bond_stereo = defaultdict(lambda: len(encoder_bond_stereo))
    encoder_bond_type_stereo = defaultdict(lambda: len(encoder_bond_type_stereo))

    target = df['Smiles'].values
    total_num = len(target)
    for i, smiles in enumerate(target):
        print(f'Creating the label encoders for atoms, bond_type, and bond_stereo ... [{i + 1} / {total_num}] done !',
              end='\r')
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)

        for atom in m.GetAtoms():
            encoder_atom[atom.GetAtomicNum()]

        for bond in m.GetBonds():
            encoder_bond_type[bond.GetBondTypeAsDouble()]
            encoder_bond_stereo[bond.GetStereo()]
            encoder_bond_type_stereo[(bond.GetBondTypeAsDouble(), bond.GetStereo())]

    return encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo

encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo = create_encoders(full)


def row2data(row, encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo):
    smiles = row.Smiles
    y = row.y

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    # Creating node feature vector
    num_nodes = len(list(m.GetAtoms()))
    x = np.zeros((num_nodes, len(encoder_atom.keys())))
    for i in m.GetAtoms():
        x[i.GetIdx(), encoder_atom[i.GetAtomicNum()]] = 1

    x = torch.from_numpy(x).float()

    # Creating edge_index and edge_type
    i = 0
    num_edges = 2 * len(list(m.GetBonds()))
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_type = np.zeros((num_edges,), dtype=np.int64)
    for edge in m.GetBonds():
        # Getting bond information
        u = min(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        v = max(edge.GetBeginAtomIdx(), edge.GetEndAtomIdx())
        bond_type = edge.GetBondTypeAsDouble()
        bond_stereo = edge.GetStereo()
        bond_label = encoder_bond_type_stereo[(bond_type, bond_stereo)]

        # Storing information
        edge_index[0, i] = u
        edge_index[1, i] = v
        edge_index[0, i + 1] = v
        edge_index[1, i + 1] = u
        edge_type[i] = bond_label
        edge_type[i + 1] = bond_label
        i += 2

    edge_index = torch.from_numpy(edge_index)
    edge_type = torch.from_numpy(edge_type)

    # Creating y
    y = torch.tensor([y]).float()

    # Wrapping all together
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        uid=row.uid
    )

    return data

print('')
for i, row in full.iterrows():
    print(f'Converting each data into torch.Data ... [{i+1} / {len(full)}] done !', end='\r')
    data = row2data(row, encoder_atom, encoder_bond_type, encoder_bond_stereo, encoder_bond_type_stereo)
    fpath = f'{dir_output}/{row.folder}/{row.uid}.pt'
    torch.save(data, fpath)
