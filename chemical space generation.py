#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description: Generate donor/acceptor chemical-space combinations and export structures.
Usage: python "chemical space generation.py"
Author: Yibin ZHANG
"""

import os.path
from dataclasses import dataclass
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit import RDLogger
logger = RDLogger.logger()
logger.setLevel(RDLogger.ERROR)


@dataclass(frozen=True)
class ProjectPaths:
    project_root: str

    @property
    def image_dir(self):
        return os.path.join(self.project_root, 'img')

    @property
    def results_dir(self):
        return os.path.join(self.project_root, 'results')

    @property
    def da_dir(self):
        return os.path.join(self.results_dir, 'DA')

    @property
    def dad_dir(self):
        return os.path.join(self.results_dir, 'DAD')

    @property
    def virtual_screen_dir(self):
        return os.path.join(self.results_dir, 'virtual_screen_data')

    @property
    def virtual_screen_aa_dir(self):
        return os.path.join(self.virtual_screen_dir, 'AA')

    @property
    def virtual_screen_dad_dir(self):
        return os.path.join(self.virtual_screen_dir, 'DAD')

    @property
    def virtual_screen_aa_dad_dir(self):
        return os.path.join(self.virtual_screen_aa_dir, 'DAD')


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_smiles_npz(save_path, file_name, ids, smiles, smiles_key='smiles'):
    ensure_dir(save_path)
    np.savez_compressed(os.path.join(save_path, file_name), id=np.array(ids), **{smiles_key: np.array(smiles)})


def build_ids(prefix, size):
    return [f'{prefix}_{idx + 1}' for idx in range(size)]


PATHS = None
DA_save_path = None
DAD_save_path = None

def addnote(mol):
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx())) #'molAtomMapNumber'
    return mol

def show_mol(mol):
    img = Draw.MolToImage(addnote(mol))
    img.show()

def show_mols(mols, highlight, save_path, name, length):
    for i in range(len(mols)):
        mols[i] = addnote(mols[i])
    # img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(500, 500), legends=[Chem.MolToSmiles(mol) for mol in mols], highlightAtomLists=highlight)
    img = Draw.MolsToGridImage(mols, molsPerRow=12, subImgSize=(1000, 1000), legends=[name + str(i) for i in range(1, length + 1)], highlightAtomLists=highlight)
    # img.show()
    img.save(save_path)

def connectmol(mol1, mol2, pos1, pos2):
    mol = Chem.EditableMol(Chem.CombineMols(mol1, mol2))
    mol.AddBond(pos1, pos2, order = Chem.rdchem.BondType.SINGLE)
    mol = mol.GetMol()
    return mol

def combineDA(D_mols, pi_mols, A_mols, D_pos, pi_pos, A_pos):
    smis = []
    DAcount = 1
    for a in range(len(A_mols)):
        acceptor = A_mols[a]
        acceptor_pos = A_pos[a][0]
        for d in range(len(D_mols)):
            donor = D_mols[d]
            donor_pos = D_pos[d][0]
            try:
                Chem.Kekulize(donor)
                Chem.Kekulize(acceptor)
                mol = connectmol(donor, acceptor, donor_pos, donor.GetNumAtoms() + acceptor_pos)
                Chem.SanitizeMol(mol)
                AllChem.EmbedMolecule(mol)
                smiles = Chem.MolToSmiles(mol)
                mol = Chem.MolFromSmiles(smiles)
                # img = Draw.MolToImage(mol, size=(500, 500))
                # img.save(os.path.join(DA_save_path, 'DA_' + str(DAcount) + '.jpg'))
                DAcount += 1
                smis.append(smiles)
                print('Successful Molecule {}, {}'.format(d, a))
            except:
                pass
                print('Failed Molecule {}, {}'.format(d, a))
            for p in range(len(pi_mols)):
                pi = pi_mols[p]
                pi_pos1 = pi_pos[p][0]
                pi_pos2 = pi_pos[p][1]
                try:
                    Chem.Kekulize(donor)
                    Chem.Kekulize(acceptor)
                    Chem.Kekulize(pi)
                    mol = connectmol(donor, pi, donor_pos, donor.GetNumAtoms() + pi_pos1)
                    mol = connectmol(mol, acceptor, donor.GetNumAtoms() + pi_pos2, donor.GetNumAtoms() + pi.GetNumAtoms() + acceptor_pos)
                    Chem.SanitizeMol(mol)
                    # AllChem.EmbedMolecule(mol)
                    smiles = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    # img = Draw.MolToImage(addnote(mol), size=(500, 500))
                    # img.save(os.path.join(DA_save_path, 'DA_' + str(DAcount) + '.jpg'))
                    DAcount += 1
                    smis.append(smiles)
                    print('Successful Molecule {}, {}, {}'.format(d, a, p))
                except:
                    pass
                    print('Failed Molecule {}, {}, {}'.format(d, a, p))
    # smis = pd.DataFrame(smis)
    # id = pd.DataFrame(columns=['id'])
    # for i in range(len(smis)):
    #     id.loc[i] = 'DA_' + str(i + 1)
    # smis = pd.merge(id, smis, left_index=True, right_index=True, sort=False)
    # smis.columns = ['id', 'smiles']
    # smis.to_csv(os.path.join(save_path, 'DA.csv'), index=False)
    # print('DA structure finished, and save to {}'.format(os.path.join(save_path, 'DA.csv')))
    smis = np.array(smis)
    smis = np.unique(smis)
    id = []
    for i in range(len(smis)):
        id.append('DA_' + str(i + 1))
    id = np.array(id)
    save_smiles_npz(PATHS.results_dir, 'DA_structures.npz', id, smis)
    print('DAD structure finished, and save to {}'.format(os.path.join(npz_path, 'DA_structures.npz')))

def combineDAD(D_mols, pi_mols, AA_mols, D_pos, pi_pos, AA_pos):
    # AllChem.EmbedMolecule(mol)
    DADcount = 1
    smis = []
    for d in range(len(D_mols)):
        donor = D_mols[d]
        donor_pos = D_pos[d][0]
        for aa in range(len(AA_mols)):
            acceptor = AA_mols[aa]
            acceptor_pos1 = AA_pos[aa][0]
            acceptor_pos2 = AA_pos[aa][1]
            try:
                Chem.Kekulize(donor)
                Chem.Kekulize(acceptor)
                mol = connectmol(donor, acceptor, donor_pos, donor.GetNumAtoms() + acceptor_pos1)
                mol = connectmol(mol, donor, donor.GetNumAtoms() + acceptor_pos2, donor.GetNumAtoms() + acceptor.GetNumAtoms() + donor_pos)
                Chem.SanitizeMol(mol)
                # AllChem.EmbedMolecule(mol)
                smiles = Chem.MolToSmiles(mol)
                mol = Chem.MolFromSmiles(smiles)
                img = Draw.MolToImage(mol, size=(300, 300))
                img.save(os.path.join(DAD_save_path, 'DAD_' + str(DADcount) + '.jpg'))
                DADcount += 1
                smis.append(smiles)
                print('Successful Molecule {}, {}'.format(d, aa))
            except:
                pass
                print('Failed Molecule {}, {}'.format(d, aa))
            for p in range(len(pi_mols)):
                pi = pi_mols[p]
                pi_pos1 = pi_pos[p][0]
                pi_pos2 = pi_pos[p][1]
                try:
                    Chem.Kekulize(donor)
                    Chem.Kekulize(acceptor)
                    Chem.Kekulize(pi)
                    mol = connectmol(donor, pi, donor_pos, donor.GetNumAtoms() + pi_pos1)
                    mol = connectmol(mol, acceptor, donor.GetNumAtoms() + pi_pos2, donor.GetNumAtoms() + pi.GetNumAtoms() + acceptor_pos1)
                    mol = connectmol(mol, pi, donor.GetNumAtoms() + pi.GetNumAtoms() + acceptor_pos2, donor.GetNumAtoms() + pi.GetNumAtoms() + acceptor.GetNumAtoms() + pi_pos2)
                    mol = connectmol(mol, donor, donor.GetNumAtoms() + pi.GetNumAtoms() + acceptor.GetNumAtoms() + pi_pos1, donor.GetNumAtoms() + pi.GetNumAtoms() + acceptor.GetNumAtoms() + pi.GetNumAtoms() + donor_pos)
                    Chem.SanitizeMol(mol)
                    # AllChem.EmbedMolecule(mol)
                    smiles = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    img = Draw.MolToImage(mol, size=(300, 300))
                    img.save(os.path.join(DAD_save_path, 'DAD_' + str(DADcount) + '.jpg'))
                    DADcount += 1
                    smis.append(smiles)
                    print('Successful Molecule {}, {}, {}'.format(d, aa, p))
                except:
                    pass
                    print('Failed Molecule {}, {}, {}'.format(d, aa, p))
    smis = pd.DataFrame(smis)
    id = pd.DataFrame(columns=['id'])
    for i in range(len(smis)):
        id.loc[i] = 'DAD_' + str(i + 1)
    smis = pd.merge(id, smis, left_index=True, right_index=True, sort=False)
    smis.columns = ['id', 'smiles']
    smis.to_csv(os.path.join(save_path, 'DAD.csv'), index=False)
    print('DAD structure finished, and save to {}'.format(os.path.join(save_path, 'DAD.csv')))

def combineD1AD2(D_mols, pi_mols, AA_mols, D_pos, pi_pos, AA_pos):
    img_save_path = PATHS.virtual_screen_dad_dir
    ensure_dir(img_save_path)
    # AllChem.EmbedMolecule(mol)
    DADcount = 1
    save_path = PATHS.virtual_screen_dir
    ensure_dir(save_path)
    smis = []
    for d1 in range(len(D_mols)):
        print('Current D1: ', d1)
        donor1 = D_mols[d1]
        donor1_pos = D_pos[d1][0]
        for d2 in range(len(D_mols)):
            donor2 = D_mols[d2]
            donor2_pos = D_pos[d2][0]
            for aa in range(len(AA_mols)):
                acceptor = AA_mols[aa]
                acceptor_pos1 = AA_pos[aa][0]
                acceptor_pos2 = AA_pos[aa][1]
                try:
                    # Chem.Kekulize(donor1)
                    # Chem.Kekulize(donor2)
                    # Chem.Kekulize(acceptor)
                    mol = connectmol(donor1, acceptor, donor1_pos, donor1.GetNumAtoms() + acceptor_pos1)
                    mol = connectmol(mol, donor2, donor1.GetNumAtoms() + acceptor_pos2, donor1.GetNumAtoms() + acceptor.GetNumAtoms() + donor2_pos)
                    # Chem.SanitizeMol(mol)
                    # AllChem.EmbedMolecule(mol)
                    smiles = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    # img = Draw.MolToImage(mol, size=(300, 300))
                    # img.save(os.path.join(img_save_path, 'DAD_' + str(DADcount) + '.jpg'))
                    DADcount += 1
                    smis.append(smiles)
                    # print('Successful Molecule {}, {}'.format(d1, aa))
                except:
                    print('Failed Molecule {}, {}'.format(d1, aa))
                for p1 in range(len(pi_mols)):
                    pi1 = pi_mols[p1]
                    pi1_pos1 = pi_pos[p1][0]
                    pi1_pos2 = pi_pos[p1][1]
                    for p2 in range(len(pi_mols)):
                        pi2 = pi_mols[p2]
                        pi2_pos1 = pi_pos[p2][0]
                        pi2_pos2 = pi_pos[p2][1]
                        try:
                            # Chem.Kekulize(donor1)
                            # Chem.Kekulize(donor2)
                            # Chem.Kekulize(acceptor)
                            # Chem.Kekulize(pi1)
                            # Chem.Kekulize(pi2)
                            mol = connectmol(donor1, pi1, donor1_pos, donor1.GetNumAtoms() + pi1_pos1)
                            mol = connectmol(mol, acceptor, donor1.GetNumAtoms() + pi1_pos2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor_pos1)
                            mol = connectmol(mol, pi2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor_pos2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor.GetNumAtoms() + pi2_pos2)
                            mol = connectmol(mol, donor2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor.GetNumAtoms() + pi2_pos1, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor.GetNumAtoms() + pi2.GetNumAtoms() + donor2_pos)
                            # Chem.SanitizeMol(mol)
                            # AllChem.EmbedMolecule(mol)
                            smiles = Chem.MolToSmiles(mol)
                            mol = Chem.MolFromSmiles(smiles)
                            # img = Draw.MolToImage(mol, size=(300, 300))
                            # img.save(os.path.join(img_save_path, 'DAD_' + str(DADcount) + '.jpg'))
                            DADcount += 1
                            smis.append(smiles)
                            # print('Successful Molecule {}, {}, {}, {}, {}'.format(d1, p1, aa, p2, d2))
                        except:
                            print('Failed Molecule {}, {}, {}, {}, {}'.format(d1, p1, aa, p2, d2))

    smis = list(set(smis))
    id = []
    for i in range(len(smis)):
        id.append('DAD_' + str(i + 1))
    # try:
    #     df = pd.DataFrame(columns=['id', 'SMILES'])
    #     start = 0
    #     end = 1e6
    #     df['id'] = id[start:end]
    #     df['SMILES'] = smis[start:end]
    #     df.to_csv(os.path.join(save_path, 'DAD_structures_Donor_' + str(d1) + '_first_part.csv'), index=False)
    #     df = pd.DataFrame(columns=['id', 'SMILES'])
    #     df['id'] = id[end:]
    #     df['SMILES'] = smis[end:]
    #     df.to_csv(os.path.join(save_path, 'DAD_structures_Donor_' + str(d1) + '_second_part.csv'), index=False)
    # except:
    #     pass
    try:
        smis = np.array(smis)
        id = np.array(id)
        np.savez_compressed(os.path.join(save_path, 'DAD_structures_Donor_new_AA.npz'), id=id, SMILES=smis)
        print('DAD structure finished, and save to {}'.format(os.path.join(save_path, 'DAD_structures_Donor_new_AA.npz')))
    except:
        pass
    # smis = np.array(smis)
    # smis = list(set(smis))
    # id = []
    # for i in range(len(smis)):
    #     id.append('DAD_' + str(i + 1))
    # # save_path = '/public3/home/scg5611/XZR/2024-12-01'
    # try:
    #     file_count = int(len(smis) / 1e6) + 1
    #     for i in range(file_count):
    #         df = pd.DataFrame(columns=['id', 'SMILES'])
    #         start = int(i * 1e6)
    #         end = int((i + 1) * 1e6)
    #         df['id'] = id[start:end]
    #         df['SMILES'] = smis[start:end]
    #         df.to_csv(os.path.join(save_path, 'DAD_structures_' + str(i + 1) + '_1e6.csv'), index=False)
    # except:
    #     pass
    # try:
    #     smis = np.array(smis)
    #     id = np.array(id)
    #     np.savez_compressed(os.path.join(save_path, 'DAD_structures.npz'), id=id, smiles=smis)
    #     print('DAD structure finished, and save to {}'.format(os.path.join(save_path, 'DAD_structures.npz')))
    # except:
    #     pass

def combineD1AD2_ID_AA(D_mols, pi_mols, AA_mols, D_pos, pi_pos, AA_pos):
    length_all = 0
    img_save_path = PATHS.virtual_screen_aa_dad_dir
    ensure_dir(img_save_path)
    # AllChem.EmbedMolecule(mol)
    save_path = PATHS.virtual_screen_aa_dir
    ensure_dir(save_path)
    for aa in range(len(AA_mols)):
        DADcount = 0
        smis = []
        print('Current AA: ', aa)
        acceptor = AA_mols[aa]
        acceptor_pos1 = AA_pos[aa][0]
        acceptor_pos2 = AA_pos[aa][1]
        for d1 in range(len(D_mols)):
            donor1 = D_mols[d1]
            donor1_pos = D_pos[d1][0]
            for d2 in range(len(D_mols)):
                donor2 = D_mols[d2]
                donor2_pos = D_pos[d2][0]
                try:
                    # Chem.Kekulize(donor1)
                    # Chem.Kekulize(donor2)
                    # Chem.Kekulize(acceptor)
                    mol = connectmol(donor1, acceptor, donor1_pos, donor1.GetNumAtoms() + acceptor_pos1)
                    mol = connectmol(mol, donor2, donor1.GetNumAtoms() + acceptor_pos2, donor1.GetNumAtoms() + acceptor.GetNumAtoms() + donor2_pos)
                    # Chem.SanitizeMol(mol)
                    # AllChem.EmbedMolecule(mol)
                    smiles = Chem.MolToSmiles(mol)
                    mol = Chem.MolFromSmiles(smiles)
                    # img = Draw.MolToImage(mol, size=(300, 300))
                    # img.save(os.path.join(img_save_path, 'DAD_' + str(DADcount) + '.jpg'))
                    DADcount += 1
                    smis.append(smiles)
                    # print('Successful Molecule {}, {}'.format(d1, aa))
                except:
                    print('Failed Molecule {}, {}, {}'.format(d1, aa, d2))
                for p1 in range(len(pi_mols)):
                    pi1 = pi_mols[p1]
                    pi1_pos1 = pi_pos[p1][0]
                    pi1_pos2 = pi_pos[p1][1]
                    for p2 in range(len(pi_mols)):
                        pi2 = pi_mols[p2]
                        pi2_pos1 = pi_pos[p2][0]
                        pi2_pos2 = pi_pos[p2][1]
                        try:
                            # Chem.Kekulize(donor1)
                            # Chem.Kekulize(donor2)
                            # Chem.Kekulize(acceptor)
                            # Chem.Kekulize(pi1)
                            # Chem.Kekulize(pi2)
                            mol = connectmol(donor1, pi1, donor1_pos, donor1.GetNumAtoms() + pi1_pos1)
                            mol = connectmol(mol, acceptor, donor1.GetNumAtoms() + pi1_pos2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor_pos1)
                            mol = connectmol(mol, pi2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor_pos2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor.GetNumAtoms() + pi2_pos2)
                            mol = connectmol(mol, donor2, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor.GetNumAtoms() + pi2_pos1, donor1.GetNumAtoms() + pi1.GetNumAtoms() + acceptor.GetNumAtoms() + pi2.GetNumAtoms() + donor2_pos)
                            # Chem.SanitizeMol(mol)
                            # AllChem.EmbedMolecule(mol)
                            smiles = Chem.MolToSmiles(mol)
                            mol = Chem.MolFromSmiles(smiles)
                            # img = Draw.MolToImage(mol, size=(300, 300))
                            # img.save(os.path.join(img_save_path, 'DAD_' + str(DADcount) + '.jpg'))
                            DADcount += 1
                            smis.append(smiles)
                            # print('Successful Molecule {}, {}, {}, {}, {}'.format(d1, p1, aa, p2, d2))
                        except:
                            print('Failed Molecule {}, {}, {}, {}, {}'.format(d1, p1, aa, p2, d2))
        smis = list(set(smis))
        try:
            smis = np.array(smis)
            np.save(os.path.join(save_path, 'DAD_structures_AA_' + str(aa) + '.npy'), smis)
            print('DAD structure finished, and save to {}'.format(os.path.join(save_path, 'DAD_structures_AA_' + str(aa) + '.npy')))
        except:
            pass
        try:
            df = pd.DataFrame(columns=['id', 'SMILES'])
            id = []
            for i in range(len(smis)):
                id.append('DAD_AA_' + str(aa) + '_' + str(i + 1))
            df['id'] = id
            df['SMILES'] = smis
            df.to_csv(os.path.join(save_path, 'DAD_structures_AA_' + str(aa) + '.csv'), index=False)
            print('DAD structure finished, and save to {}'.format(os.path.join(save_path, 'DAD_structures_AA_' + str(aa) + '.csv')))
        except:
            pass
        length_all += DADcount
        print('Length of AA_{} is {}, total count is {}', str(aa), DADcount, length_all)

    # smis = np.array(smis)
    # smis = list(set(smis))
    # id = []
    # for i in range(len(smis)):
    #     id.append('DAD_' + str(i + 1))
    # # save_path = '/public3/home/scg5611/XZR/2024-12-01'
    # try:
    #     file_count = int(len(smis) / 1e6) + 1
    #     for i in range(file_count):
    #         df = pd.DataFrame(columns=['id', 'SMILES'])
    #         start = int(i * 1e6)
    #         end = int((i + 1) * 1e6)
    #         df['id'] = id[start:end]
    #         df['SMILES'] = smis[start:end]
    #         df.to_csv(os.path.join(save_path, 'DAD_structures_' + str(i + 1) + '_1e6.csv'), index=False)
    # except:
    #     pass
    # try:
    #     smis = np.array(smis)
    #     id = np.array(id)
    #     np.savez_compressed(os.path.join(save_path, 'DAD_structures.npz'), id=id, smiles=smis)
    #     print('DAD structure finished, and save to {}'.format(os.path.join(save_path, 'DAD_structures.npz')))
    # except:
    #     pass

if __name__ == '__main__':
    opts = DrawingOptions()
    opts.includeAtomNumbers = True
    path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(path, 'img')
    DA_save_path = os.path.join(path, 'results\\DA')
    DAD_save_path = os.path.join(path, 'results\\DAD')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(DAD_save_path):
        os.makedirs(DAD_save_path)
    if not os.path.exists(DA_save_path):
        os.makedirs(DA_save_path)
    D_smiles = [
        'C1(NC2=CC=CC=C2)=CC=CC=C1',
        'C1(NC2=CC=CC=C2)=CC=CC3=C1C=CC=C3',
        'CC1=CC=C(NC2=CC=C(C)C=C2)C=C1',
        'CC(C)(C)C1=CC=C(NC2=CC=C(C(C)(C)C)C=C2)C=C1',
        'COC1=CC=C(NC2=CC=C(OC)C=C2)C=C1',
        'CC1(C)C2=C(C=CC=C2)C3=CC=CC=C31',
        'CN1C2=C(C=CC=C2)C3=CC=CC=C31',
        'C1(N(C2=CC=CC=C2)C3=CC=CC=C3)=CC=CC=C1',
        'C1(N(C2=CC=CC=C2)C3=CC=CC=C3)=CC=CC4=C1C=CC=C4',
        'CC1=CC=C(N(C2=CC=C(C)C=C2)C3=CC=CC=C3)C=C1',
        'CC(C)(C)C1=CC=C(N(C2=CC=C(C(C)(C)C)C=C2)C3=CC=CC=C3)C=C1',
        'COC1=CC=C(N(C2=CC=C(OC)C=C2)C3=CC=CC=C3)C=C1',
        'CC1(C)C2=C(C=CC=C2)C3=CC=C(N(C4=CC=CC=C4)C5=CC=CC=C5)C=C31',
        'CC1(C)C2=C(C=CC=C2)C3=CC=C(N(C4=CC=C(OC)C=C4)C5=CC=C(OC)C=C5)C=C31',
        'CN1C2=C(C=CC=C2)C3=CC=C(N(C4=CC=CC=C4)C5=CC=CC=C5)C=C31',
        'CN1C2=C(C=CC=C2)C3=CC=C(N(C4=CC=C(OC)C=C4)C5=CC=C(OC)C=C5)C=C31',
        'C1(/C(C2=CC=CC=C2)=C\C3=CC=CC=C3)=CC=CC=C1',
        'C1(/C(C2=CC=CC=C2)=C(C3=CC=CC=C3)\C4=CC=CC=C4)=CC=CC=C1',
        'CC1=CC=C(/C(C2=CC=C(C)C=C2)=C(C3=CC=CC=C3)\C4=CC=CC=C4)C=C1',
        'COC1=CC=C(/C(C2=CC=C(OC)C=C2)=C(C3=CC=CC=C3)\C4=CC=CC=C4)C=C1',
        'CN(C)C1=CC=C(/C(C2=CC=C(N(C)C)C=C2)=C(C3=CC=CC=C3)\C4=CC=CC=C4)C=C1',
        'C1(/C(C2=CC=CC=C2)=C(C3=CC=C(NC4=CC=CC=C4)C=C3)\C5=CC=CC=C5)=CC=CC=C1',
        'COC1=CC=C(NC2=CC=C(/C(C3=CC=CC=C3)=C(C4=CC=CC=C4)/C5=CC=CC=C5)C=C2)C=C1',
        'COC1=CC=C(NC2=CC=C(/C(C3=CC=CC=C3)=C(C4=CC=C(OC)C=C4)/C5=CC=C(OC)C=C5)C=C2)C=C1',
        'C1(/C(C2=CC=CC=C2)=C(C3=CC=C(N(C4=CC=CC=C4)C5=CC=CC=C5)C=C3)\C6=CC=CC=C6)=CC=CC=C1',
        'COC1=CC=C(N(C2=CC=CC=C2)C3=CC=C(/C(C4=CC=CC=C4)=C(C5=CC=CC=C5)/C6=CC=CC=C6)C=C3)C=C1',
        'COC1=CC=C(N(C2=CC=CC=C2)C3=CC=C(/C(C4=CC=CC=C4)=C(C5=CC=C(OC)C=C5)/C6=CC=C(OC)C=C6)C=C3)C=C1',
        'C1(N(C2=CC=CC=C2)C3=CC=CC=C3)=CC=C(N(C4=CC=CC=C4)C5=CC=CC=C5)C=C1',
        'CC1=CC=C(N(C2=CC=CC=C2)C3=CC=C(N(C4=CC=CC=C4)C5=CC=CC=C5)C=C3)C=C1',
        'COC1=CC=C(N(C2=CC=CC=C2)C3=CC=C(N(C4=CC=CC=C4)C5=CC=CC=C5)C=C3)C=C1',
        'C1(N(C2=CC=CC=C2)C3=CC=CC=C3)=CC=C(N(C4=CC=C(N(C5=CC=CC=C5)C6=CC=CC=C6)C=C4)C7=CC=CC=C7)C=C1',
        'C1(C2=C(C3=CC=CC=C3)N=C(C4=CC=CC=C4)C(C5=CC=CC=C5)=N2)=CC=CC=C1',
        'CC1=CC=C(C2=C(C3=CC=C(C)C=C3)N=C(C4=CC=CC=C4)C(C5=CC=CC=C5)=N2)C=C1',
        'COC1=CC=C(C2=C(C3=CC=C(OC)C=C3)N=C(C4=CC=CC=C4)C(C5=CC=CC=C5)=N2)C=C1',
        'C1(C2=C(C3=CC=CC=C3)N=C(C4=CC=CC=C4)C(C5=CC=C(C6=CC=CS6)C=C5)=N2)=CC=CC=C1',
        'CC1=CC=C(C2=C(C3=CC=C(C)C=C3)N=C(C4=CC=CC=C4)C(C5=CC=C(C6=CC=CS6)C=C5)=N2)C=C1',
        'COC1=CC=C(C2=C(C3=CC=C(OC)C=C3)N=C(C4=CC=CC=C4)C(C5=CC=C(C6=CC=CS6)C=C5)=N2)C=C1',
    ]
    pi_smiles = [
        'C=C',
        'C1=CC=CC=C1',
        'C1=CC=CC1',
        'CN1C=CC=C1',
        'C1=CC=CO1',
        'C1=CC=CS1',
        'C1=CC=C[Se]1',
        'CC1=CSC=C1',
        'COC1=CSC=C1',
        'C12=C(SC=C2)C=CS1',
        'C12=C(SC3=C2SC=C3)C=CS1',
        'C1(C2=CC=CS2)=CC=CS1',
        'C12=CSC=C1OCCO2',
        'C12=CSC=C1OCCCO2',
        'C12=CSC=C1C=CC=C2',
        'CC1(C)C2=C(C=CC=C2)C3=CC=CC=C31',
        'CN1C2=C(C=CC=C2)C3=CC=CC=C31',
        'CC1=CC2=CSC=C2S1',
        'C1(C=CS2)=C2C(SC=C3)=C3C=C1',
        'C1(C=CS2)=C2C=C(C=CS3)C3=C1',
        'COC1=C2C(C=CS2)=C(OC)C3=C1C=CS3',
        'C1(C=CS2)=C2C(SC=C3)=C3OC1',
        'CC(C1=C2SC=C1)(C)C3=C2SC=C3',
        'C[Si](C1=C2SC=C1)(C)C3=C2SC=C3',
        'CN(C1=C2SC=C1)C3=C2SC=C3',
        'C12=CC=CC=C1N=S=N2',
        'C12=CC=NC=C1N=S=N2',
        'C12=CC=CC=C1N=[Se]=N2',
        'C12=CC=NC=C1N=[Se]=N2'
    ]
    A_smiles = [
        'C=C(C#N)C(O)=O',
        'C=C(C#N)C#N',
        'C=C(C1=CC=C([N+]([O-])=O)C=C1)C#N',
        'O=C(C(C1=O)=C)C2=C1C=CC=C2',
        'O=C(C1=C(/C2=C(C#N)\C#N)C=CC=C1)C2=C',
        'C=C(/C1=C(C#N)/C#N)/C(C2=C1C=CC=C2)=C(C#N)/C#N',
        'C=C(C#N)C1=CC=C(N)C=C1',
        'O=C(C(S1)=C)N(C)C1=S',
        'O=C(C(C1=O)=C)C2=C1C=C(C=CC=C3)C3=C2',
        'C12=NNN=C1C=CC=C2',
        'O=C(C1=C(/C2=C(C#N)\C#N)C=C3C(C=CC=C3)=C1)C2=C',
        'C[N+]1=CC=CC=C1',
        'C=CC1=[N+](CC)C(C=CC=C2)=C2S1',
        'CC1(C)C(C=C)=[N+](C)C2=C1C=CC=C2',
        'CC1(C)C(C=C)=[N+](C)C2=C1C=CC3=C2C=CC=C3',
        'C[N+]1=C(C=C)C2=CC=CC3=C2C1=CC=C3',
        'CCN(C1=CC=CC=C1/2)C(C=C)=CC2=C(C#N)/C#N',
        'C=CC(OC1=CC=CC=C1/2)=CC2=C(C#N)\C#N',
        'C=CC(N1CC)=C/C(C2=CC=CC=C12)=C(C#N)\C#N',
        'C=CC(OC1=CC=CC=C/21)=CC2=C(C#N)/C#N',
        'CC(O/1)(C)C(C=C)=C(C#N)C1=C(C#N)/C#N',
        'C1(C=CC=C2)=C2C(C=CC=C3)=C3C4=C1N=C(C5=CC=CC=C5)N4C6=CC=CC=C6',
        'C1(C=CC=C2)=C2C(C=CC=C3)=C3C4=C1N=CN4C5=CC=CC=C5',
        'C1(C=CC2=CC=C3)=C(C2=C3C=C4)C4=CC=C1'
    ]
    AA_smiles = [
        'C=C',
        'N#C/C=C/C#N',
        'CN1N=C2C=CC=CC2=N1',
        'CN1N=C2C=C(F)C(F)=CC2=N1',
        'C1=CC=CC=C1',
        'C1=CC=CO1',
        'C1=CC=CS1',
        'C12=CC=CC=C1C=CC=C2',
        'C12=CC=CC=C1N=S=N2',
        'C12=CC=NC=C1N=S=N2',
        'FC1=C(F)C=C2C(N=S=N2)=C1',
        'O=[N+](C1=C([N+]([O-])=O)C=C2C(N=S=N2)=C1)[O-]',
        'N#CC1=C(C#N)C=C2C(N=S=N2)=C1',
        'C12=C(C=CC=C2)C=C(N=S=N3)C3=C1',
        'C12=CSC=C1N=S=N2',
        'C1(C=C(C=CC=C2)C2=C3)=C3C=CC=C1',
        'C12=CC=CC=C1CC3=CC=CC=C3C2',
        'C12=CC=CC=C1N=[Se]=N2',
        'C12=CC=NC=C1N=[Se]=N2',
        'FC1=C(F)C=C2C(N=[Se]=N2)=C1',
        'O=[N+](C1=C([N+]([O-])=O)C=C2C(N=[Se]=N2)=C1)[O-]',
        'N#CC1=C(C#N)C=C2C(N=[Se]=N2)=C1',
        'C12=C(C=CC=C2)C=C(N=[Se]=N3)C3=C1',
        'C12=CSC=C1N=[Se]=N2',
        'C1(CCC2=CC=C3)=C(C2=C3CC4)C4=CC=C1',
        'O=C1C(C)C=C2C(C(C)C=C21)=O',
        'O=C1N(C)C=C2C(N(C)C=C21)=O',
        'O=C(/C1=C(C2=C(N3C)C=CS2)/C3=O)N(C)C4=C1SC=C4',
        'O=C1N(C)C2=C(C=CC=C2)/C1=C(C(C=CC=C3)=C3N4C)\C4=O',
        'O=C1N(C)C(C(N2C)=C3C(N(C)C=C3C2=O)=O)=C4C(N(C)C=C41)=O',
        'O=C1C(C2=CN1CCCCCC)=CN(CCCCCC)C2=O',
        'O=C1C2=C(C=CC=C2)C(C3=CC=CC=C31)=O',
        'N#CC1=NC=CN=C1C#N',
        'N#CC1=NC(C2=CC=CC=C2)=C(C3=CC=CC=C3)N=C1C#N',
        'N#CC1=NC(C2=C3C=CC=C2)=C(C4=CC=CC=C43)N=C1C#N',
        'N#CC1=NC(C2=C(C3=CC=C4)C4=CC=C2)=C3N=C1C#N',
        'C12=CC=C(N=S=N3)C3=C1C=CC4=C2N=S=N4',
        'C12=C(N=C(C=CC=C3)C3=N4)C4=C(C=CC=C5)C5=C1C=CC=C2',
        'C12=C(N=CC=N2)C=C(N=S=N3)C3=C1',
        'C12=C(N=C(C3=CC=CC=C3)C(C4=CC=CC=C4)=N2)C=C(N=S=N5)C5=C1',
        'C12=C(N=C(C3=CC=CC=C3C4=C5C=CC=C4)C5=N2)C=C(N=S=N6)C6=C1',
        'C12=C(N=C(C3=CC=CC4=C3C5=CC=C4)C5=N2)C=C(N=S=N6)C6=C1',
        'C12=C(C(C=CS3)=C3C4=C2C=CS4)C=C(N=S=N5)C5=C1',
        'C12=C(N=C(C(C=CS3)=C3C4=C5C=CS4)C5=N2)C=C(N=S=N6)C6=C1',
        'N#CC(C=C1)=CC=C1C2=NC3=CC=CC=C3N=C2C4=CC=C(C#N)C=C4',
        'CN1C(C2=C(C1=O)C=C3C(N=S=N3)=C2)=O',
        'CN1C(C2=CC3=NN(C)N=C3C=C2C1=O)=O',
        'CN1C(C2=CC(C(N(C)C3=O)=O)=C3C=C2C1=O)=O',
        'CN1N=C(C=C(N=S=N2)C2=C3)C3=N1',
        'C12=NSN=C1C=C(N=S=N3)C3=C2',
        'C12=NSN=C1C=C(N=[Se]=N3)C3=C2',
        'C12=N[Se]N=C1C=C(N=[Se]=N3)C3=C2',
        'CCN1C2=C(C=CC=C2)C3=C1C=CC=C3',
        'O=C1C2=C(C=CC=C2)C3=C1C=CC=C3',
        'N#C/C(C#N)=C1C2=C(C=CC=C2)C3=C/1C=CC=C3',
        'N#C/C(C#N)=C1C2=C(C=CC=N2)C3=C/1N=CC=C3',
        'CN1C2=CC=C(N=S=N3)C3=C2C4=C5N=S=NC5=CC=C41',
        'CN1C2=C3C(C(SC=C4)=C4N3C)=C(N=S=N5)C5=C2C6=C1C=CS6',
        'O=C(N(C)C1=O)C2=CC=C3C4=C2C1=CC=C4C(N(C)C3=O)=O',
        'O=C(N1C)C2=C3C4=C(C=C2)C5=C(C6=C7C=C5)C(CC=C6C(N(C)C7=O)=O)=C4C=CC3C1=O',
        'O=C(N1C)C2=C3C(C(C4=NC5=C6)=CC=C3C1=O)=C(C4=NC5=CC7=C6N=C(C8=CC=C(C9=C%10C=CC%11=C98)C(N(C%10=O)C)=O)C%11=N7)C=C2',
        'C[Si]1(C)C=C(C2=CC=CC=C2)C(C3=CC=CC=C3)=C1',
        'C[Si]1(C2=CC=CC=C2)C=C(C3=CC=CC=C3)C(C4=CC=CC=C4)=C1',
        'CCC(CCCC)COC1=C2C(C=CS2)=C(OCC(CCCC)CC)C3=C1C=CS3',
        'CCC(CCCC)COC1=C(C=C2)C(S2(=O)=O)=C(OCC(CCC)CC)C(C=C3)=C1S3(=O)=O',
        'C1(/C(C2=CC=CC=C2)=C(C3=CC=CC=C3)\C4=CC=CC=C4)=CC=CC=C1',
    ]
    D_count = int(len(D_smiles))
    pi_count = int(len(pi_smiles))
    A_count = int(len(A_smiles))
    AA_count = int(len(AA_smiles))
    print(D_count, pi_count, A_count, AA_count)
    D_mols = []
    pi_mols = []
    A_mols = []
    AA_mols = []
    D_pos = [
        [1],
        [1],
        [5],
        [8],
        [6],
        [7],
        [6],
        [11],
        [11],
        [16],
        [22],
        [18],
        [7],
        [7],
        [6],
        [6],
        [8],
        [18],
        [23],
        [25],
        [27],
        [13],
        [6],
        [6],
        [23],
        [10],
        [10],
        [27],
        [9],
        [10],
        [40],
        [6],
        [19],
        [21],
        [25],
        [30],
        [32],
    ]
    pi_pos = [
        [0,1],
        [0,3],
        [0,3],
        [2,5],
        [0,3],
        [0,3],
        [0,3],
        [2,4,],
        [3,5],
        [3,6],
        [6,9],
        [4, 8],
        [1,3],
        [1,3],
        [1,3],
        [7,12],
        [6,11],
        [4,6],
        [2,7],
        [2,8],
        [6,14],
        [2,7],
        [5,11],
        [5,11],
        [5,10],
        [1, 4],
        [1, 4],
        [1, 4],
        [1, 4],
    ]
    A_pos= [
        [0],
        [0],
        [0],
        [5],
        [15],
        [0],
        [0],
        [4],
        [5],
        [2],
        [19],
        [4],
        [0],
        [5],
        [5],
        [4],
        [11],
        [0],
        [0],
        [0],
        [6],
        [19],
        [15],
        [15]
    ]
    AA_pos = [
        [0, 1],
        [2, 3],
        [4, 7],
        [4, 9],
        [0, 3],
        [0, 3],
        [0, 3],
        [4, 9],
        [1, 4],
        [1, 4],
        [4, 10],
        [7, 13],
        [6, 12],
        [6, 12],
        [1, 3],
        [1, 8],
        [6, 13],
        [1, 4],
        [1, 4],
        [4, 10],
        [7, 13],
        [6, 12],
        [6, 12],
        [1, 3],
        [5, 14],
        [4, 9],
        [4, 9],
        [9, 18],
        [8, 15],
        [12, 21],
        [4, 12],
        [5, 12],
        [4, 5],
        [8, 15],
        [8, 15],
        [10, 12],
        [2, 10],
        [4, 7],
        [6, 12],
        [18, 24],
        [18, 24],
        [16, 22],
        [12, 18],
        [16, 22],
        [11, 14],
        [7, 13],
        [4, 11],
        [4, 13],
        [4, 10],
        [5, 11],
        [5, 11],
        [5, 11],
        [7, 12],
        [6, 11],
        [10, 15],
        [10, 15],
        [4, 17],
        [7, 21],
        [7, 14],
        [8, 17],
        [11, 21],
        [3, 18],
        [8, 23],
        [13, 28],
        [12, 28],
        [18, 23],
    ]
    for i in range(D_count):
        mol = Chem.MolFromSmiles(D_smiles[i])
        D_mols.append(mol)
    for i in range(pi_count):
        mol = Chem.MolFromSmiles(pi_smiles[i])
        pi_mols.append(mol)
    for i in range(A_count):
        mol = Chem.MolFromSmiles(A_smiles[i])
        A_mols.append(mol)
    for i in range(AA_count):
        mol = Chem.MolFromSmiles(AA_smiles[i])
        AA_mols.append(mol)
    img_save_path = PATHS.image_dir
    ensure_dir(img_save_path)
    show_mols(D_mols, D_pos, os.path.join(img_save_path, 'D-group.jpg'), 'Donor', D_count)
    show_mols(pi_mols, pi_pos, os.path.join(img_save_path, 'pi-group.jpg'), 'pi-bridge', pi_count)
    show_mols(A_mols, A_pos, os.path.join(img_save_path, 'A-group.jpg'), 'Acceptor', A_count)
    show_mols(AA_mols, AA_pos, os.path.join(img_save_path, 'AA-group.jpg'), 'AAceptor', AA_count)
    D_smis = pd.DataFrame(D_smiles)
    pi_smis = pd.DataFrame(pi_smiles)
    A_smis = pd.DataFrame(A_smiles)
    AA_smis = pd.DataFrame(AA_smiles)
    all_group = pd.concat(
        [D_smis, pd.DataFrame(D_pos), pi_smis, pd.DataFrame(pi_pos), A_smis, pd.DataFrame(A_pos), AA_smis,
         pd.DataFrame(AA_pos)], axis=1)
    all_group.columns = ['D_smis', 'D_pos', 'pi_smis', 'pi_pos1', 'pi_pos2', 'A_smis', 'A_pos', 'AA_smis', 'AA_pos1',
                         'AA_pos2']
    all_group.to_csv(os.path.join(path, 'Chemical Space Substructure.csv'), index=False)
    # combineDA(D_mols, pi_mols, A_mols, D_pos, pi_pos, A_pos)
    # combineDAD(D_mols, pi_mols, AA_mols, D_pos, pi_pos, AA_pos)
    # combineD1AD2(D_mols, pi_mols, AA_mols, D_pos, pi_pos, AA_pos)
    # combineD1AD2_ID_AA(D_mols, pi_mols, AA_mols, D_pos, pi_pos, AA_pos)
    print('end')
