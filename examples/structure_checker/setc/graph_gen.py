"""
Module managing the generation of graph objects from crystallographic information
files (CIFs), including file reading, descriptor calculation, and graph structure.

"""

import os

import numpy as np

from typing import Union

import torch
from torch_geometric.data import Data

# from pymatgen.core import Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import IsayevNN, JmolNN

from logger import *
from elements import atomic_properties
from descriptors import generate_descriptors

# skipping extraneous pymatgen warnings
import warnings

warnings.filterwarnings("ignore")

logger = get_logger("classify")

os.makedirs("tmp_cif", exist_ok=True)


# construct graph representation from input .cif
def generate_graph(
    cif: str,
    save: Union[bool, str] = False,
    atomic: bool = False,
    mosaec: bool = False,
    localenv: bool = False,
    chemist: bool = False,
) -> Data:
    """
    Construct featurized structure graph from an input CIF file.

    Parameters:
        cif (str): path to the CIF file to be read.
        atomic (bool | str):  save path for graph files or 
                              False if saving is unwanted.
        atomic (bool):  whether to calculate atomic node features.
        mosaec (bool):  whether to calculate MOSAEC node features.
        localenv (bool):  whether to calculate local environment node features.
        chemist (bool):  whether to calculate chemist (Z+MOSAEC) node features.

    Returns:
        graph_data (Data): structure graph containing node features & edges.
    """

    # check that at least 1 node feature selected
    assert any(
        [atomic, mosaec, localenv, chemist]
    ), "Must select at least 1 node feature (atomic, mosaec, localenv, chemist)"
    # check if graph is available to read & skip repeat calculation
    if save:
        graph_name = f"{save}/" + os.path.basename(cif).replace(".cif", ".pt")
        if os.path.exists(graph_name):
            logger.debug(f"Found {graph_name} ... loading from previous file")
            return torch.load(graph_name)
    logger.info(f"preparing graph representation ... {cif}")
    # read in structure
    # cstruct = Structure.from_file(cif)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # set occupancy tolerance to avoid skipping erroneous atomic sites
        cstruct = (
            CifParser(cif, occupancy_tolerance=100)
            .parse_structures(primitive=False)
            .pop()
        )
    # output temporary cif from pymatgen
    # (needed to fix ordering issue b/w pymatgen & CSD (MOSAEC))
    cif = f"tmp_cif/{os.path.basename(cif)}"
    cstruct.to(cif, fmt="cif")
    #
    atom_symbols = [atom.specie.symbol for atom in cstruct]
    atom_labels = []
    label_counter = {element: 0 for element in set(atom_symbols)}
    for symbol in atom_symbols:
        label_counter[symbol] += 1
        atom_labels.append(f"{symbol}{label_counter[symbol]}")

    # calculate node features
    feature_list = []
    if atomic:
        logger.debug("Calculating ATOMIC node features...")
        f_atomic = np.array([atomic_properties[symbol] for symbol in atom_symbols])
        feature_list.append(f_atomic)
    if chemist:
        logger.debug("Calculating ATOMIC (Z only) node features...")
        f_atomic = np.array([[atomic_properties[symbol][0]] for symbol in atom_symbols])
        feature_list.append(f_atomic)
    if mosaec or chemist:
        from mosaec_descriptors import generate_mosaec
        logger.debug("Calculating MOSAEC node features...")
        os1, os2, os3 = generate_mosaec(cif)
        inner, outer, glob = [*os1.values()], [*os2.values()], [*os3.values()]
        f_mosaec = np.array([inner, outer, glob]).T.tolist()
        feature_list.append(f_mosaec)
    if localenv:
        logger.debug("Calculating LOCALENV features...")
        f_locenv = generate_descriptors(cstruct, atom_symbols, atom_labels)
        feature_list.append(f_locenv)
    node_features = np.hstack(feature_list)

    # calculate edges (Isayev NN algo.)
    try:
        nn_graph = IsayevNN(tol=0.5).get_bonded_structure(
            structure=cstruct, weights=True
        )
    except Exception as e:
        # swap to simpler distance-based NN if IsayevNN failed
        # (common for disconnected components)
        logger.debug(
            f"{os.path.basename(cif)} failed in IsayevNN ... changing to JmolNN"
        )
        nn_graph = JmolNN().get_bonded_structure(structure=cstruct, weights=True)

    cif_bond_info = (nn_graph.as_dict())["graphs"]["adjacency"]
    bonds_half = np.array(
        [(i, bond["id"]) for i, bonds in enumerate(cif_bond_info) for bond in bonds]
    )
    bonds_full = np.vstack([bonds_half, bonds_half[:, [1, 0]]]).T
    # calculate edge features (distance)
    bond_features = [
        [cstruct.get_distance(x, bonds_full[1][i])] for i, x in enumerate(bonds_full[0])
    ]

    # compile nodes & edges to graph
    graph_data = Data(
        x=torch.tensor(node_features, dtype=torch.float).contiguous(),
        edge_index=torch.tensor(bonds_full, dtype=torch.long).contiguous(),
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr=torch.tensor(bond_features, dtype=torch.float).contiguous(),
    )
    # save graph data if requested
    if save:
        torch.save(graph_data, graph_name)
    return graph_data
