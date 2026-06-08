#!/usr/bin/env python3
import os
import sys
import glob
import argparse

import torch

import numpy as np
import pandas as pd

from time import perf_counter
from typing import Union

from torch_geometric.data import Data

from logger import *
from model import ErrGAT
from graph_gen import generate_graph

import concurrent.futures
from threading import Thread
from queue import Queue
import multiprocessing as mp

# collect cli arguments
code_desc = "Predict error types present in given crystal structure file using pretrained GAT models."
parser = argparse.ArgumentParser(description=code_desc)
parser.add_argument("cif", help="path to CIF file OR directory containing CIF files")
parser.add_argument(
    "--node_features",
    nargs="+",
    default=["atomic"],
    choices=["atomic", "mosaec", "localenv", "chemist"],
    help="select node feature combination.",
)
parser.add_argument(
    "--binary",
    action="store_true",
    help="predict error labels using binary relevance models.",
)
parser.add_argument(
    "--multi",
    action="store_true",
    help="predict error labels using multi-label models.",
)
parser.add_argument(
    "--error",
    action="store_true",
    help="predict presence of error rather than/in addition to error type.",
)
parser.add_argument(
    "--store_graphs",
    action="store_true",
    help="save graphs as .pt files",
)
parser.add_argument(
    "--log_level",
    default="INFO",
    choices=["INFO", "DEBUG", "WARNING"],
    help="output logging message level",
)

parser.add_argument(
    "--jobs",
    type=int,
    default=os.cpu_count(),
    help="number of parallel worker processes (default: CPU count)",
)

args = parser.parse_args()

# init timer
t_start = perf_counter()

# init logger
logger = logger_setup("classify", "setc.log", args.log_level)

# define device & code paths & GLOBALS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Device: {device}")

# search for data files
GAT_PATH = os.path.dirname(os.path.realpath(__file__))
logger.debug(f"Code File Path: {GAT_PATH}")

# save graphs
if args.store_graphs:
    dir_sffx = "_".join(sorted(args.node_features))
    SETC_GRAPH = f"tmp_pt_{dir_sffx}"
    os.makedirs(SETC_GRAPH, exist_ok=True)
    logger.debug(f"Temp. Graph File Path: {SETC_GRAPH}")
else:
    SETC_GRAPH = False

# feature combo to be included in output csvs
FEAT_ARGS = {x: True for x in args.node_features}
NODE_FEATS = "-".join(sorted(args.node_features))

# check for csd python api when mosaec is called
if ("mosaec" in args.node_features) or ("chemist" in args.node_features):
    try:
        import ccdc
    except Exception as e:
        logger.error(
            "CCDC IMPORT FAILED ... Exiting (CSD python API required for MOSAEC descriptors)"
        )
        sys.exit()

# check that only one feature given if chemist is called
if ("chemist" in args.node_features) and (len(args.node_features) != 1):
    logger.error(
        "Improper node feature list ... Chemist is mutually exclusive with all other categories"
    )
    logger.error(f"{str(args.node_features)} | exiting ... ")
    sys.exit()





STOP = object()

def writer_thread(queue: Queue, csv_file: str):
    header_written = os.path.exists(csv_file)
    while True:
        item = queue.get()
        if item is STOP:
            break
        df = pd.DataFrame([item])
        if header_written and os.path.exists(csv_file):
            df.to_csv(csv_file, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_file, mode="w", header=True, index=False)
            header_written = True



def _multi_worker(cif: str, node_features: list, node_feats_tag: str, store_graphs: bool) -> dict:
    feat_args_local = {x: True for x in node_features}
    local_graph_dir = False
    if store_graphs:
        dir_sffx = "_".join(sorted(node_features))
        local_graph_dir = f"tmp_pt_{dir_sffx}_{os.getpid()}"
        os.makedirs(local_graph_dir, exist_ok=True)

    model = load_model(3).to(device)
    model.eval()

    graph = generate_graph(cif, local_graph_dir, **feat_args_local)
    sclgraph = standard_scale(graph, node_features)
    with torch.inference_mode():
        y = model(sclgraph.to(device)).detach().cpu().numpy().flatten()

    logger.info(f"{cif}|        h: {y[0] > 0.5} ({y[0]:.3f})")
    logger.info(f"{cif}|   charge: {y[1] > 0.5} ({y[1]:.3f})")
    logger.info(f"{cif}| disorder: {y[2] > 0.5} ({y[2]:.3f})")

    return {
        "cif": os.path.basename(cif),
        "h_pred_raw": float(y[0]),
        "h_pred_threshold": float(y[0] > 0.5),
        "charge_pred_raw": float(y[1]),
        "charge_pred_threshold": float(y[1] > 0.5),
        "disorder_pred_raw": float(y[2]),
        "disorder_pred_threshold": float(y[2] > 0.5),
        "node_features": node_feats_tag,
    }




def optim_hypers(param_path: str) -> dict[str, Union[str, int, float]]:
    """
    Read model hyperparameters from optimized file.

        Parameters:
            param_path (str): path to the *.optim file containing
                              optimized hyperparameter

        Returns:
            param_dict (dict): Dictionary of hyperparameters to necessary
                               to load ErrGAT model
    """

    def convert_type(s: str) -> Union[bool, int, float, None]:
        """Convert string object from .optim file to correct data type."""
        try:
            new_s = float(s)
            if new_s % 1 == 0:
                return int(new_s)
            else:
                return new_s
        except Exception as e:
            if "True" in s:
                return True
            elif "False" in s:
                return False
            elif "None" in s:
                return None
            else:
                return s

    # read optimized hyperparameters
    with open(param_path, "r") as ppf:
        param_pairs = [p.split("=") for p in ppf.read().split("\n") if "=" in p]
    ignore = ["batch", "lr"]
    param_dict = {
        pp[0]: convert_type(pp[1]) for pp in param_pairs if pp[0] not in ignore
    }
    # param_dict = {pp[0]: convert_type(pp[1]) for pp in param_pairs if len(pp) == 2}
    logger.debug(f"Loaded Hyperparameter dict ... {str(param_dict)}")
    return param_dict


def load_model(no_targets: int, classifier: str = "multi") -> ErrGAT:
    """
    Load pretrained classification model from torch.state_dict().

        Parameters:
            no_targets (int): output [label] dimension i.e. error types
            classifier (str): model types to load i.e. h, charge, disorder, multi

        Returns:
            model (ErrGAT | torch.nn.Module): classification model
    """

    feats = args.node_features
    feat_len = {"atomic": 8, "mosaec": 3, "localenv": 168, "chemist": 4}
    model_base = "_".join(sorted(feats)) + f"_{classifier}"
    #
    logger.info(f"Loading GAT model ... {model_base}.pt")
    # # load state dict
    # chkpt = torch.load(
    #     os.path.join(GAT_PATH, f"models/{model_base}.pt"), map_location=device
    # )
    model_saved = os.path.join(GAT_PATH, f"models/{model_base}.pt")
    #

    # get model parameters
    model_params = {
        "in_chan": sum([feat_len[f] for f in feats]),
        "out_chan": None,
        "activation": "relu",
        "num_targets": no_targets,
    }
    opt_params = optim_hypers(
        os.path.join(GAT_PATH, f"models/hyperparameters/{model_base}.optim")
    )
    model_params.update(opt_params)
    #
    # model = ErrGAT(**model_params)
    # model.load_state_dict(chkpt["state_dict"])
    model = torch.load(model_saved, map_location=device)
    model.eval()
    return model.to(device)


def append_to_csv(row_data: dict[str, Union[str, float]], csv_file: str) -> None:
    """
    Adds the input results to a csv file summarizing error labels.

        Parameters:
            row_data (dict[str, str | float]): dict of predicted labels.
            csv_file (str): path to csv file.

        Returns:
            None
    """

    df = pd.DataFrame([row_data])
    logger.debug(f"Output {str(row_data)} to {csv_file}")
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file, mode="w", header=True, index=False)
    return 0


def standard_scale(graph: Data, feature_list: list) -> Data:
    """
    Standardize node features of an input graph.

        Parameters:
            graph (torch_geometric.data.Data): structure graph data
            feature_list (list): list containing types of node features
                                       used in the graph representation

        Returns:
            scaled_graph (torch_geometric.data.Data): structure graph
                                        data with scaled node features
    """

    scales = np.load(os.path.join(GAT_PATH, "models/stnd_scale.npz"))
    #
    if "chemist" in feature_list:
        chemist_mean = [scales[f"atomic_mean"][0], scales[f"mosaec_mean"]]
        means = np.concatenate(chemist_mean, axis=None)
        chemist_std = [scales[f"atomic_std"][0], scales[f"mosaec_std"]]
        stds = np.concatenate(chemist_std, axis=None)
    else:
        means = np.concatenate(
            [scales[f"{feat}_mean"] for feat in feature_list], axis=None
        )
        stds = np.concatenate(
            [scales[f"{feat}_std"] for feat in feature_list], axis=None
        )
    scaled_graph = graph
    scaled_graph.x = (graph.x - means) / stds
    return scaled_graph


def classify_multi(cif_files: list) -> None:
    node_feats_tag = "-".join(sorted(args.node_features))

    q: Queue = Queue(maxsize=2 * (args.jobs or 1))
    wt = Thread(target=writer_thread, args=(q, "multi_pred.csv"), daemon=True)
    wt.start()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    max_workers = max(1, args.jobs or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_multi_worker, cif, args.node_features, node_feats_tag, args.store_graphs)
            for cif in cif_files
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
                q.put(row)
            except Exception as e:
                logger.error(f"Failed on a CIF task: {e}", exc_info=True)

    q.put(STOP)
    wt.join()

# def classify_multi(cif_files: list) -> None:
    # """
    # Perform multi-label classification on input crystal structure.

    #     Parameters:
    #        cif_files (list): list of paths to crystallographic information
    #                           (cif) files to be analyzed

    #     Returns:
    #         None

    #     Outputs:
    #         predicted error labels (multi-label): [h charge disorder]

    # """

    # # load model
    # model = load_model(3).to(device)
    # for cif in cif_files:
    #     # create graph
    #     graph = generate_graph(cif, SETC_GRAPH, **FEAT_ARGS)
    #     # standard scale node features
    #     sclgraph = standard_scale(graph, args.node_features)
    #     with torch.inference_mode():
    #         model_Y = model(sclgraph.to(device)).detach().cpu().numpy().flatten()
    #         logger.info(f"{cif}|        h: {model_Y[0] > 0.5} ({model_Y[0]:.3f})")
    #         logger.info(f"{cif}|   charge: {model_Y[1] > 0.5} ({model_Y[1]:.3f})")
    #         logger.info(f"{cif}| disorder: {model_Y[2] > 0.5} ({model_Y[2]:.3f})")
    #         data = {
    #             "cif": os.path.basename(cif),
    #             f"h_pred_raw": float(model_Y[0]),
    #             f"h_pred_threshold": float(model_Y[0] > 0.5),
    #             f"charge_pred_raw": float(model_Y[1]),
    #             f"charge_pred_threshold": float(model_Y[1] > 0.5),
    #             f"disorder_pred_raw": float(model_Y[2]),
    #             f"disorder_pred_threshold": float(model_Y[2] > 0.5),
    #             f"node_features": NODE_FEATS,
    #         }
    #         append_to_csv(data, f"multi_pred.csv")


def classify_binary(cif_files: list) -> None:
    """
    Perform binary classification on input crystal structure.

        Parameters:
           cif_files (list): list of paths to crystallographic information
                              (cif) files to be analyzed

        Returns:
            None

        Outputs:
            predicted error labels (binary): h, charge, disorder

    """

    # load models into dict
    errs = ["h", "charge", "disorder"]
    err_models = {err: load_model(1, classifier=err) for err in errs}
    for cif in cif_files:
        # create graph
        graph = generate_graph(cif, SETC_GRAPH, **FEAT_ARGS)
        # standard scale node features
        sclgraph = standard_scale(graph, args.node_features)
        for err in errs:
            model = err_models[err]
            # model.eval()
            with torch.inference_mode():
                model_Y = model(sclgraph.to(device)).detach().cpu().numpy().flatten()
                logger.info(f"{cif}| {err} : {model_Y[0] > 0.5} ({model_Y[0]:.3f})")
                data = {
                    "cif": os.path.basename(cif),
                    f"{err}_pred_raw": float(model_Y[0]),
                    f"{err}_pred_threshold": float(model_Y[0] > 0.5),
                    f"node_features": NODE_FEATS,
                }
                append_to_csv(data, f"{err}_pred.csv")


def classify_any_error(cif_files: list) -> None:
    """
    Perform error classification on input crystal structure.

        Parameters:
            cif_files (list): list of paths to crystallographic information
                              (cif) files to be analyzed

        Returns:
            None

        Outputs:
            predicted error label (presence/absence): error

    """

    # load model
    model = load_model(1, classifier="error").to(device)
    for cif in cif_files:
        # create graph
        graph = generate_graph(cif, SETC_GRAPH, **FEAT_ARGS)
        # standard scale node features
        sclgraph = standard_scale(graph, args.node_features)
        with torch.inference_mode():
            model_Y = model(sclgraph.to(device)).detach().cpu().numpy().flatten()
            logger.info(
                f"{cif}| error (presence) : {model_Y[0] > 0.5} ({model_Y[0]:.3f})"
            )
            data = {
                "cif": os.path.basename(cif),
                "error_pred_raw": float(model_Y[0]),
                "error_pred_threshold": float(model_Y[0] > 0.5),
                f"node_features": NODE_FEATS,
            }
            append_to_csv(data, "error_pred.csv")


def main(args):
    # handle single-file/directory of files input options
    if os.path.isfile(args.cif):
        cifs = [args.cif]
    else:
        cifs = glob.glob(f"{args.cif}/*.cif")
    logger.info(f"Number of CIF files found == {len(cifs)}")

    # binary relevance h, charge, disorder classification
    if args.binary:
        logger.info(
            " ##########  Binary Relevance Error Type Classification ########## "
        )
        classify_binary(cifs)
    # multi-label  [h charge disorder] classification
    if args.multi:
        logger.info(
            " ##########    Multi-Label Error Type Classification    ########## "
        )
        classify_multi(cifs)
    # error presence/absence classification
    if args.error:
        logger.info(
            " ##########    Error Presence/Absence Classification    ########## "
        )
        classify_any_error(cifs)
    
    # clean-up
    t_end = perf_counter()
    logger.info(f"Finished successfully | Elapsed Time: {t_end - t_start}")
    return 0


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main(args)
