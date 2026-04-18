import os
from typing import Optional

import pandas as pd
import scipy
import scipy.sparse

from btorch import connectome
from btorch.connectome import augment, connection
from btorch.utils import hdf5_utils, pandas_utils

def save_conn_mats(conn_mats_h5, conn_mats: dict):
    data = {}
    for key, matrix in conn_mats.items():
        arrays_dict = {}
        if matrix.format in ("csc", "csr", "bsr"):
            arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
        elif matrix.format == "dia":
            arrays_dict.update(offsets=matrix.offsets)
        elif matrix.format == "coo":
            arrays_dict.update(row=matrix.row, col=matrix.col)
        arrays_dict.update(
            format=matrix.format.encode("ascii"), shape=matrix.shape, data=matrix.data
        )
        if isinstance(matrix, scipy.sparse.sparray):
            arrays_dict.update(_is_array=True)
        data[key] = arrays_dict

    hdf5_utils.save_dict_to_hdf5(conn_mats_h5, data)


def load_conn_mats(file):
    loaded = hdf5_utils.load_dict_from_hdf5(file)

    def make_sparray(mat):
        sparse_format = mat.get("format")
        if sparse_format is None:
            raise ValueError(
                f"The file {file} does not contain " f"a sparse array or matrix."
            )

        if not isinstance(sparse_format, str):
            # Play safe with Python 2 vs 3 backward compatibility;
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            sparse_format = sparse_format.decode("ascii")

        if mat.get("_is_array"):
            sparse_type = sparse_format + "_array"
        else:
            sparse_type = sparse_format + "_matrix"

        try:
            cls = getattr(scipy.sparse, f"{sparse_type}")
        except AttributeError as e:
            raise ValueError(f'Unknown format "{sparse_type}"') from e

        if sparse_format in ("csc", "csr", "bsr"):
            return cls((mat["data"], mat["indices"], mat["indptr"]), shape=mat["shape"])
        elif sparse_format == "dia":
            return cls((mat["data"], mat["offsets"]), shape=mat["shape"])
        elif sparse_format == "coo":
            return cls((mat["data"], (mat["row"], mat["col"])), shape=mat["shape"])
        else:
            raise NotImplementedError(
                f"Load is not implemented for "
                f"sparse matrix of format {sparse_format}."
            )

    for key, mat in loaded.items():
        loaded[key] = make_sparray(mat)
    return loaded

def load_and_preprocess_mice(data_dir, processed_dir: Optional[str] = None):
    if processed_dir is None:
        processed_dir = data_dir
    conn_mats_h5 = f"{processed_dir}/mice_conn_mats_processed.h5"
    neuron_class_parquet = f"{processed_dir}/mice_neurons_processed.parquet"
    connections_parquet = f"{processed_dir}/mice_connections_processed.parquet"

    if (
        os.path.exists(conn_mats_h5)
        and os.path.exists(neuron_class_parquet)
        and os.path.exists(connections_parquet)
        and False #TODO: remove this
    ):
        neurons = pd.read_parquet(neuron_class_parquet)
        connections = pd.read_parquet(connections_parquet)
        conn_mats = load_conn_mats(conn_mats_h5)
    else:
        neurons = pd.read_csv(f"{data_dir}/neurons.csv.gz")
        connections = pd.read_csv(f"{data_dir}/connections.csv.gz")
        #打印connections的keys
        #print(f"connections columns: {connections.columns}")
        #print(f"neurons columns: {neurons.columns}")
        #breakpoint()
        conn_mats = pandas_utils.groupby_to_dict(connections, by="EI")
        # print(f"conn_mats keys: {conn_mats.keys()}")
        # 重要修改：使用make_hetersynapse_conn函数生成conn_mats，而不是make_sparse_mat函数
        # conn_mats = {
        #     ei: connection.make_sparse_mat(
        #         conn, shape=(neurons.shape[0], neurons.shape[0])
        #     )
        #     for ei, conn in conn_mats.items()
        # }
        conn_mats = connection.make_hetersynapse_conn(neurons, connections)
        #print(f"conn_mats: {conn_mats}")
        #print("--------------------------------")
        #print(f"connections: {connections}")
        #print("--------------------------------")
        #print(f"neurons: {neurons}")
        #print("--------------------------------")
        neurons.to_parquet(neuron_class_parquet, index=False)
        connections.to_parquet(connections_parquet, index=False)
        if not os.path.exists(conn_mats_h5):
            try:
                #breakpoint()
                pass
                #save_conn_mats(conn_mats_h5, conn_mats)
            except BlockingIOError:
                pass  # 别人正在写，我就不写了，反正内容一样
        #save_conn_mats(conn_mats_h5, conn_mats)
    return neurons, conn_mats, connections


def load_and_preprocess(data_dir, processed_dir: Optional[str] = None):
    if processed_dir is None:
        processed_dir = data_dir
    conn_mats_h5 = f"{processed_dir}/flywire_conn_mats_processed.h5"
    neuron_class_parquet = f"{processed_dir}/flywire_neurons_processed.parquet"
    connections_parquet = f"{processed_dir}/flywire_connections_processed.parquet"

    if (
        os.path.exists(conn_mats_h5)
        and os.path.exists(neuron_class_parquet)
        and os.path.exists(connections_parquet)
    ):
        neurons = pd.read_parquet(neuron_class_parquet)
        connections = pd.read_parquet(connections_parquet)
        conn_mats = load_conn_mats(conn_mats_h5)
    else:
        raw_data = connectome.load_raw_data(data_dir=data_dir)
        neurons, connections, coordinates = connectome.prepare_data(**raw_data)
        neurons = augment.drop_no_conn(neurons, connections)
        connectome.add_simple_id_from_root_id(neurons, connections)
        connectome.add_ei_from_nt_type(connections)
        conn_mats = pandas_utils.groupby_to_dict(connections, by="EI")
        conn_mats = {
            ei: connectome.make_sparse_mat(
                conn, shape=(neurons.shape[0], neurons.shape[0])
            )
            for ei, conn in conn_mats.items()
        }
        neurons = neurons.merge(coordinates, how="left", on="root_id")

        # Save neurons, connections, and conn_mats separately
        neurons.to_parquet(neuron_class_parquet, index=False)
        connections.to_parquet(connections_parquet, index=False)
        save_conn_mats(conn_mats_h5, conn_mats)

    # print(f"neurons")
    # print(neurons)
    # print("--------------------------------")
    # print("conn_mats")
    # print(conn_mats)
    # print("--------------------------------")
    # print("connections")
    # print(connections)
    # print("--------------------------------")
    return neurons, conn_mats, connections
