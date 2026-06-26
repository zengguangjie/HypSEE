#!/usr/bin/env python3
"""Verify graph_id alignment between processed dataset and precomputed hypergraphs."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.io import fs

from datasets.tu_dataset import TUDatasetExt, get_dataset, shuffle_dataset
from datasets.feature_expansion import FeatureExpander


def load_processed_direct(root, name, processed_filename):
    """Load a specific processed .pt without triggering re-process."""
    path = os.path.join(root, name, "processed", processed_filename)
    if not os.path.isfile(path):
        return None, path
    d = TUDatasetExt(
        root, name,
        pre_transform=FeatureExpander().transform,
        use_node_attr=True,
        processed_filename=processed_filename,
    )
    return d, path


def graph_fingerprint(data):
    ei = data.edge_index
  # canonical undirected-ish fingerprint
    src = ei[0].tolist()
    dst = ei[1].tolist()
    edges = tuple(sorted(zip(src, dst)))
    gid = data.graph_id
    if hasattr(gid, "item"):
        gid = int(gid.view(-1)[0].item())
    else:
        gid = int(gid)
    return gid, int(data.num_nodes), int(data.num_edges), edges


def check_dataset(name, root, processed_filename, hypergraph_lengths=(4, 6), mode="RW"):
    print(f"\n{'='*72}")
    print(f"Dataset file: {processed_filename}")
    dataset, path = load_processed_direct(root, name, processed_filename)
    if dataset is None:
        print(f"  MISSING: {path}")
        return None

    n = len(dataset)
    print(f"  path: {path}")
    print(f"  num_graphs: {n}")

    fps = []
    missing_gid = 0
    for i in range(n):
        data = dataset.get(i)
        if not hasattr(data, "graph_id") or data.graph_id is None:
            missing_gid += 1
            continue
        fps.append((i, graph_fingerprint(data)))

    print(f"  graphs missing graph_id: {missing_gid}")
    gids = [fp[1][0] for fp in fps]
    unique_gids = sorted(set(gids))
    print(f"  graph_id range: {min(gids) if gids else 'N/A'} .. {max(gids) if gids else 'N/A'}")
    print(f"  unique graph_ids: {len(unique_gids)} / {n}")
    if len(unique_gids) != n:
        print("  WARNING: duplicate graph_ids in dataset!")

    idx_eq_gid = sum(1 for i, (gid, *_ ) in fps if i == gid)
    print(f"  index == graph_id for {idx_eq_gid}/{n} graphs")

    hg_dir = os.path.join(root, name, mode)
    for length in hypergraph_lengths:
        hg_path = os.path.join(hg_dir, f"len_{length}.pt")
        if not os.path.isfile(hg_path):
            print(f"  hypergraph MISSING: {hg_path}")
            continue
        hg_dict = fs.torch_load(hg_path)
        hg_keys = sorted(hg_dict.keys())
        print(f"\n  hypergraph len_{length}.pt: {len(hg_keys)} entries, keys {hg_keys[0]}..{hg_keys[-1]}")

        missing_keys = [gid for gid in gids if gid not in hg_dict]
        orphan_keys = [k for k in hg_keys if k not in set(gids)]
        print(f"    dataset graph_ids not in hypergraph: {len(missing_keys)}")
        if missing_keys[:5]:
            print(f"      examples: {missing_keys[:5]}")
        print(f"    hypergraph keys not in dataset: {len(orphan_keys)}")

        node_mismatch = []
        for i, (gid, num_nodes, num_edges, _) in fps:
            if gid not in hg_dict:
                continue
            hg_nodes = hg_dict[gid].size()[0]
            if hg_nodes != num_nodes:
                node_mismatch.append((i, gid, num_nodes, hg_nodes))

        print(f"    num_nodes mismatch (dataset vs hypergraph): {len(node_mismatch)}")
        if node_mismatch[:5]:
            for row in node_mismatch[:5]:
                print(f"      idx={row[0]} graph_id={row[1]} dataset_nodes={row[2]} hg_nodes={row[3]}")

    return fps


def cross_compare_fps(fps_a, label_a, fps_b, label_b, sample=5):
    if not fps_a or not fps_b:
        return
    print(f"\n{'='*72}")
    print(f"Cross-compare graph structure by graph_id: {label_a} vs {label_b}")
    map_a = {fp[1][0]: fp[1] for fp in fps_a}
    map_b = {fp[1][0]: fp[1] for fp in fps_b}
    common = sorted(set(map_a) & set(map_b))
    struct_mismatch = []
    for gid in common:
        ga = map_a[gid]
        gb = map_b[gid]
        if ga[1:] != gb[1:]:  # nodes, edges, edge list
            struct_mismatch.append((gid, ga, gb))
    print(f"  common graph_ids: {len(common)}")
    print(f"  structure mismatch at same graph_id: {len(struct_mismatch)}")
    for gid, ga, gb in struct_mismatch[:sample]:
        print(f"    graph_id={gid}: {label_a} nodes/edges={ga[1]}/{ga[2]} | {label_b} nodes/edges={gb[1]}/{gb[2]}")


def check_shuffled(dataset, hypergraph_lengths=(4, 6), mode="RW", seed=0):
    print(f"\n{'='*72}")
    print("After shuffle_dataset (graph_id should travel with each graph)")
    shuffled = shuffle_dataset(dataset)
    hg_path = os.path.join(dataset.root, dataset.name, mode, f"len_{hypergraph_lengths[0]}.pt")
    hg_dict = fs.torch_load(hg_path)
    mismatch = 0
    for i in range(min(20, len(shuffled))):
        data = shuffled.get(i)
        gid = int(data.graph_id.view(-1)[0].item()) if hasattr(data.graph_id, "item") else int(data.graph_id)
        if gid not in hg_dict:
            mismatch += 1
            continue
        if hg_dict[gid].size()[0] != data.num_nodes:
            mismatch += 1
    print(f"  first-20 shuffled samples with hg node mismatch or missing key: {mismatch}/20")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "PROTEINS"
    root = sys.argv[2] if len(sys.argv) > 2 else "data"
    root = os.path.normpath(root)

    files = ["data.pt", "data_deg.pt", "data_.pt", "data_deg+odeg100.pt"]
    all_fps = {}
    for fn in files:
        fps = check_dataset(name, root, fn)
        if fps is not None:
            all_fps[fn] = fps

    if "data.pt" in all_fps and "data_.pt" in all_fps:
        cross_compare_fps(all_fps["data.pt"], "data.pt", all_fps["data_.pt"], "data_.pt")
    if "data.pt" in all_fps and "data_deg.pt" in all_fps:
        cross_compare_fps(all_fps["data.pt"], "data.pt", all_fps["data_deg.pt"], "data_deg.pt")

    # Reference hypergraphs were likely built with default feat_str=deg
    ds, _ = load_processed_direct(root, name, "data_deg.pt")
    if ds is None:
        ds, _ = load_processed_direct(root, name, "data.pt")
    if ds is not None:
        check_shuffled(ds)


if __name__ == "__main__":
    main()
