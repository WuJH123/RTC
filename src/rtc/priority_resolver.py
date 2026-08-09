from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inp import discover_nodes


def _coordinates(inp_path: str | Path) -> pd.DataFrame:
    section = ""
    rows: list[tuple[str, float, float]] = []
    for raw in Path(inp_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        if section != "COORDINATES" or not line:
            continue
        tokens = line.split()
        if len(tokens) >= 3:
            rows.append((tokens[0], float(tokens[1]), float(tokens[2])))
    frame = pd.DataFrame(rows, columns=["node_id", "x", "y"])
    valid = set(discover_nodes(inp_path))
    return frame[frame["node_id"].astype(str).isin(valid)].drop_duplicates("node_id")


def resolve_priority_points(
    inp_path: str | Path,
    points: pd.DataFrame,
    *,
    max_distance: float | None = None,
) -> pd.DataFrame:
    """Map observed ponding-point coordinates to nearest SWMM nodes without guessing IDs."""

    required = {"priority_id", "x", "y"}
    missing = sorted(required - set(points.columns))
    if missing:
        raise ValueError(f"priority point table missing columns: {missing}")
    if len(points) != 8 or points["priority_id"].astype(str).nunique() != 8:
        raise ValueError("Wuhan formal priority mapping requires exactly eight unique observed points")
    nodes = _coordinates(inp_path)
    if nodes.empty:
        raise ValueError("INP contains no usable [COORDINATES]")
    xy = nodes[["x", "y"]].to_numpy(dtype=float)
    output: list[dict[str, object]] = []
    for _, row in points.iterrows():
        delta = xy - np.asarray([float(row["x"]), float(row["y"])])
        distance = np.sqrt(np.square(delta).sum(axis=1))
        idx = int(np.argmin(distance))
        d = float(distance[idx])
        if max_distance is not None and d > max_distance:
            raise ValueError(
                f"priority point {row['priority_id']} nearest-node distance {d} exceeds {max_distance}"
            )
        output.append({
            "priority_id": str(row["priority_id"]),
            "source_x": float(row["x"]),
            "source_y": float(row["y"]),
            "node_id": str(nodes.iloc[idx]["node_id"]),
            "node_x": float(nodes.iloc[idx]["x"]),
            "node_y": float(nodes.iloc[idx]["y"]),
            "distance_in_inp_coordinate_units": d,
        })
    result = pd.DataFrame(output)
    if result["node_id"].duplicated().any():
        dup = result.loc[result["node_id"].duplicated(False), ["priority_id", "node_id"]]
        raise ValueError(f"multiple observed points map to the same node; manual review required: {dup.to_dict('records')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Map eight observed ponding points to nodes using INP coordinates")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--points", required=True, help="CSV columns: priority_id,x,y")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-nodes", required=True)
    parser.add_argument("--max-distance", type=float)
    args = parser.parse_args()
    result = resolve_priority_points(
        args.inp, pd.read_csv(args.points), max_distance=args.max_distance
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False)
    Path(args.out_nodes).write_text("\n".join(result["node_id"].tolist()) + "\n", encoding="utf-8")
    print(json.dumps({
        "mapped": len(result),
        "max_distance": float(result["distance_in_inp_coordinate_units"].max()),
        "nodes_file": args.out_nodes,
        "mapping_csv": args.out_csv,
    }, indent=2))


if __name__ == "__main__":
    main()
