from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .closed_loop import CausalObservation, ControllerAction, run_authoritative_closed_loop


class ContinuousExplorationController:
    """Causal development-only bounded random walk with no fixed active actuator subset."""

    def __init__(
        self,
        *,
        seed: int,
        perturbation_std: float = 0.12,
        change_probability: float = 0.35,
        max_delta_per_update: float = 0.20,
    ):
        if perturbation_std <= 0 or max_delta_per_update <= 0:
            raise ValueError("exploration perturbation and max delta must be positive")
        if not 0.0 < change_probability <= 1.0:
            raise ValueError("change_probability must lie in (0,1]")
        self.rng = np.random.default_rng(seed)
        self.perturbation_std = float(perturbation_std)
        self.change_probability = float(change_probability)
        self.max_delta = float(max_delta_per_update)
        self.current: np.ndarray | None = None

    def __call__(self, obs: CausalObservation) -> ControllerAction:
        readback = np.asarray(obs.actuator_current_setting, dtype=float)
        if self.current is None:
            self.current = readback.copy()
        eligible = self.rng.random(len(readback)) < self.change_probability
        # All facilities are independently eligible at every update; the stochastic mask is
        # data coverage only, never a frozen online active set.
        delta = self.rng.normal(0.0, self.perturbation_std, size=len(readback))
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        proposed = np.clip(self.current + eligible * delta, 0.0, 1.0)
        self.current = proposed
        return ControllerAction(
            settings=dict(zip(obs.actuator_ids, proposed, strict=True)),
            source="D1_CONTINUOUS_EXPLORATION",
            diagnostics={
                "all_actuators_eligible": True,
                "changed_actuators": int(np.count_nonzero(np.abs(proposed - readback) > 1e-9)),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate development-only controlled D1 full-event trajectory")
    parser.add_argument("--inp", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--control-update-seconds", type=int, required=True)
    parser.add_argument("--control-start-minutes", type=int, default=0)
    parser.add_argument("--perturbation-std", type=float, default=0.12)
    parser.add_argument("--change-probability", type=float, default=0.35)
    parser.add_argument("--max-delta", type=float, default=0.20)
    args = parser.parse_args()
    sensors = tuple(
        line.strip()
        for line in Path(args.sensors).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if args.control_update_seconds % args.model_step_seconds:
        raise ValueError("control update must be an integer multiple of model step")
    controller = ContinuousExplorationController(
        seed=args.seed,
        perturbation_std=args.perturbation_std,
        change_probability=args.change_probability,
        max_delta_per_update=args.max_delta,
    )
    result = run_authoritative_closed_loop(
        inp_path=args.inp,
        output_dir=args.out_dir,
        run_id=args.run_id,
        sensor_nodes=sensors,
        controller=controller,
        control_start_minutes=args.control_start_minutes,
        control_update_seconds=args.control_update_seconds,
        observation_update_seconds=args.model_step_seconds,
        record_stride_seconds=args.model_step_seconds,
        exact_global_peak=False,
    )
    sidecar = Path(args.out_dir) / f"{args.run_id}.d1_exploration.json"
    payload = {
        "contract": "D1_DEVELOPMENT_CONTINUOUS_EXPLORATION_V1",
        "scientific_split_allowed": "development/train",
        "all_actuators_eligible": True,
        "fixed_active_subset": False,
        "binary_mask": False,
        "seed": args.seed,
        "perturbation_std": args.perturbation_std,
        "change_probability": args.change_probability,
        "max_delta_per_update": args.max_delta,
        "main_metadata_path": result.metadata_path,
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
