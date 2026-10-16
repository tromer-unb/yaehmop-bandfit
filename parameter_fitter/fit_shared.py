#!/usr/bin/env python3
"""Shared-parameter YAeHMOP fitter for multiple band-structure images."""
from __future__ import annotations
from pathlib import Path
import argparse, copy, csv, json, sys, time

import numpy as np
import yaml
from ase.io import read
from scipy.optimize import differential_evolution, minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_from_image import build_specs, apply_x, physical_ok, parse_points
from image_target import load_band_image, trace_bands
from yaehmop_generic import (
    BandPath, normalized_path_coordinate, orbital_count, parse_band,
    parse_parameter_table, resolve_yaehmop_assets, run_bind,
    subset_for_elements, valence_electrons, write_parameter_file,
    write_periodic_input_generic,
)



def valid_basis(params):
    """Drop placeholder orbital channels with non-positive zeta1."""
    out = copy.deepcopy(params)
    for el in out:
        for orb in ("s", "p", "d", "f"):
            if orb in out[el] and float(out[el][orb]["zeta1"]) <= 0:
                del out[el][orb]
    return out


def load_config(path):
    cfg_path = Path(path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg, dict) or not isinstance(cfg.get("systems"), list):
        raise ValueError("YAML must contain a top-level 'systems' list")
    shared = cfg.get("shared") or {}
    if not isinstance(shared, dict):
        raise ValueError("'shared' must be a mapping")
    return cfg_path, shared, cfg["systems"]


def validate_schema(shared, systems):
    if len(systems) < 2:
        raise ValueError("Shared fitting requires at least two systems")
    names = []
    required = {"name", "structure", "reference", "emin", "emax",
                "symmetry_points", "labels"}
    for s in systems:
        if not isinstance(s, dict):
            raise ValueError("Each system entry must be a mapping")
        missing = required - set(s)
        if missing:
            raise ValueError(f"System {s.get('name','?')}: missing {sorted(missing)}")
        role = str(s.get("role", "train")).lower()
        if role not in {"train", "test"}:
            raise ValueError(f"System {s['name']}: role must be train or test")
        if float(s["emin"]) >= float(s["emax"]):
            raise ValueError(f"System {s['name']}: require emin < emax")
        if len(s["labels"]) != len(s["symmetry_points"]):
            raise ValueError(f"System {s['name']}: labels/points length mismatch")
        names.append(str(s["name"]))
    if len(names) != len(set(names)):
        raise ValueError("System names must be unique")
    if sum(str(s.get("role", "train")).lower() == "train" for s in systems) < 2:
        raise ValueError("At least two systems must have role: train")
    fw = shared.get("fit_window")
    if fw is not None and (len(fw) != 2 or float(fw[0]) >= float(fw[1])):
        raise ValueError("shared.fit_window must be [EMIN, EMAX] with EMIN < EMAX")


def resolve_relative(base, value):
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def subset_params(params, elements):
    return {el: params[el] for el in elements}

def prepare_system(entry, base, defaults, shared):
    name = str(entry["name"])
    cif = resolve_relative(base, entry["structure"])
    image = resolve_relative(base, entry["reference"])
    atoms = read(cif)
    symbols = atoms.get_chemical_symbols()
    elements = list(dict.fromkeys(s.capitalize() for s in symbols))
    params = subset_params(defaults, elements)
    norb = orbital_count(params, symbols)
    nelec = valence_electrons(params, symbols)
    nocc = nelec // 2
    if norb <= nocc:
        raise ValueError(f"{name}: invalid basis/electron count")

    point_dim = int(entry.get("point_dim", 2))
    flat = [v for p in entry["symmetry_points"] for v in p]
    points = parse_points(flat, point_dim)
    labels = tuple(str(v) for v in entry["labels"])
    nline = int(entry.get("points_per_line", shared.get("points_per_line", 30)))
    trace_nx = int(entry.get("trace_nx", shared.get("trace_nx", 401)))
    path = BandPath(labels, points, nline)

    target = load_band_image(image, float(entry["emin"]), float(entry["emax"]))
    xtrace, bands, counts, seedidx = trace_bands(target, norb, trace_nx)
    tv = float(bands[:, nocc - 1].max())
    tc = float(bands[:, nocc].min())
    target_gap = tc - tv
    target_mid = 0.5 * (tv + tc)
    pts = []
    for a, b in zip(points[:-1], points[1:]):
        for j in range(nline):
            pts.append(a + (b - a) * (j / nline))
    pts.append(points[-1])
    xcoord = normalized_path_coordinate(np.asarray(pts), atoms.cell)
    T = np.column_stack([np.interp(xcoord, xtrace, bands[:, j])
                         for j in range(norb)])
    centers = np.median(T, axis=0)
    W = np.ones_like(T) * (1 + 3 * np.exp(-(centers / 4.0) ** 2))[None, :]

    fw = entry.get("fit_window", shared.get("fit_window"))
    if fw is None:
        mask = np.ones_like(T, dtype=bool)
        fit_window = [float(entry["emin"]), float(entry["emax"])]
    else:
        lo, hi = map(float, fw)
        if not lo < hi:
            raise ValueError(f"{name}: fit_window requires EMIN < EMAX")
        mask = (T >= lo) & (T <= hi)
        fit_window = [lo, hi]
    if not np.any(mask):
        raise ValueError(f"{name}: no target points fall inside fit_window")

    return {"name": name, "role": str(entry.get("role", "train")).lower(),
            "weight": float(entry.get("weight", 1.0)), "cif": cif,
            "image": image, "atoms": atoms, "symbols": symbols,
            "elements": elements, "norb": norb, "nelec": nelec,
            "nocc": nocc, "path": path, "xcoord": xcoord, "T": T,
            "W": W, "mask": mask, "target_gap": target_gap,
            "target_mid": target_mid, "fit_window": fit_window,
            "trace_seed_x": float(xtrace[seedidx]), "trace_counts": counts}

def metrics_for_model(system, model):
    T, W, mask = system["T"], system["W"], system["mask"]
    nocc = system["nocc"]
    mv = float(model[:, nocc - 1].max())
    mc = float(model[:, nocc].min())
    model_gap = mc - mv
    shift = system["target_mid"] - 0.5 * (mv + mc)
    D = model + shift - T
    return {
        "fit_rmse_eV": float(np.sqrt(np.mean(D[mask] ** 2))),
        "fit_wrmse_eV": float(np.sqrt(np.sum(D[mask] ** 2 * W[mask]) / np.sum(W[mask]))),
        "full_rmse_eV": float(np.sqrt(np.mean(D ** 2))),
        "full_wrmse_eV": float(np.sqrt(np.sum(D ** 2 * W) / np.sum(W))),
        "target_gap_eV": float(system["target_gap"]),
        "model_gap_eV": model_gap,
        "gap_error_eV": model_gap - float(system["target_gap"]),
        "shift_eV": shift,
    }


def average_initial_results(x0, names, paths, base):
    if not paths:
        return np.asarray(x0, float)
    values = {n: [] for n in names}
    for item in paths:
        data = json.loads(resolve_relative(base, item).read_text())
        for n, v in data.get("best_x", {}).items():
            if n in values:
                values[n].append(float(v))
    x = np.asarray(x0, float).copy()
    for i, n in enumerate(names):
        if values[n]:
            x[i] = float(np.mean(values[n]))
    return x

def main():
    ap = argparse.ArgumentParser(description="Fit one shared YAeHMOP parameter set to multiple systems")
    ap.add_argument("config", help="YAML configuration file")
    ap.add_argument("--outdir", default="shared_fit_results")
    ap.add_argument("--tag", default="shared")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--validate-only", action="store_true",
                    help="validate YAML schema and exit without reading structures/images")
    args = ap.parse_args()

    cfg_path, shared, entries = load_config(args.config)
    validate_schema(shared, entries)
    if args.validate_only:
        print(json.dumps({"config": str(cfg_path), "systems": [s["name"] for s in entries],
                          "train": [s["name"] for s in entries if str(s.get("role","train")).lower()=="train"],
                          "test": [s["name"] for s in entries if str(s.get("role","train")).lower()=="test"]}, indent=2))
        return

    BIND, PARAM_TABLE = resolve_yaehmop_assets()
    base = cfg_path.parent
    all_elements = []
    for s in entries:
        atoms = read(resolve_relative(base, s["structure"]))
        for sym in atoms.get_chemical_symbols():
            el = sym.capitalize()
            if el not in all_elements:
                all_elements.append(el)
    table = parse_parameter_table(PARAM_TABLE)
    defaults = valid_basis(subset_for_elements(table, all_elements))
    systems = [prepare_system(s, base, defaults, shared) for s in entries]
    train = [s for s in systems if s["role"] == "train"]
    test = [s for s in systems if s["role"] == "test"]
    train_elements = {el for s in train for el in s["elements"]}
    unseen = sorted({el for s in test for el in s["elements"] if el not in train_elements})
    if unseen:
        raise ValueError(f"Test systems contain elements never seen in training: {unseen}")

    specs, x0, bounds, scales = build_specs(defaults, all_elements)
    names = [f"{a}_{b}_{c}".strip("_") for a, b, c in specs]
    bounds = list(bounds)
    for i, (_, _, kind) in enumerate(specs):
        if kind == "Hii":
            bounds[i] = (x0[i] - 5.0, x0[i] + 5.0)
    xcur = average_initial_results(x0, names, shared.get("initial_results", []), base)
    xcur = np.asarray([min(max(v, lo), hi) for v, (lo, hi) in zip(xcur, bounds)], float)

    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    seed = int(args.seed if args.seed is not None else shared.get("seed", 20260928))
    gap_weight = float(shared.get("gap_weight", 0.0))
    reg_weight = float(shared.get("reg_weight", 0.20 if shared.get("fit_window") else 0.04))
    history, best = [], [float("inf"), None, None]
    base_objective = str(shared.get("base_objective","legacy"))
    if base_objective not in ("legacy","balanced"):
        raise ValueError("shared.base_objective must be legacy or balanced")
    base_scales = {}   # defined from stock only, never held-out systems
    neval = 0
    t0 = time.time()
    def run_system(system, params, K, suffix):
        ps = subset_params(params, system["elements"])
        swork = work / system["name"]
        swork.mkdir(exist_ok=True)
        parm = swork / f"{suffix}.parms"
        inp = swork / f"{suffix}.bind"
        log = swork / f"{suffix}.log"
        write_parameter_file(parm, ps)
        write_periodic_input_generic(system["cif"], inp, system["path"], K, True, system["nelec"])
        run_bind(BIND, inp, parm, log)
        _, model = parse_band(str(inp) + ".band")
        if model.shape != system["T"].shape or not np.isfinite(model).all():
            raise ValueError(f"{system['name']}: model/target shape mismatch")
        return model

    def objective(x, record=True, stage=""):
        nonlocal neval, best
        neval += 1
        params, K = apply_x(defaults, specs, x)
        if not physical_ok(params):
            return 100.0
        per_system, weighted_sum, total_weight = {}, 0.0, 0.0
        try:
            for system in train:
                model = run_system(system, params, K, "shared")
                met = metrics_for_model(system, model)
                if base_objective=="balanced":
                    # Each component normalized by the same TRAINING system's
                    # initial stock-parameter residual. The frozen scales are
                    # set during the very first objective call at xcur, before
                    # any material sees the optimized parameter vector.
                    if system["name"] not in base_scales:
                        base_scales[system["name"]] = (
                            max(met["full_rmse_eV"],0.05),
                            max(met["fit_rmse_eV"],0.05))
                    sf,sw=base_scales[system["name"]]
                    loss_i=(.50*met["full_rmse_eV"]/sf
                            +.50*met["fit_rmse_eV"]/sw
                            +gap_weight*abs(met["gap_error_eV"]))
                else:
                    loss_i = (0.4 * met["fit_rmse_eV"] + 0.6 * met["fit_wrmse_eV"]
                              + gap_weight * abs(met["gap_error_eV"]))
                per_system[system["name"]] = {**met, "loss": float(loss_i)}
                weighted_sum += system["weight"] * loss_i
                total_weight += system["weight"]
            reg = float(np.sqrt(np.mean(((np.asarray(x) - x0) / scales) ** 2)))
            loss = float(weighted_sum / total_weight + reg_weight * reg)
        except Exception:
            return 100.0
        if record:
            row = {"eval": neval, "stage": stage, "loss": loss,
                   "regularization": reg, "K": float(K)}
            for name, met in per_system.items():
                row[f"{name}_fit_rmse_eV"] = met["fit_rmse_eV"]
                row[f"{name}_gap_error_eV"] = met["gap_error_eV"]
            history.append(row)
            if loss < best[0]:
                best = [loss, np.asarray(x, float).copy(), copy.deepcopy(per_system)]
                print("BEST", json.dumps(row), flush=True)
        return loss

    print(json.dumps({
        "train_systems": [s["name"] for s in train],
        "test_systems": [s["name"] for s in test],
        "shared_elements": all_elements,
        "variables": names,
        "gap_weight": gap_weight,
        "reg_weight": reg_weight,
        "base_objective": base_objective,
        "initial_results": shared.get("initial_results", []),
    }, indent=2), flush=True)

    base_loss = objective(xcur, True, "initial")
    stages = []

    def optimize_subset(indices, label, maxiter, popsize, localiter):
        nonlocal xcur
        ind = np.asarray(indices, int)
        sub_bounds = [bounds[i] for i in ind]
        def f(y):
            x = xcur.copy(); x[ind] = y
            return objective(x, True, label)
        de = differential_evolution(f, sub_bounds, maxiter=maxiter, popsize=popsize,
                                    seed=seed + len(stages), x0=xcur[ind], polish=False,
                                    workers=1, updating="immediate", tol=1e-5)
        loc = minimize(f, de.x, method="Nelder-Mead", bounds=sub_bounds,
                       options={"maxiter": localiter, "maxfev": max(160, localiter * 10),
                                "adaptive": True})
        xcur[ind] = loc.x if loc.fun < de.fun else de.x
        stages.append({"stage": label, "de_fun": float(de.fun),
                       "local_fun": float(loc.fun)})

    opt = shared.get("optimizer") or {}
    shape_idx = [i for i, s in enumerate(specs) if s[2] != "Hii"]
    hii_idx = [i for i, s in enumerate(specs) if s[2] == "Hii"]
    optimize_subset(shape_idx, "shape",
                    int(opt.get("shape_maxiter", 4)),
                    int(opt.get("popsize", 3)),
                    int(opt.get("shape_local_maxiter", 90)))
    optimize_subset(hii_idx, "hii",
                    int(opt.get("hii_maxiter", 4)),
                    int(opt.get("popsize", 3)),
                    int(opt.get("hii_local_maxiter", 110)))

    joint = minimize(lambda z: objective(z, True, "joint"), xcur,
                     method="Nelder-Mead", bounds=bounds,
                     options={"maxiter": int(opt.get("joint_maxiter", 180)),
                              "maxfev": int(opt.get("joint_maxfev", 700)),
                              "adaptive": True})
    if joint.fun < objective(xcur, False):
        xcur = joint.x.copy()
    final_loss = objective(xcur, True, "final")
    pbest, Kbest = apply_x(defaults, specs, xcur)

    final_metrics = {}
    for system in systems:
        model = run_system(system, pbest, Kbest, "final")
        final_metrics[system["name"]] = {
            "role": system["role"], "weight": system["weight"],
            "elements": system["elements"], "fit_window_eV": system["fit_window"],
            **metrics_for_model(system, model),
        }

    if history:
        fields = list(dict.fromkeys(k for row in history for k in row))
        with (out / f"{args.tag}_history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(history)

    native = out / f"param_{args.tag}_native.dat"
    write_parameter_file(native, pbest)
    friendly = out / f"param_{args.tag}.txt"
    friendly.write_text(
        f"# Shared YAeHMOP parameters fitted simultaneously to multiple systems\n"
        f"K = {Kbest:.14g}\nWEIGHTED_HIJ = true\n\n" + native.read_text()
    )

    result = {
        "tag": args.tag, "config": str(cfg_path),
        "train_systems": [s["name"] for s in train],
        "test_systems": [s["name"] for s in test],
        "shared_elements": all_elements, "variables": names,
        "K": float(Kbest), "base_loss": float(base_loss),
        "final_loss": float(final_loss), "gap_weight": gap_weight,
        "reg_weight": reg_weight, "best_x": {n: float(v) for n, v in zip(names, xcur)},
        "metrics": final_metrics, "stages": stages, "evaluations": neval,
        "elapsed_s": time.time() - t0,
    }
    (out / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FINAL", json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
