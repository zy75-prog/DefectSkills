#!/usr/bin/env python3
"""Read VASP static results and export pydefect-corrected formation energies.

No material-specific energies or chemical potentials are embedded. Source files
are read-only; generated results go to defect/postprocessing unless --output is set.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import yaml
from pymatgen.core import Structure
from pymatgen.io.vasp import Outcar, Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

SCHEMA = "agent-defects/1"
ENERGY_RE = re.compile(r"energy\(sigma->0\)\s*=\s*([-+\d.EeDd]+)")
MATCH_KEYS = ("ENCUT", "GGA", "LHFCALC", "AEXX", "HFSCREEN", "METAGGA", "LSORBIT", "LDAU")


def dump(path, data):
    """Write JSON atomically, refusing NaN or infinite numeric results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def versions():
    result = {}
    for package in ("pymatgen", "pydefect", "py-sc-fermi"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def resolve_one(base, filename="vasprun.xml", supplied=None):
    if supplied:
        p = Path(supplied).expanduser().resolve()
        return p.parent if p.is_file() else p
    paths = sorted(p.parent for p in Path(base).rglob(filename)) if Path(base).exists() else []
    if len(paths) != 1:
        raise ValueError(f"Expected one {filename} under {base}; found {len(paths)}. "
                         "Specify the calculation path explicitly if there are multiple runs.")
    return paths[0]


def sigma_zero_energy(outcar):
    matches = ENERGY_RE.findall(Path(outcar).read_text(errors="replace"))
    if not matches:
        raise ValueError(f"No energy(sigma->0) found in {outcar}")
    energy = float(matches[-1].replace("D", "E").replace("d", "e"))
    if not math.isfinite(energy):
        raise ValueError(f"Invalid total energy in {outcar}")
    return energy


def load_run(directory, static=True, dos=False):
    directory = Path(directory).resolve()
    for name in ("OUTCAR", "vasprun.xml", "POSCAR"):
        if not (directory / name).is_file():
            raise ValueError(f"Missing {directory / name}")
    vr = Vasprun(directory / "vasprun.xml", parse_potcar_file=False,
                 parse_projected_eigen=False, parse_dos=dos)
    if not vr.converged_electronic:
        raise ValueError(f"Electronic convergence failed: {directory}")
    nsw = int(vr.parameters.get("NSW", 0))
    if static and nsw != 0:
        raise ValueError(f"Expected NSW=0 static calculation: {directory}")
    if nsw > 0 and int(vr.parameters.get("IBRION", -1)) in (1, 2, 3) and not vr.converged_ionic:
        raise ValueError(f"Ionic relaxation did not converge: {directory}")
    out = Outcar(directory / "OUTCAR")
    return {"path": directory, "vr": vr, "out": out,
            "structure": vr.final_structure,
            "energy": sigma_zero_energy(directory / "OUTCAR"),
            "input_structure": Structure.from_file(directory / "POSCAR")}


def fingerprint(run):
    p = run["vr"].parameters
    # VASP defaults are resolved in vasprun.xml rather than inferred from INCAR.
    keys = {k: p.get(k) for k in MATCH_KEYS}
    keys["potcar_titles"] = list(run["vr"].potcar_symbols)
    return keys


def potential_titles(titles):
    return {re.match(r"[A-Z][a-z]?", x.split()[1]).group(0): x for x in titles}


def compatible(bulk, defect):
    a, b = fingerprint(bulk), fingerprint(defect)
    mismatch = [k for k in MATCH_KEYS if a[k] != b[k]]
    # Defects may add or remove species; the potential for each shared element
    # must retain the same dataset title.
    av = potential_titles(a["potcar_titles"])
    bv = potential_titles(b["potcar_titles"])
    if any(av[x] != bv[x] for x in av.keys() & bv.keys()):
        mismatch.append("POTCAR dataset titles")
    if not np.allclose(bulk["structure"].lattice.matrix,
                       defect["structure"].lattice.matrix, atol=1e-5, rtol=0):
        mismatch.append("supercell lattice")
    if bulk["vr"].kpoints.as_dict() != defect["vr"].kpoints.as_dict():
        # Comments do not change sampling; compare the defining mesh and shift.
        bk, dk = bulk["vr"].kpoints, defect["vr"].kpoints
        if bk.style != dk.style or bk.kpts != dk.kpts or bk.kpts_shift != dk.kpts_shift:
            mismatch.append("k-point sampling")
    if mismatch:
        raise ValueError(f"Bulk/defect mismatch {mismatch}: {defect['path']}")


def composition_change(bulk, defect):
    a = bulk["structure"].composition.get_el_amt_dict()
    b = defect["structure"].composition.get_el_amt_dict()
    return {e: int(round(b.get(e, 0) - a.get(e, 0))) for e in sorted(a.keys() | b.keys())
            if abs(b.get(e, 0) - a.get(e, 0)) > 1e-8}


def read_charge(run):
    if not hasattr(run["out"], "zval_dict"):
        run["out"].read_pseudo_zval()
    zval = run["out"].zval_dict
    if not zval:
        raise ValueError(f"Missing valence-electron counts in OUTCAR: {run['path']}")
    neutral = sum(zval[e] * n for e, n in run["structure"].composition.get_el_amt_dict().items())
    q = neutral - float(run["vr"].parameters["NELECT"])
    if abs(q - round(q)) > 1e-5:
        raise ValueError(f"Noninteger inferred charge {q}: {run['path']}")
    q = int(round(q))
    folder = next((p for p in [run["path"], *run["path"].parents]
                   if re.fullmatch(r"charge_[-+]?\d+", p.name)), None)
    if folder and int(folder.name.split("_", 1)[1]) != q:
        raise ValueError(f"Folder charge differs from NELECT: {run['path']}")
    return q


def tensor_from_outcar(path, ionic=False):
    lines = Path(path).read_text(errors="replace").splitlines()
    key = "MACROSCOPIC STATIC DIELECTRIC TENSOR"
    matrices = []
    for i, line in enumerate(lines):
        if key not in line:
            continue
        if ionic != ("IONIC CONTRIBUTION" in line.upper()):
            continue
        if not ionic and "excluding local field" in line.lower():
            continue
        rows = []
        for line2 in lines[i+1:i+12]:
            fields = line2.split()
            if len(fields) == 3:
                try:
                    rows.append([float(x) for x in fields])
                except ValueError:
                    pass
            if len(rows) == 3:
                matrices.append(rows)
                break
    if not matrices:
        raise ValueError(f"No {'ionic' if ionic else 'electronic'} dielectric tensor in {path}")
    return np.array(matrices[-1], dtype=float)


def optical_zero(path):
    vr = Vasprun(Path(path).parent / "vasprun.xml", parse_potcar_file=False,
                 parse_projected_eigen=False, parse_eigen=False)
    if not vr.converged_electronic or not vr.dielectric:
        raise ValueError(f"Missing converged optical response: {path}")
    energy, real, _ = vr.dielectric
    if abs(float(energy[0])) > 1e-8:
        raise ValueError("Optical grid does not include zero frequency")
    xx, yy, zz, xy, yz, xz = real[0]
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def read_dielectric(root, electronic=None, ionic=None, electronic_only=False):
    if electronic_only and ionic:
        raise ValueError("Electronic-only screening cannot be combined with an ionic OUTCAR")
    ep = resolve_one(root / "dielectric/electronic", "OUTCAR", electronic) / "OUTCAR"
    ev = Vasprun(ep.parent / "vasprun.xml", parse_potcar_file=False,
                 parse_eigen=False, parse_projected_eigen=False, parse_dos=False)
    if not ev.converged_electronic:
        raise ValueError("Dielectric calculation is not electronically converged")
    ele = optical_zero(ep) if ev.parameters.get("LOPTICS", False) else tensor_from_outcar(ep)
    ip = None
    if electronic_only:
        ion = np.zeros((3, 3))
    else:
        ip = resolve_one(root / "dielectric/ionic", "OUTCAR", ionic) / "OUTCAR"
        iv = Vasprun(ip.parent / "vasprun.xml", parse_potcar_file=False,
                     parse_eigen=False, parse_projected_eigen=False, parse_dos=False)
        if not iv.converged_electronic:
            raise ValueError("Ionic dielectric calculation is not electronically converged")
        ion = tensor_from_outcar(ip, ionic=True)
    total = (ele + ion + (ele + ion).T) / 2
    if np.linalg.eigvalsh(total).min() <= 0:
        raise ValueError("Static dielectric tensor is not positive definite")
    return ele, ion, {"electronic": str(ep), "ionic": str(ip) if ip else None,
                      "screening_convention": "electronic-only" if electronic_only else "electronic+ionic"}


def orbit_multiplicity(structure, frac_coords, symprec=0.01):
    """Crystallographic site multiplicity; no assumed spin degeneracy."""
    ops = SpacegroupAnalyzer(structure, symprec=symprec).get_symmetry_operations()
    points = np.array([op.operate(frac_coords) % 1 for op in ops])
    # Cluster in Cartesian distance with periodic boundary conditions.
    unique = []
    for point in points:
        if not unique or structure.lattice.get_all_distances([point], unique).min() > symprec:
            unique.append(point)
    return len(unique)


def configuration_multiplicity(root, relative_path, bulk, delta, center, symprec=0.01):
    """Count host sites for vacancies/antisites, not displaced correction centers."""
    from pydefect.analyzer.defect_structure_comparator import DefectStructureComparator
    ideal_path = root / "defect/relaxation/bulk/POSCAR"
    ideal = Structure.from_file(ideal_path) if ideal_path.exists() else bulk["input_structure"]
    if not np.allclose(ideal.lattice.matrix, bulk["structure"].lattice.matrix, atol=1e-5):
        raise ValueError("Defect-free relaxation/static lattices differ")
    initial_path = root / "defect/relaxation" / relative_path / "POSCAR"
    if not initial_path.exists():
        initial_path = root / "defect/static" / relative_path / "POSCAR"
    initial = Structure.from_file(initial_path)
    comparison = DefectStructureComparator(initial, ideal)
    if any(n < 0 for n in delta.values()):
        removed = comparison.removed_indices
        if len(removed) != 1:
            raise ValueError(f"Cannot identify one parent host site: {initial_path}")
        coords = ideal[next(iter(removed))].frac_coords
        method = "crystallographic orbit of ideal parent host site"
    else:
        coords = comparison.defect_center_coord
        method = "crystallographic orbit of initial interstitial configuration; review across charge states"
    return orbit_multiplicity(ideal, coords, symprec), list(map(float, coords)), method


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Compound folder")
    parser.add_argument("--static", type=Path, help="Override defect/static folder")
    parser.add_argument("--bulk", type=Path, help="Override pristine static calculation")
    parser.add_argument("--dos", type=Path, help="Dense-grid unit-cell DOS calculation folder or vasprun.xml")
    parser.add_argument("--electronic-outcar", type=Path)
    parser.add_argument("--ionic-outcar", type=Path)
    parser.add_argument("--electronic-only", action="store_true",
                        help="Explicitly use electronic-only screening; do not infer a missing ionic response")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--symprec", type=float, default=0.01)
    parser.add_argument("--check", action="store_true", help="Validate inputs without writing results")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    static = (args.static or root / "defect/static").resolve()
    output = (args.output or root / "defect/postprocessing").resolve()
    bulk = load_run(resolve_one(static / "bulk", supplied=args.bulk))
    if read_charge(bulk) != 0:
        raise ValueError("Pristine bulk reference must be neutral")
    # The supercell supplies the formation-energy VBM reference, not the gap.
    _, _, vbm, _ = bulk["vr"].eigenvalue_band_properties
    unitcell_dos = load_run(resolve_one(root / "dos", supplied=args.dos), dos=True)
    if unitcell_dos["vr"].complete_dos is None:
        raise ValueError("The dense-grid unit-cell calculation must contain DOS data")
    if (unitcell_dos["structure"].composition.reduced_composition !=
            bulk["structure"].composition.reduced_composition):
        raise ValueError("Unit-cell DOS and defect-free supercell compositions differ")
    a, b = fingerprint(bulk), fingerprint(unitcell_dos)
    if any(a[k] != b[k] for k in MATCH_KEYS) or a["potcar_titles"] != b["potcar_titles"]:
        raise ValueError("Unit-cell DOS and defect-free supercell settings or datasets differ")
    gap, unitcell_cbm, unitcell_vbm, _ = unitcell_dos["vr"].eigenvalue_band_properties
    if not np.isfinite([gap, vbm, unitcell_cbm, unitcell_vbm]).all() or gap <= 0:
        raise ValueError("Unit-cell eigenvalues must provide a positive fundamental gap and finite band edges")
    cbm = vbm + gap
    band_edge_sources = {"vbm": str(bulk["path"]),
                         "fundamental_gap": str(unitcell_dos["path"]),
                         "gap_method": "dense-grid eigenvalues and occupations",
                         "cbm_convention": "defect-free supercell VBM plus unit-cell fundamental gap"}
    ele, ion, dielectric_sources = read_dielectric(root, args.electronic_outcar,
                                                 args.ionic_outcar, args.electronic_only)
    paths = sorted(p.parent for p in static.rglob("vasprun.xml")
                   if bulk["path"] not in p.parents and "bulk" not in p.relative_to(static).parts)
    if not paths:
        raise ValueError(f"No defect static outputs found under {static}")
    # A leaf static folder may contain a continuation; require one output per charge.
    checked = []
    for path in paths:
        run = load_run(path)
        compatible(bulk, run)
        charge = read_charge(run)
        rel = path.relative_to(static)
        parts = rel.parts
        if not any(re.fullmatch(r"charge_[-+]?\d+", x) for x in parts):
            raise ValueError(f"Expected a charge_x folder for defect: {path}")
        checked.append((run, charge, composition_change(bulk, run), parts[0]))
    # Detect explicitly generated but missing calculation outputs, rather than silently skipping them.
    for inc in static.rglob("INCAR"):
        if "bulk" not in inc.relative_to(static).parts and not (inc.parent / "vasprun.xml").exists():
            raise ValueError(f"Missing static output: {inc.parent}")
    if args.check:
        print(json.dumps({"status": "validated", "defect_runs": len(checked),
                          "bulk": str(bulk["path"]), "gap_eV": gap,
                          "band_edge_sources": band_edge_sources,
                          "writes": 0, "versions": versions()}, indent=2))
        return 0
    from pydefect.cli.vasp.make_calc_results import make_calc_results_from_vasp
    from pydefect.cli.main_functions import make_efnv_correction
    from monty.json import MontyEncoder
    bcr = make_calc_results_from_vasp(bulk["vr"], bulk["out"])
    output.mkdir(parents=True, exist_ok=True)
    unit = {"system": bulk["structure"].composition.reduced_formula, "vbm": float(vbm),
            "cbm": float(cbm), "ele_dielectric_const": ele.tolist(), "ion_dielectric_const": ion.tolist()}
    (output / "unitcell.yaml").write_text(yaml.safe_dump(unit, sort_keys=False))
    elements = set(bulk["structure"].composition.get_el_amt_dict())
    for run, *_ in checked:
        elements.update(run["structure"].composition.get_el_amt_dict())
    (output / "standard_energies.yaml").write_text(yaml.safe_dump({e: 0.0 for e in sorted(elements)}))
    rows = []
    for run, charge, delta, name in checked:
        if run["out"].electrostatic_potential is None or bulk["out"].electrostatic_potential is None:
            raise ValueError(f"Atomic-site potentials unavailable: {run['path']}")
        dcr = make_calc_results_from_vasp(run["vr"], run["out"])
        if len(dcr.potentials) != len(run["structure"]) or len(bcr.potentials) != len(bulk["structure"]):
            raise ValueError(f"Atomic-site potentials unavailable: {run['path']}")
        correction = make_efnv_correction(charge, dcr, bcr, (ele+ion+(ele+ion).T)/2)
        energy_correction = float(correction.correction_energy)
        baseline = run["energy"] - bulk["energy"] + charge * vbm + energy_correction
        rel = run["path"].relative_to(static)
        work = output / "corrections" / rel
        work.mkdir(parents=True, exist_ok=True)
        (work / "correction.json").write_text(json.dumps(correction, cls=MontyEncoder, indent=2))
        center = list(map(float, correction.defect_coords))
        nsites, site_coords, site_method = configuration_multiplicity(
            root, rel, bulk, delta, center, args.symprec)
        rows.append({"defect_name": name, "charge": charge,
                     "configuration": str(rel.parent), "source": str(run["path"]),
                     "delta_atoms": delta, "static_energy_sigma0_eV": run["energy"],
                     "uncorrected_formation_energy_at_vbm_eV": float(baseline-energy_correction),
                     "correction_eV": energy_correction,
                     "formation_energy_at_vbm_zero_mu_eV": float(baseline),
                     "defect_center_frac_coords": center,
                     "site_multiplicity_bulk": nsites,
                     "site_multiplicity_coords": site_coords,
                     "site_multiplicity_method": site_method,
                     "correction_diagnostics": str(work / "correction.json")})
    groups = []
    for name in sorted({r["defect_name"] for r in rows}):
        candidates = [r for r in rows if r["defect_name"] == name]
        if any(r["delta_atoms"] != candidates[0]["delta_atoms"] for r in candidates):
            raise ValueError(f"Different stoichiometries grouped as {name}")
        selected = []
        for charge in sorted({r["charge"] for r in candidates}):
            selected.append(min((r for r in candidates if r["charge"] == charge),
                                key=lambda x: x["formation_energy_at_vbm_zero_mu_eV"]))
        groups.append({"defect_name": name, "delta_atoms": selected[0]["delta_atoms"],
                       "charge_states": [r["charge"] for r in selected],
                       "formation_energies_at_vbm_eV": [r["formation_energy_at_vbm_zero_mu_eV"] for r in selected],
                       "states": selected})
    result = {"schema": SCHEMA, "compound": unit["system"], "root": str(root),
              "energy_convention": "final static energy(sigma->0)",
              "chemical_potential_reference": "absolute chemical potentials all zero",
              "vbm_eV": float(vbm), "cbm_eV": float(cbm), "band_gap_eV": float(gap),
              "bulk": {"source": str(bulk["path"]), "energy_sigma0_eV": bulk["energy"],
                       "volume_A3": bulk["structure"].volume, "fingerprint": fingerprint(bulk)},
              "band_edge_sources": band_edge_sources,
              "dielectric_sources": dielectric_sources, "versions": versions(),
              "defects": groups, "all_configurations": rows,
              "diagnostics_review": "Inspect correction.json site potentials before accepting corrections."}
    dump(output / "formation_energies.json", result)
    print(json.dumps({"status": "complete", "result": str(output / "formation_energies.json"),
                      "defect_runs": len(rows), "defect_species": len(groups)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
