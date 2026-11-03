#!/usr/bin/env python3
"""Read postprocessed defects, phase calculations and DOS; solve equilibrium/quench.

Targets the py-sc-fermi 2.x API. No VASP inputs,
outputs, or reference notebooks are modified.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from xml.etree.ElementTree import ParseError

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp import Vasprun

from defect_postprocessing import (SCHEMA, MATCH_KEYS, compatible, dump, fingerprint,
                                   load_run, potential_titles, resolve_one, versions)


def final_phase_runs(root):
    """Select one final second-relaxation source per candidate under phase_diagram.

    The highest-numbered continuation supersedes earlier attempts. If it is
    incomplete, exclude the candidate rather than falling back to an older run.
    """
    phase_dir = (Path(root) / "phase_diagram").resolve()
    candidates = sorted({p.parent for name in ("relax_1", "relax_2")
                         for p in phase_dir.rglob(name) if p.is_dir()})
    if not candidates:
        raise ValueError("No candidate relax_1/relax_2 folders found under phase_diagram")
    selected, excluded = [], []
    for candidate in candidates:
        stage = candidate / "relax_2"
        source = stage
        try:
            if not stage.is_dir():
                raise ValueError("Missing second relaxation (relax_2)")
            continuations = [p for p in stage.iterdir()
                             if p.is_dir() and re.fullmatch(r"continue_[1-9][0-9]*", p.name)]
            if continuations:
                source = max(continuations, key=lambda p: int(p.name.split("_")[1]))
            # Do not follow links to external calculations or silently accept nested histories.
            source.resolve().relative_to(phase_dir)
            if any(p.is_dir() for p in source.glob("continue_*")):
                raise ValueError("Nested continuation folders require correction")
            run = load_run(source, static=False)
            vr = run["vr"]
            if int(vr.parameters.get("NSW", 0)) <= 0 or int(vr.parameters.get("IBRION", -1)) not in (1, 2, 3):
                raise ValueError("Source is not a structural relaxation")
            if not vr.converged_electronic or not vr.converged_ionic:
                raise ValueError("Second relaxation is not electronically and ionically converged")
            if "General timing and accounting" not in (source / "OUTCAR").read_text(errors="replace"):
                raise ValueError("Normal VASP termination is missing")
            forces = np.asarray(vr.ionic_steps[-1]["forces"], dtype=float)
            max_force = float(np.linalg.norm(forces, axis=1).max())
            if not np.isfinite(max_force) or max_force > 0.01:
                raise ValueError(f"Maximum force {max_force} exceeds 0.01 eV/Angstrom")
            run["candidate"] = candidate
            run["max_force_eV_A"] = max_force
            selected.append(run)
        except (ValueError, OSError, KeyError, IndexError, ParseError) as exc:
            excluded.append({"candidate": str(candidate), "source": str(source), "reason": str(exc)})
    return selected, excluded


def chemical_potential_vertices(root, data):
    """Use only final converged phase_diagram second relaxations, including the host."""
    from pymatgen.core import Composition
    runs, excluded = final_phase_runs(root)
    if excluded:
        print(json.dumps({"excluded_phase_candidates": excluded}, indent=2), file=sys.stderr)
    if not runs:
        raise ValueError("No completed second relaxations under phase_diagram")
    ref = data["bulk"]["fingerprint"]
    entries, provenance = [], []
    for run in runs:
        path = run["path"]
        params = fingerprint(run)
        bad = [k for k in MATCH_KEYS if params[k] != ref[k]]
        if bad:
            raise ValueError(f"Phase/defect functional mismatch {bad}: {path}")
        # Match PAW variants for every shared species; never read proprietary POTCAR files.
        ref_titles = potential_titles(ref["potcar_titles"])
        titles = potential_titles(params["potcar_titles"])
        if any(titles[x] != ref_titles[x] for x in titles.keys() & ref_titles.keys()):
            raise ValueError(f"Phase POTCAR version mismatch: {path}")
        comp = run["structure"].composition
        entries.append(ComputedEntry(comp, run["energy"], entry_id=str(path)))
        provenance.append({"source": str(path), "candidate": str(run["candidate"]),
                           "formula": comp.reduced_formula, "energy_sigma0_eV": run["energy"],
                           "max_force_eV_A": run["max_force_eV_A"]})
    target = Composition(data["compound"])
    host = [e for e in entries if e.composition.reduced_composition == target.reduced_composition]
    if not host:
        raise ValueError("Host completed second-relaxation energy is missing under phase_diagram")
    # Use all required elements, including dopants; missing elemental endpoints
    # cause pymatgen to raise rather than silently substituting zero.
    pd = PhaseDiagram(entries)
    best = min(host, key=lambda e: e.energy_per_atom)
    if pd.get_e_above_hull(best) > 1e-6:
        raise ValueError("Target is above the calculated hull; no equilibrium stability region")
    vertices = pd.get_all_chempots(target)
    if not vertices:
        raise ValueError("No chemical-potential vertices found")
    result = {name: {str(e): float(mu) for e, mu in values.items()}
              for name, values in vertices.items()}
    needed = {e for d in data["defects"] for e in d["delta_atoms"]}
    for name, mus in result.items():
        if not needed <= mus.keys():
            raise ValueError(f"Chemical potentials for {sorted(needed-mus.keys())} missing at {name}. "
                             "Include dopant-containing phases or provide their calculated phase results.")
    return result, provenance


def read_dos(path, target_gap):
    """Use the unit-cell DOS and eigenvalue gap, with its VBM as zero."""
    from py_sc_fermi.dos import DOS
    run = load_run(path, static=True, dos=True)
    vr = run["vr"]
    if vr.complete_dos is None:
        raise ValueError(f"DOS is missing from {path}")
    dos_gap, _, dos_vbm, _ = vr.eigenvalue_band_properties
    energy = np.asarray(vr.complete_dos.energies, dtype=float) - dos_vbm
    densities = vr.complete_dos.densities
    if target_gap <= 0 or dos_gap <= 0:
        raise ValueError("The unit-cell DOS calculation must be semiconducting")
    if not np.isclose(target_gap, dos_gap, atol=1e-6, rtol=0):
        raise ValueError("Formation-energy gap differs from the unit-cell DOS eigenvalue gap; "
                         "regenerate postprocessing with the same unit-cell DOS calculation")
    if not np.all(np.diff(energy) > 0):
        raise ValueError("DOS energy grid must be strictly increasing")
    if energy.min() >= 0 or energy.max() <= target_gap:
        raise ValueError("DOS energy window must include both occupied and unoccupied bands")
    spin = len(densities) == 2
    values = np.array([densities[Spin.up], densities[Spin.down]]) if spin else np.array(densities[Spin.up])
    nelect = float(vr.parameters["NELECT"])
    if abs(nelect-round(nelect)) > 1e-6:
        raise ValueError("Unit-cell DOS must have an integer electron count")
    model = DOS(dos=values, edos=energy, bandgap=dos_gap,
                nelect=int(round(nelect)), spin_polarised=spin)
    meta = {"source": str(path), "volume_A3": run["structure"].volume,
            "nelect": nelect, "dos_vbm_eV": float(dos_vbm),
            "original_gap_eV": float(dos_gap), "gap_used_eV": float(dos_gap),
            "conduction_energy_shift_eV": 0.0,
            "spin_polarised": spin}
    return model, run, meta


def shifted_energy(defect, energy, chemical_potentials):
    return float(energy - sum(n * chemical_potentials[e] for e, n in defect["delta_atoms"].items()))


def build_system(data, dos, volume, chemical_potentials, temperature, degeneracy,
                 site_multiplicities_per_formula=None, formula_units=1):
    from py_sc_fermi.defect_charge_state import DefectChargeState
    from py_sc_fermi.defect_species import DefectSpecies
    from py_sc_fermi.defect_system import DefectSystem
    species = []
    normalized = {}
    overrides = site_multiplicities_per_formula or {}
    unknown = set(overrides) - {d["defect_name"] for d in data["defects"]}
    if unknown:
        raise ValueError(f"Unknown defects in site-multiplicity overrides: {sorted(unknown)}")
    for defect in data["defects"]:
        name = defect["defect_name"]
        if name in overrides:
            nsites = overrides[name] * formula_units
        else:
            multiplicities = {x["site_multiplicity_bulk"] for x in defect["states"]}
            if len(multiplicities) != 1:
                raise ValueError(f"Site multiplicity changes with charge for {name}; resolve configuration/site mapping first")
            nsites = multiplicities.pop() * volume / data["bulk"]["volume_A3"]
        if abs(nsites-round(nsites)) > 0.05:
            raise ValueError(f"Noninteger site count in DOS cell for {name}: {nsites}; verify defect-site symmetry")
        nsites = int(round(nsites))
        if nsites < 1:
            raise ValueError(f"Invalid site count for {name}")
        charge_states = {}
        for q, base in zip(defect["charge_states"], defect["formation_energies_at_vbm_eV"]):
            energy = shifted_energy(defect, base, chemical_potentials)
            charge_states[q] = DefectChargeState(charge=q, degeneracy=degeneracy, energy=energy)
        species.append(DefectSpecies(name=name, nsites=nsites, charge_states=charge_states))
        normalized[name] = nsites
    return DefectSystem(defect_species=species, dos=dos, volume=volume,
                        temperature=temperature), normalized


def snapshots(system, synthesis_temperature, quench_temperature):
    """Freeze species totals in counts/cell, then re-equilibrate charge states."""
    high = deepcopy(system)
    high.temperature = synthesis_temperature
    high_cell = high.concentration_dict(per_volume=False)
    high_cm3 = high.concentration_dict(per_volume=True)
    quenched = deepcopy(system)
    quenched.temperature = quench_temperature
    for name in high.defect_species_names:
        quenched.defect_species_by_name(name).fix_concentration(high_cell[name])
    low_cell = quenched.concentration_dict(per_volume=False)
    for name in high.defect_species_names:
        if not np.isclose(low_cell[name], high_cell[name], rtol=1e-8, atol=1e-100):
            raise ValueError(f"Quench failed to conserve total concentration: {name}")
    return {"synthesis": {"temperature_K": synthesis_temperature, "concentrations_cm3": high_cm3,
                          "charge_states_cm3": high.concentration_dict(decomposed=True)},
            "quenched": {"temperature_K": quench_temperature,
                         "concentrations_cm3": quenched.concentration_dict(),
                         "charge_states_cm3": quenched.concentration_dict(decomposed=True)},
            "fermi_level_reference": "VBM = 0 eV",
            "frozen_quantity": "total concentration of each defect species"}


def plot_formation_energies(data, mus, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fermi = np.linspace(0, data["band_gap_eV"], 501)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for defect in data["defects"]:
        lines = [shifted_energy(defect, e, mus) + q * fermi
                 for q, e in zip(defect["charge_states"], defect["formation_energies_at_vbm_eV"])]
        ax.plot(fermi, np.min(lines, axis=0), label=defect["defect_name"])
    ax.set(xlabel="Fermi level relative to VBM (eV)", ylabel="Formation energy (eV)",
           xlim=(0, data["band_gap_eV"]))
    ax.legend(fontsize="small", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Compound folder")
    parser.add_argument("--results", type=Path, help="Postprocessing JSON override")
    parser.add_argument("--dos", type=Path, help="DOS calculation folder override")
    parser.add_argument("--condition", help="Calculated chemical-potential vertex name")
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--synthesis-temperature", type=float, nargs="+", help="Synthesis temperatures in K")
    parser.add_argument("--quench-temperature", type=float, help="Requested quench temperature in K")
    parser.add_argument("--degeneracy", type=int,
                        help="Explicit common charge-state degeneracy convention; not inferred from VASP")
    parser.add_argument("--site-multiplicity", action="append", default=[], metavar="DEFECT=COUNT",
                        help="Explicit available-site count per formula unit; repeat for different defects")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate discovered inputs without solving/writing")
    args = parser.parse_args(argv)
    if int(importlib_version().split(".")[0]) != 2:
        raise ValueError("This script targets py-sc-fermi 2.x; use a compatible environment")
    root = args.root.expanduser().resolve()
    result_path = args.results or root / "defect/postprocessing/formation_energies.json"
    data = json.loads(result_path.read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError("Unrecognized formation-energy JSON schema")
    if not data["defects"]:
        raise ValueError("No defects in formation-energy results")
    for d in data["defects"]:
        if len(d["charge_states"]) != len(d["formation_energies_at_vbm_eV"]):
            raise ValueError(f"Charge/energy list mismatch: {d['defect_name']}")
    vertices, phase_sources = chemical_potential_vertices(root, data)
    if args.list_conditions:
        print(json.dumps({"absolute_chemical_potentials_eV_per_atom": vertices}, indent=2))
        return 0
    if args.condition not in vertices:
        raise ValueError(f"Select --condition from: {list(vertices)}")
    if args.degeneracy is None or args.quench_temperature is None:
        raise ValueError("Specify --degeneracy and --quench-temperature explicitly")
    if args.degeneracy < 1 or args.quench_temperature <= 0:
        raise ValueError("Degeneracy and temperature must be positive")
    if not args.synthesis_temperature or any(t <= 0 for t in args.synthesis_temperature):
        raise ValueError("Provide positive --synthesis-temperature values")
    dos_path = resolve_one(root / "dos", supplied=args.dos)
    dos, dos_run, dos_info = read_dos(dos_path, data["band_gap_eV"])
    # Check shared functional/cutoff settings against the bulk reference.
    fp = fingerprint(dos_run)
    mismatch = [k for k in MATCH_KEYS if fp[k] != data["bulk"]["fingerprint"][k]]
    if mismatch:
        raise ValueError(f"DOS/bulk functional mismatch: {mismatch}")
    mus = vertices[args.condition]
    site_overrides = {}
    for item in args.site_multiplicity:
        name, value = item.split("=", 1)
        count = float(value)
        if name in site_overrides or not np.isfinite(count) or count <= 0:
            raise ValueError(f"Invalid or duplicate site multiplicity: {item}")
        site_overrides[name] = count
    formula_units = dos_run["structure"].composition.get_reduced_composition_and_factor()[1]
    system, sites = build_system(data, dos, dos_info["volume_A3"], mus,
                                 args.synthesis_temperature[0], args.degeneracy,
                                 site_overrides, formula_units)
    if args.check:
        print(json.dumps({"status": "validated", "condition": args.condition,
                          "dos": dos_info, "sites_per_DOS_cell": sites, "writes": 0}, indent=2))
        return 0
    reports = [snapshots(system, t, args.quench_temperature) for t in args.synthesis_temperature]
    out = args.output or root / "defect/postprocessing/concentrations"
    out.mkdir(parents=True, exist_ok=True)
    result = {"schema": "agent-concentrations/1", "condition": args.condition,
              "absolute_chemical_potentials_eV_per_atom": mus, "formation_energy_source": str(result_path),
              "phase_calculations": phase_sources, "dos": dos_info,
              "sites_per_DOS_cell": sites, "charge_state_degeneracy_convention": args.degeneracy,
              "explicit_site_multiplicities_per_formula_unit": site_overrides,
              "versions": versions(), "runs": reports}
    dump(out / "concentrations.json", result)
    if not args.no_plot:
        plot_formation_energies(data, mus, out / "formation_energies.png")
    print(json.dumps({"status": "complete", "result": str(out / "concentrations.json")}, indent=2))
    return 0


def importlib_version():
    from importlib.metadata import version
    return version("py-sc-fermi")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
