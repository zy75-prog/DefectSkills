# DefectSkills

DefectSkills is a set of structured prompts that serves as natural-language code for LLM-based agents performing first-principles calculations of point defects in semiconductors and insulators using VASP. It translates scientific expertise into explicit workflow instructions, decision points, validation criteria, and recovery procedures for calculation setup, execution, postprocessing, analysis, and data backup.

Through human-in-the-loop decision points, researchers define the scientific objectives and material-specific requirements, while LLM-based agents execute and validate the prescribed workflow.

This repository accompanies:

Z. Yuan, Y. Zhu, and G. Hautier, “DefectSkills: Guiding AI agents through first-principles calculations for point defects in semiconductors and insulators,” arXiv:2611.05654 (2026).

Paper: https://arxiv.org/abs/2611.05654

## Repository contents

- `DefectSkills.tex`: LaTeX source containing Sections II and III of the paper.
- `DefectSkills.pdf`: Compiled version of the structured prompts.
- `figures/`: Workflow and user-decision figures.
- `optional_files/`: Optional reference scripts and input templates.

## Usage

Sections II and III constitute DefectSkills and are intended for direct use as agent instructions. Researchers should provide the requested scientific inputs and make decisions at the defined human-in-the-loop decision points.

The files in `optional_files/` are reference implementations and templates. They should be adapted to the installed package versions and calculation hierarchy while preserving the methods and conventions prescribed by DefectSkills.

## Supporting calculation data

The VASP calculation data used in the paper are available on Zenodo:

https://doi.org/10.5281/zenodo.23930584
