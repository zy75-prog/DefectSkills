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

## How to use

Open a new Codex chat, connect the agent to the authorized compute environment, attach `DefectSkills.pdf`, instruct it to follow the PDF, and provide the target-material formula and absolute parent directory. Then supply the requested scientific inputs and make decisions at the defined human-in-the-loop decision points.

The files in `optional_files/` are optional reference scripts and input templates licensed under the MIT License.

## Supporting calculation data

The VASP calculation data used in the paper are available on Zenodo:

https://doi.org/10.5281/zenodo.23930584

## License

The natural-language prompts and original figures are licensed under CC BY 4.0, while the files in `optional_files/` are licensed under the MIT License. `DefectSkills.pdf` remains subject to the license stated on arXiv. See [LICENSE.md](LICENSE.md) for details and pydefect attribution.
