# Project B — Z-stack motion stabilization

## Goal

Operational port of the Shipley 2020 MATLAB pipeline. Two acceptable endpoints:

1. **Revive in place** — get the existing MATLAB code running on the new UCI lab computer with reproducible setup.
2. **Refactor / port** — Python or cleaned-up MATLAB that produces equivalent stabilized z-stacks with substantially less intermediate file bloat.

Either outcome is useful; the constraint is that the original code is ground truth for correctness.

## Source

- Paper: Shipley FB et al. *Tracking Calcium Dynamics and Immune Surveillance at the Choroid Plexus Blood–Cerebrospinal Fluid Interface.* Neuron 2020;108(4):623–639. PMID 32961128.
- Code: [`LehtinenLab/Shipley2020`](https://github.com/LehtinenLab/Shipley2020) — cloned to `references/Shipley2020/`. Contains `registration/` (the master pipeline) and `vessel segmentation/`.

## Why this exists

Choroid plexus in vivo is mechanically free-floating in CSF: anchored at its base, the rest moves like seaweed in water. A z-stack captures slices over ~4 s; between slices the tissue has translated and rotated, so the resulting stack is geometrically incoherent. Standard motion-correction tools (designed for cortex, where the tissue is held still by the skull) do not work — the displacements are an order of magnitude larger.

The pipeline solves this by re-registering each frame to a reference using vasculature landmarks visible via an intravenous fluorescent dye.

## Current state

- ~5 years of one PhD student's work (Shipley) layered as scripts
- Generates 3–4× intermediate output per 1× useful product (huge disk burn — Boston lab routinely fills 16 TB drives)
- ~5–6 h runtime per 1 GB raw z-stack on the workstation that hosts it
- Only reproducibly runnable on **one** Harvard account's MATLAB setup; other users on the same machine hit unresolved errors
- MATLAB licensing differs by institution: UCI has it for all; Harvard only for grad students. UCI side is friendlier for porting decisions.

## Two paths

### Path 1 — make it run

- Pull the repo onto the UCI lab computer
- Reproduce the environment (MATLAB version, toolboxes, any compiled MEX dependencies)
- Use a small test stack (~5 min recording) as the error-reproduction case before touching real 1–2 h videos

### Path 2 — port

- Trace `registration/RegistrationMasterPipeline.m` to map data flow
- Identify which intermediate files are actually consumed downstream vs. accidentally retained
- Either: clean the MATLAB to drop the unused intermediates, or rewrite in Python (suitcase + DFT registration libraries exist)

## Data

Raw z-stacks live on Huixin's external drives — multi-GB per recording, not on this machine. To work with real data, copy in person from the lab (Huixin has 5× external SSDs in a box). For initial debug, ask for a short test stack.

## Constraints

- Equivalence to original pipeline is the bar — port must produce stabilized stacks indistinguishable from the existing one on the same input.
- This is not a hot priority right now; secondary to Project A. Set up the skeleton, learn the pipeline shape, defer execution until Project A has a first version shipped.
