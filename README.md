# Potential Well Project

Memory substrate architecture for continual learning.
Domains occupy non-overlapping partitions of a shared weight tensor.
Zero replay, zero regularization, zero parameter overhead.

## Status
- v2 linear partition: empirically confirmed, Split-MNIST delta -0.011pp
- v3 Grassmannian: formal derivation complete, implementation pending

## Structure
\xperiments/v2_linear\ — linear row partition experiments
\xperiments/v3_grassmannian\ — Grassmannian subspace experiments
\src\ — shared architecture code
\	ests\ — unit tests

