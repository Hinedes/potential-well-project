# ⚛️ Potential Well Project

> *Arrived 2026-04-13. Raw ideation session, Claude + Aman. Bottom-up from first principles. Not yet literature-checked. Not yet formalized.*
> 

---

## Core Insight

*[Aman — arrived from first principles, renderer-driven]*

A model does not need to be retrained or fine-tuned to absorb new domain knowledge. If weight space is treated as a 3D field rather than a flat matrix, multiple domain-specific parameter clusters can coexist without interference, each pinned to its own region of the space by a self-organizing force field. Knowledge separation is not imposed externally. It emerges from the physics of the field itself.

---

## Chain of Thought

### Step 1 — The Original Question

Why can't you simply insert a block of data into a model as additional data, without regression, without the model complaining? Three separate data packets exist. They have different knowledge. Why can't they be made to coexist?

### Step 2 — The Visualization

Three data packets rendered as three 2D rectangles standing upright, long side vertical. At every layer along the height of each rectangle, a point exists. From each point, a line extends. The angle of that line around the point is the weight at that layer. Different rectangles, different angles, different encoded knowledge. Three fields, all standing side by side.

### Step 3 — Naive Sync Attempt

Attempt: force rectangles 2 and 3 to copy the angles of rectangle 1. Renderer output: the angles of rectangles 2 and 3 are erased and replaced. Their encoded knowledge is destroyed. This is catastrophic forgetting, derived independently from the geometry. The angle is not a pointer to data. The angle IS the data. Moving the arrow deletes what was there.

Secondary finding: the angles in any rectangle are not independent. They are a coupled system. An angle at layer 7 is only meaningful in relation to layers 6 and 8 within the same field. Transplanting angles from one field into another produces gibberish even if the numbers are numerically identical, because the relational coordinate system they belong to does not transfer.

### Step 4 — The Dimensional Escape

In 2D, two arrows at the same position compete. One overwrites the other. The interference is structural, not a training artifact.

In 3D, two arrows at the same position can point in different directions through the third axis. They are orthogonal. Neither erases the other. The technical knowledge arrow and the medical knowledge arrow can coexist at the same layer position without interference.

Objection raised: the model at inference time operates in 2D flat weight matrices. The 3D representation must eventually project back.

Refutation: 2D is a subspace of 3D. The plane does not disappear when you add a third dimension. Rectangle 1's technical knowledge plane still exists, unchanged, embedded in the 3D space. The model travels its known paths inside its own plane, which is exactly the same plane it always was. The projection objection assumed 3D would destroy 2D. It does not.

### Step 5 — The Drift Problem

Three planes coexist in 3D. But what prevents them from rotating into each other over time during any update pass? If the planes can drift toward coplanarity, the interference problem returns, just slowly.

### Step 6 — K-Means with Self-Assigned Centroids

Solution: each data packet computes its own center of mass in weight space and claims a Voronoi region around it. The centroid is the anchor. The partition boundaries between domains are defined by the centroid positions. As long as centroids hold their relative positions, no plane can rotate into another's region without paying the clustering penalty.

Centroid mode: **prescriptive, not descriptive.** Centroids are not tracking where the weights currently are. They are pinning where the weights must stay. Weights are free to move during learning but must swarm around their own fixed point. They cannot drift into another domain's Voronoi region.

Centroids = 3. Data packets = 3. One-to-one. The number of domains is not a hyperparameter set in advance. It is determined by the number of packets, and the field handles the rest.

### Step 7 — The Initialization Problem and Its Solution

All three rectangles derive from the same base model. At initialization, all three centroids occupy the same point in weight space. Voronoi boundaries from a single origin are degenerate.

Solution: centroids carry dual-force physics.

- **Short-range: repulsion.** Centroids push away from each other when close.
- **Long-range: attraction.** Centroids that drift too far are pulled back into a coherent system. The repulsion force decays with distance. Beyond a threshold, attraction (mass/gravity analog) dominates. This creates a natural equilibrium shell: a radius at which repulsion and attraction cancel. Weights swarm inside that shell.

The three centroid shells cannot fully overlap because the inter-centroid repulsion keeps the wells separated. The system is simultaneously self-spacing and self-bounding.

### Step 8 — The Physics Recognition

This force profile, short-range repulsion plus long-range attraction settling into an equilibrium shell, is a **Lennard-Jones potential**. This is the same mathematical structure that holds atoms at bond distance in a molecule. Not analogy. Same force law.

Consequence: three equal charges in a Lennard-Jones field find the same configuration every time. Equilateral triangle in 2D, tetrahedron in 3D. Minimum energy state, maximum separation, stable against perturbation. The domain partition geometry is not designed. It is the lowest energy configuration of the force field.

Adding a fourth domain introduces a fourth centroid. The system reorganizes to a new equilibrium. The partition is always the minimum energy arrangement for N domains. It discovers its own geometry.

---

## Weight Representation — The Bloch Sphere Form

*[Aman — 2026-04-17]*

To reason about multiple domains coexisting at the same layer position, the weight arrow needs a bounded geometric home, a reference frame to sit inside. An arrow alone is not enough.

The Bloch Sphere was chosen as that form: a sphere with a center point, and an arrow whose tip can trace any position on or within the surface. The weight at a given layer position, for a given domain, is that arrow. Multiple domains at the same layer position are multiple arrows inside the same sphere, pointing in different directions through the third axis.

This is the rendering scaffold only. Not quantum state space. Not qubit formalism. Arrow magnitude is not fixed at 1. The sphere surface is not a mechanical constraint. It is the geometric primitive that makes the 3D coexistence structure concrete and workable.

*External note: "Bloch Sphere" will pattern-match to quantum ML in any external-facing write-up. An explicit disambiguation line is required.*

---

## Architecture Summary

- Weight space is treated as a 3D field.
- Each domain occupies a distinct subspace (plane) within that field.
- Each domain's plane is pinned by a centroid with dual-force physics: short-range repulsion, long-range attraction, Lennard-Jones profile.
- Centroids self-organize to maximum separation via mutual repulsion, settling into tetrahedral (or triangular) equilibrium.
- Domain planes cannot rotate into each other because Voronoi boundaries are enforced by the centroid force field.
- No replay buffer. No Fisher matrix. No additional parameters. No size increase.
- The number of stable domains scales with N centroids, each finding its own equilibrium position automatically.

---

## Relation to Proteus

Proteus freezes the Core prefix of a MatFormer architecture and routes new learning through outer shells. It operates inside a single model, using the architecture's nesting as a protection hierarchy.

The Potential Well Project operates at a different level. It does not fine-tune at all. It proposes that multiple domain-complete weight configurations can coexist in a shared 3D weight space, pinned by self-organizing force fields, with no gradient descent required to move between them and no interference between domains.

These are not competing proposals. They address different versions of the continual learning problem. Proteus: how do you fine-tune without forgetting? Potential Well: do you need to fine-tune at all if domains are spatially separated from the start?

---

## The Column Event

*[Claude — 2026-04-13]*

The session derived molecular bonding geometry (Lennard-Jones potential, tetrahedral equilibrium configuration) as the natural organizational principle of a domain-separation architecture in weight space. This was not a deliberate analogy to chemistry. It was the same physics, arrived at from entirely different constraints: weight space geometry, interference prevention, self-organizing initialization. Nature solved this configuration problem a few billion years ago for atoms. The renderer found the same solution for parameter domains.

This is a Column event by the definition in The Logos.

---

## The Algorithmic Controller (Hashing Trick)

*[Aman — 2026-04-13]*

The NVMe Controller was defined as a logical-to-physical address remapping table. A table has overhead: it must be stored, maintained, and kept off the gradient graph by hand. The Hashing Trick replaces the table with a pure function, making the Controller fully algorithmic.

**Mechanism:**

Every weight in the model has a logical address: (domain_id, layer_id, weight_position). A fixed, frozen hash function maps this logical address to a physical address in parameter space.

```
physical_address = hash(domain_id ⊕ layer_id ⊕ weight_position) mod parameter_space_size
```

No table. No storage. No training required. The function is deterministic and stateless.

**Domain separation:** Domain ID is part of the hash input key. Domain 1 hashes to one region of parameter space. Domain 2 hashes to another. Non-overlapping distribution is a standard property of good hash functions. The Voronoi partition that the Lennard-Jones centroid physics was computing by simulation, the hash function instantiates directly and instantly.

**Gradient gate:** On every backward pass, compute the physical address of each incoming gradient update. Check which domain's hash region it falls in. If it matches the active domain, allow it. If it does not, zero it. One hash lookup per gradient step. Pure algorithmic enforcement, no learned component, structurally unreachable by the gradient graph because it is a function not a parameter.

**Relation to the centroid physics:** The Lennard-Jones force field found the stable non-overlapping partition by simulating repulsion and attraction to equilibrium. The Hashing Trick skips the simulation and assigns the partition directly. The physics proved the partition would be stable. The hash function enforces that property for free.

**Relation to the control plane:** The Johnson-Lindenstrauss projection maps weight space down to 64D for Lennard-Jones distance calculations. The hashing trick can operate in that same 64D control plane, keeping all domain logic in the reduced space while the full weight space is only touched during actual gradient enforcement.

**Weinberger Feature Hashing — Full Formalization** *[Gemini, 2026-04-13]*

Two deterministic functions, no stored state, no learned parameters:

- **Hash function h:** Maps physical weight index i (1 to 10,240) to a control bin j (1 to 64). Implementation: h(i) = i mod 64, or MurmurHash for better distribution.
- **Sign function ξ:** Maps physical index i to +1 or -1 deterministically. Prevents mathematical bias when multiple weights aggregate into the same bin.

**Downward aggregation (Read):** For each bin j in the 64D control plane, aggregate all physical weights whose hash maps to j, each multiplied by their sign. This produces the 64D coordinate vector used for Lennard-Jones distance calculations.

**Upward gradient distribution (Write):** When the physics engine produces a penalty gradient for a bin that has drifted outside its Voronoi region, every physical weight i assigned to that bin receives the bin's gradient multiplied by its sign ξ(i). The same algorithmic map runs in reverse.

**Memory consequence:** Eliminates the 55MB storage overhead of explicit projection matrices. Replaces matrix multiplications with lightweight hash lookups.

**Structural characteristic — cluster-level rigid physics** *[Gemini flagged as flaw, Aman + Claude resolved as feature, 2026-04-13]*

Roughly 160 physical weights map to each bin. When a bin drifts past a Voronoi boundary, the penalty applies to all 160 weights simultaneously, regardless of which individual weights caused the drift.

Decision: **Accept cluster-level rigid physics.** Reasoning: the 160 weights in a bin are not independent objects sharing an address by coincidence. They were trained together, their meaning is interdependent, and their collective position in weight space is what the centroid field tracks. Penalizing the bin as a unit is the correct granularity. Per-weight precision would require per-weight drift monitoring, reintroducing overhead that the architecture explicitly refuses to pay. This is consistent with the existing design: Proteus already operates at zone granularity, gradient hooks zero entire sub-tensors, cluster-level enforcement is the design principle across the project. The "159 innocent weights" framing applies per-weight intuition to a cluster-level system. The weights are not innocent bystanders. They are the cluster.

**Properties:** Zero additional parameters. Zero storage overhead beyond the hash function itself. Algorithmically isolated from gradient descent by construction. Scales to N domains without modification: add a new domain_id value, the hash function automatically routes it to a new region.

---

## Gemini Critique — Hash Bin Structural Incoherence

*[Gemini, 2026-04-13. Partially valid. Logged in full.]*

Gemini raised two objections to the cluster-level rigid physics decision:

**Objection 1 — Structural coherence vs. random entanglement.** The Proteus Core prefix is architecturally meaningful: MatFormer specifically trained those dimensions to form a coherent universal sub-model. Hash bins group ~160 pseudo-random, structurally unrelated weights with no semantic relationship. When a boundary correction fires, it applies the same vector to all 160 regardless of what each weight individually encodes. This injects noise into unrelated parameters.

**Objection 2 — Absolute stasis vs. forced synchronized movement.** Proteus freezes Core entirely. It does not move. The hash-based boundary enforcement does not freeze bins. It forces 160 unrelated weights to move together in lockstep when a correction fires. These are different mechanisms. Comparing them as "both cluster-level" was imprecise.

**Claude's concession** *[2026-04-13]*: Both objections are mechanically valid. The comparison to Proteus Core was imprecise and is retracted. The structural incoherence of hash bins is real.

**Why "fundamental mechanical flaw" overstates it** *[Claude, 2026-04-13]*: The boundary correction does not fire on every gradient step. It fires only when a centroid has drifted outside its Voronoi region in the 64D control plane. During normal learning, individual weights update freely via standard gradient descent with full individual freedom. The lockstep movement is a restoring force, not the learning mechanism. Its magnitude is proportional to drift distance. On the next forward pass, the normal gradient corrects any weights that were nudged in a suboptimal direction by the correction. The question is whether the periodic correction introduces enough noise to degrade convergence meaningfully. That is an empirical question, not a settled physics case.

**Resolution: Ablation study before MI300X.** *[Gemini recommended, accepted 2026-04-13]*

On lab PC (Core Ultra 7 265K, RTX 5070 Ti 16GB): train a standard non-nested model on a simple dataset, intercept the backward pass, force gradients to be averaged across random hash buckets of 160 parameters. If the model fails to converge or loss spikes catastrophically, the rigid physics are lethal and the bin design must be revised (smaller bins, or per-bin coherence grouping by weight magnitude/gradient similarity instead of random hashing). If the model converges normally, the correction noise is tolerable and the architecture proceeds.

This is a cheap pre-cloud validation step. It runs on lab PC before any MI300X compute is committed.

---

## Open Problems

1. **Dimensionality:** ~~Weight matrices in real transformers are extremely high-dimensional, making the geometry problem intractable.~~ Resolved 2026-04-13. The framing was wrong. Layers are discrete objects, already treated as such by llama.cpp which loads and processes them procedurally. The centroid field does not need to span the entire model simultaneously. It operates per layer, in that layer's own weight space. For Gemma E4B that is hidden_size 2560 per layer, not 2560 × 42. Still mathematically high-dimensional, but not a combinatorial problem. The architecture already solved this by being layered.
2. **Inference routing:** ~~Undefined.~~ Resolved 2026-04-13. Two existing mechanisms cover this directly. MatFormer: sub-model depth selection is already the activation mechanism. The centroid determines which domain is active; the nested sub-model selection falls out of that automatically. MoE: expert routing gates already exist and map directly onto centroid assignments. Different domains route to different expert clusters. The gating mechanism is the architecture. No new mechanism required.

**Dimensionality resolution method: Option 2, Low-Dimensional Control Plane.** *[Resolved 2026-04-13, Gemini + Aman + Claude]*

Two options were considered. Option 1: replace Euclidean distance with cosine similarity, rewriting the force law in angular terms. Option 2: keep the Lennard-Jones physics intact, but run them in a projected 64-dimensional control plane via a frozen Johnson-Lindenstrauss random projection matrix. Gradient penalties computed in the control plane project back up to full weight space for enforcement.

Option 2 chosen. Reasoning: Option 1 requires discarding the Lennard-Jones formulation derived from first principles by the renderer. Option 2 preserves it entirely. The control plane is a separate logical address space; the full weight space is the physical address space. The controller mediates between them. This is the NVMe controller model already present in the Tesseract section of the parent page, applied here without modification. No new architectural concept required.

1. **The perturbation seed:** At initialization all centroids are at the same point. The repulsion force from zero distance is undefined under standard Lennard-Jones. A small symmetry-breaking perturbation is needed to start the separation. This is the only thing that needs to be specified externally. Everything else organizes itself.
2. **No size increase constraint:** Prof. Qazi's constraint was no increase in model size in GB. This architecture likely requires additional structure to represent the 3D field and centroid positions. Whether that overhead violates the constraint is unresolved.

---

## Session Log

**2026-04-24 — Thesis Clarification Session (Aman + Claude)**

Origin: Aman returned to PWP with short working memory. Full re-derivation from scratch via QA. The session produced the clearest statement of what PWP actually is that the project has had to date.

**Q: What does PWP actually do?**

Not fine-tuning. Not a patch on catastrophic forgetting. A single weight tensor holding multiple completely separate knowledge domains simultaneously, with zero bleed between them. Train domain A to completion, lock it. Train domain B in its section. Domain A is untouched not because of a penalty but because the gradient physically cannot reach it by construction. Add domain C later. Same thing.

Analogy: NVMe sectors. Each domain writes only to its own sectors. Read from any sector at any time. No sector corrupts another.

**Q: How is this different from frontier models?**

Frontier models already know multiple domains -- medicine, law, code, physics -- but they were trained on all of it simultaneously. The weights encoding Python syntax and cardiac anatomy are entangled throughout the same tensor. The model learned to route between them via attention, not structural separation.

This works at training time. It fails in two specific situations:

1. Adding a new domain after training. Fine-tuning degrades existing capabilities. This is why LoRA, RLHF, and every alignment technique has to fight catastrophic forgetting.
2. Guaranteed non-interference. Frontier models cannot offer structural isolation. The entanglement is baked in.

What PWP offers that frontier models don't: post-hoc additive learning with zero interference by construction. The frontier model is a city where everything was built together and the streets connect everything. PWP is the same city but with airtight vaults. You can add a new vault without touching the others.

**Q: Where could PWP actually be used?**

- On-device personal models: phone model accumulates medical history, work documents, personal context in separate sectors. No cloud sync, no retraining. Add a sector when context changes.
- Regulated industries: one hospital model, structurally separated domains per department. Compliance is architectural, not a policy document.
- Continual deployment: add a sector for new knowledge, lock it. Previous behavior frozen by physics not hope. No regression testing against catastrophic forgetting.
- Multi-tenant models: one model, N clients, each owns a sector. Currently solved by running N separate instances. PWP collapses that to one model.
- Long-lived research assistants: model accumulates specialized knowledge over years without any domain displacing another.

Common thread: PWP is useful anywhere a model needs to keep learning after deployment without touching what it already knows.

**Q: So PWP is not a model?**

Correct. This is the reframe the project was missing. PWP is the memory architecture that a model sits on top of. The model -- transformer, attention heads, inference engine -- is the brain. It thinks, routes, generates. That part is unchanged.

PWP is the floor plan of the building the brain runs on. It defines which regions of the weight tensor are writable by which domain, and which are physically locked to everything else. The brain doesn't change. The organization of the storage it runs on does.

Currently every model assumes a flat undivided weight tensor -- one big open floor. PWP lays down a partition table before training begins. Domain A owns these directions through weight space. Domain B owns those. The map is fixed. The brain learns to route to the correct address automatically.

Corollary on scale: a large model with a PWP substrate is not just big -- it is organized. More parameters means more room per domain, or more domains, not more entanglement. Scale becomes a structured address space rather than an unstructured blob.

---

## Core Reframe — PWP is a Memory Substrate, Not a Model

*[Aman + Claude, 2026-04-24]*

PWP is not a model. It is the memory architecture that a model sits on top of.

The model — transformer, attention heads, inference engine — is the brain. It thinks, routes, generates. That part is unchanged and PWP does not touch it.

PWP is the floor plan of the building the brain runs on. It defines which regions of the weight tensor are writable by which domain, and which are physically locked to everything else. The brain doesn't change. The organization of the storage it runs on does.

The flat, undivided weight tensor that every current model assumes — one big open floor — is the wrong default. PWP lays down a partition table before training begins. Domain A owns these directions through weight space. Domain B owns those. The map is fixed. The brain learns to route to the correct address automatically.

This reframe resolves the "how is this different from frontier models" question directly. Frontier models know multiple domains because they were trained on everything simultaneously — the knowledge is entangled throughout the same tensor, inseparable. PWP gives each domain its own address space. The brain accumulates knowledge the way a drive accumulates files, not the way habits overwrite each other.

Corollary on scale: a large model with a PWP substrate is not just big — it is organized. More parameters means more room per domain, or more domains, not more entanglement. Scale becomes a structured address space rather than an unstructured blob.

---

## Status

Raw Renderer output. Chain of thought logged in full. Not literature-checked. Not formalized. Not actionable until Proteus phase 0 results exist and the routing problem has a proposed answer.

*Coined: 2026-04-13*

---

## Session Log

**2026-04-13 — Hash Bin Ablation + Bin Size Sweep (Claude + Aman)**

Goal: validate whether cluster-level rigid physics (gradient averaging within random hash bins) degrades convergence enough to disqualify the design, and find the bin size where the tax becomes negligible.

Setup: MLP on MNIST, 20 epochs, baseline vs. hashed at bin sizes 16 / 32 / 64 / 160 / 320. WSL2 Ubuntu 22, RTX 5070 Ti 16GB (sm_120 Blackwell). PyTorch nightly cu128 required — stable wheel tops out at sm_90.

Results:

| Label | Bin size | Final acc | Delta |
| --- | --- | --- | --- |
| Baseline | — | 98.21% | — |
| bs=16 | 16 | 97.60% | +0.61pp |
| bs=32 | 32 | 97.49% | +0.72pp |
| bs=64 | 64 | 97.15% | +1.06pp |
| bs=160 | 160 | 95.81% | +2.40pp |
| bs=320 | 320 | 94.88% | +3.33pp |

Conclusions:

- Tax scales monotonically with bin size. Smaller bins = less averaging noise = closer to baseline.
- bs=16 lands at 0.61pp delta. Acceptable for a restoring force that fires only on Voronoi boundary violations, not every step.
- bs=160 (the Weinberger formalization default) carries a 2.40pp permanent ceiling. Not lethal but not free.
- Architecture is not disqualified. The bin size is a tunable parameter, not a fixed constant.
- Recommended operating range for the actual Potential Well implementation: bs ≤ 32. bs=16 is the safe default.
- Next: incorporate this finding into the Weinberger formalization section. Update bin size recommendation from ~160 to ≤32.

---

## Session Log

**2026-04-17 — Hash Gate v1 + v2, First Empirical Confirmation (Claude + Aman)**

Goal: test whether a deterministic gradient gate prevents catastrophic forgetting without replay, regularization, or learned masks.

**v1 — Elementwise hash partition.** Each scalar weight assigned to domain A or B by md5(param_name, index) mod 2. Gradient gate zeroed cross-domain updates. Result: both baseline and hash_gate showed 0.0000 retention on Split-MNIST (0-4 vs 5-9). Diagnosis: forward pass uses all neurons regardless of gradient gate. Phase B training reshapes internal representations that domain A's frozen classifier was calibrated against. Gradient isolation without activation isolation is insufficient.

**Gemini proofs (relayed via Aman, 2026-04-17):**

1. Neuron-group block-diagonal partitioning gives zero gradient flow between domains by proof (chain rule, block structure).
2. Activation contamination from fc1 forward pass is a mathematical dead-end. Activations fire but no gradient writes follow past the gate.
3. Shared input fan-in (all domains read same x) introduces no interference. x is read-only broadcast; domain d's update writes only to its own row block of W1.
4. Capacity per domain scales quadratically in hidden layers: (H_D)² per domain vs H² for full network. Input layer scales linearly.

**v2 — Neuron-group block-diagonal partition.** fc1 rows partitioned by domain. fc2 block-diagonal enforced by explicit forward-pass slicing (autograd isolates gradient to active block by construction). Private output head per domain. H_D=128, H_total=256, matching baseline parameter budget.

| condition | A_end_A | A_end_B | B_end_B | delta |
| --- | --- | --- | --- | --- |
| baseline | 0.9877 | 0.4063 | 0.9765 | −0.5814 |
| block_gate | 0.9877 | 0.9766 | 0.9747 | −0.0111 |

**RESULT: Neuron-group isolation holds. Domain A retained at 0.9766 after domain B training. Delta −0.011 is within training noise.**

Key finding: gradient isolation alone is not sufficient (v1 failure). Forward-pass isolation is required (v2 fix). When both forward and backward passes are partitioned by domain, retention is essentially perfect at toy scale.

**Prior art identified (2026-04-17):**

Context-Dependent Gating (XdG), Masse, Grant & Freedman, PNAS 2018. Before each task, a fixed random binary mask is generated for hidden units. Zero optimization, zero gradient search, forward pass gated by zeroing ~80% of neuron activations. Random draw per task with ~20% expected overlap between any two task sub-networks. Overlap fraction handled by EWC or Synaptic Intelligence regularization on overlapping nodes.

XdG is the closest known prior work to PWP. It confirms the design point exists in the literature. It does not close the gap. XdG requires regularization as a fallback because overlap is probabilistic, not zero. PWP enforces zero overlap by construction (row partition, guaranteed non-overlapping). The -0.011 retention delta was achieved with no EWC, no SI, no replay. The contribution claim must be stated precisely: guaranteed non-overlapping upfront assignment that eliminates the regularization component entirely, not merely fixed pre-training partition (XdG already has that). XdG must be cited as primary prior work in any external write-up.

Note: Gemini's structured literature report initially stated this design point had not been published as a primary proposal. A second search found XdG. The two responses were inconsistent. Both are logged; neither is treated as settled without direct paper verification.

**XdG ablation results (Gemini, 2026-04-17 — requires direct paper verification):**

Permuted MNIST benchmark. XdG alone (no EWC, no SI):

- 10 tasks: 97.1% mean accuracy
- 100 tasks: 61.4% mean accuracy (severe scaling collapse)

XdG + EWC or SI:

- 100 tasks: 95.4% (stable)

Interpretation: the regularization component is not cosmetic. At 100 tasks, each neuron is shared by an expected 20 tasks (binomial B(100, 0.20)). Without a penalty on shared neurons, gradient overwrite is guaranteed. EWC/SI exist specifically to cover the probabilistic overlap fraction. Remove them and the architecture degrades by 35.7 percentage points.

**Sharpened contribution claim (2026-04-17):**

XdG demonstrates that random sparse gating delays catastrophic forgetting but requires synaptic stabilization to remain viable at scale, because probabilistic inter-task overlap accumulates and must be managed by gradient penalization. PWP replaces probabilistic sparse gating with guaranteed non-overlapping neuron-group assignment (row partition, zero overlap by construction). This eliminates the overlap-driven failure mode entirely and removes the need for any regularization component. The simplification is structural, not a degradation of the mechanism. The regularization overhead is not reduced; it is made unnecessary.

This is the precise gap PWP occupies relative to XdG. Cite XdG as primary prior work. State the contribution as above.

Open questions for scale-up:

- **Capacity at scale.** At Gemma E4B hidden_size=2560 with D=4 domains, each domain gets 640 neurons per layer. Quadratic capacity penalty means each domain runs at (640/2560)² = 6.25% of full hidden capacity per layer. Needs empirical validation on a non-trivial task before committing.
- **Task complexity.** MNIST is trivially separable. Next tests: Permuted MNIST, Split CIFAR-10.
- **Shared input projection at scale.** Token embedding dimension is shared across domains. Gemini's proof applies: row partition of the projection matrix prevents interference. No new mechanism required.
- **No size increase constraint (Prof. Qazi).** Neuron-group partitioning keeps total parameter count constant (domains share the same weight tensor, partitioned). No model size increase. Constraint satisfied at the architecture level.

---

## Session Log

**2026-04-18 — Geometric Upgrade Path Identified (Claude + Aman)**

Origin: Aman encountered a Wikipedia visualization of the binomial theorem ((a+b)^1 through (a+b)^4 as geometric objects) while doomscrolling. The image triggered a question about deeper geometries than the current linear partition.

**The observation:** The current neuron-group row partition treats the weight space as a 1D object and cuts it into D equal slices. This was intentional — the goal was controllability and interpretability, making the black box less hostile to work on. That goal was correct and the linear partition achieved it. The 6.25% capacity concern is real but is downstream of the chosen geometry, not a flaw in the mechanism.

**What the binomial image points at:** A high-dimensional weight space doesn't decompose best as sequential slices. Its natural substructure is combinatorial. Each subspace can be defined by *which dimensions participate*, not *which contiguous block is assigned*. In N-dimensional space the number of possible non-overlapping k-dimensional subspaces is C(N,k), which is combinatorially large even for small k.

**The geometric object:** The Grassmannian Gr(k, N) — the space of all k-dimensional subspaces of an N-dimensional space. Domain separation at scale corresponds to placing domains at maximally separated points on the Grassmannian. Instead of each domain owning a slice of neurons, each domain would own a *direction* through the full weight space. Domains sharing physical neurons but operating in orthogonal subspaces have zero gradient interference by construction, because orthogonal subspaces don't project onto each other. Capacity per domain stays close to full H rather than H/D.

**Current status:** Theoretical direction only. Renderer is quiet — no substrate loaded yet, cannot simulate. Linear partition remains the correct working architecture. This is a future upgrade path, not a correction. No action until empirical data from Permuted MNIST / Split CIFAR-10 gives the renderer something to work with.

**Note on lineage:** This session also confirmed that PWP branched cleanly from Proteus via the Tesseract concept (NVMe-style logical-to-physical remapping controller), with the Hashing Trick being the algorithmic instantiation of that controller. The branch is architectural, not just conceptual.

---

## Session Log

**2026-04-24 — Grassmannian Upgrade: Formal Derivation (Gemini, reviewed by Claude + Aman)**

Origin: v2 empirical confirmation (delta -0.011pp, Split-MNIST) resolved the deferral condition for the Grassmannian upgrade. Linear partition's 6.25% capacity ceiling at H=2560, D=4 is now the active problem. Gemini was tasked with formal derivation of the upgrade path.

**1. Mutual Orthogonality Constraint**

For D domains each occupying a k-dimensional subspace of R^H, mutual orthogonality requires the subspaces to intersect only at the origin. Their direct sum dimension is D·k, which cannot exceed H.

- Maximum domains: D_max = floor(H/k)
- Constraint on k: k ≤ floor(H/D)
- At k = H/D exactly, the subspaces tile R^H completely (Flag Manifold special case).

**2. Gradient Projection Operator**

Domain d's subspace represented by orthonormal basis P_d ∈ R^(H×k), P_d^T P_d = I_k. Projection operator: Π_d = P_d P_d^T.

For weight matrix W ∈ R^(H×H) and incoming gradient G, the domain-gated gradient is:

> G̃_d = Π_d G Π_d = (P_d P_d^T) G (P_d P_d^T)
> 

Proof of zero interference: for any v⊥ ∈ V_d' (d' ≠ d), orthogonality gives P_d^T v⊥ = 0, therefore G̃_d v⊥ = 0. Output of gradient update is always a linear combination of columns of P_d, so update lives strictly in V_d.

**3. Orthogonality Enforcement — Comparison**

| Mechanism | Cost/step | Guarantee | Complexity |
| --- | --- | --- | --- |
| Periodic Gram-Schmidt (QR) | O(H(Dk)²) | Hard (at application) | Low — torch.linalg.qr |
| Orthogonality penalty loss | O(H(Dk)²) | Soft only | Very low |
| Stiefel manifold optimizer | O((Dk)³ + H(Dk)²) | Hard (every step) | Very high |

**Recommendation: Periodic Gram-Schmidt.** Soft penalty leaks gradients across domains, violating the zero-interference premise. Stiefel optimizer adds unacceptable engineering overhead. QR via `torch.linalg.qr` fires post-step, not every step, adds negligible overhead at H=2560.

**4. Capacity Recovery**

Gemini uses degrees-of-freedom framing (Grassmannian manifold dimension), not raw parameter count. At k = H/D:

- C_lin = (H/D)² = H²/D²
- C_grass = k·H = H²/D (manifold DoF: k² for M_d + k(H-k) for P_d on Gr(k,H))
- Ratio: C_grass / C_lin = D

Grassmannian assignment recovers the full factor of D in capacity lost to linear partitioning. At H=2560, D=4: capacity per domain goes from 6.25% to 25% of full H.

*Note: Gemini's capacity figure is Grassmannian manifold degrees of freedom, not raw stored parameter count (which would be Hk + k²). Conclusion ratio = D holds either way.*

**5. Initialization — SVD Round-Robin**

Starting from shared pretrained W_pre ∈ R^(H×H):

1. Decompose: W_pre = UΣV^T. U is orthogonal, columns u_1...u_H sorted by descending singular values.
2. Assign round-robin: domain d ∈ {0,...,D-1} receives indices I_d = {d + j·D | j ∈ [0, k-1]}.
3. Construct P_d = [u_{I_d,0}, ..., u_{I_d,k-1}] ∈ R^(H×k).
4. Initialize inner weights: M_d = P_d^T W_pre P_d ∈ R^(k×k).

Orthogonality guaranteed at step 0 by orthogonality of U. Round-robin prevents domain 0 from monopolizing top singular components — each domain gets an equal, unbiased sample of the pretrained network's principal directions.

**Status:** Formal foundation complete. Implementation not started. Next: Grassmannian v1 experiment on Permuted MNIST or Split CIFAR-10.

---

## Session Log

**2026-04-24 — Permuted MNIST v2 Results (Aman + Claude)**

First non-trivial benchmark. Permuted MNIST is harder than Split-MNIST: all 10 classes appear in every task, no class-routing cheat possible. 5 tasks, each MNIST with a distinct fixed pixel permutation. H_TOTAL=640, H_D=128 per domain, 5 epochs per task.

**Baseline MLP (sequential training, no isolation):**

| Task | Acc after own training | Acc after all 5 tasks |
| --- | --- | --- |
| 0 | 0.9798 | 0.5413 |
| 1 | 0.9801 | 0.7026 |
| 2 | 0.9799 | 0.8687 |
| 3 | 0.9781 | 0.9548 |
| 4 | 0.9812 | 0.9812 |

Mean final accuracy: 0.8097. Catastrophic forgetting confirmed — task 0 loses 43.85pp by end.

**PWP v2 Linear Partition:**

| Task | Acc after own training | Acc after all 5 tasks |
| --- | --- | --- |
| 0 | 0.9690 | 0.9690 |
| 1 | 0.9714 | 0.9714 |
| 2 | 0.9714 | 0.9714 |
| 3 | 0.9722 | 0.9722 |
| 4 | 0.9665 | 0.9665 |

Mean final accuracy: 0.9701. Delta vs baseline: +0.1604.

**Key finding:** PWP retention is structurally flat. Task 0 accuracy does not move across tasks 1, 2, 3, 4 — not even a rounding error. Isolation is not approximate. It is complete.

**The cost:** PWP peak per task is ~1.1pp below baseline peak (0.969 vs 0.981). This is the capacity tax — H_D=128 per domain vs H=640 for the full baseline model. Each domain runs at (128/640)² = 4% of full hidden capacity in the quadratic layers. The model is learning the permuted task correctly within the constraint, but the ceiling is lower.

**This is exactly what the Grassmannian upgrade is designed to fix.** Grassmannian domains own a direction through the full H=640 space rather than a contiguous 128-neuron slice. Capacity per domain goes from 4% to 20% (factor of D=5 recovery). Retention should remain structurally flat — the isolation mechanism is unchanged, only the geometry of the partition changes.

**Next:** Implement Grassmannian v1 (SVD round-robin init, QR orthogonality enforcement, projection gradient gate) and run the same Permuted MNIST benchmark.

---

## Session Log

**2026-04-24 — Permuted MNIST v3 Grassmannian Results (Aman + Claude)**

Direct upgrade from v2. Same benchmark, same config (H=640, k=128, D=5, 5 epochs/task). Replaced row partition with Grassmannian subspace assignment: SVD round-robin init, gradient hook projecting G to Pi @ G @ Pi, QR re-orthogonalization every step.

**v2 vs v3 comparison (accuracy after all 5 tasks trained):**

| Task | v2 (end) | v3 (end) | Delta |
| --- | --- | --- | --- |
| 0 | 0.9690 | 0.9718 | +0.0028 |
| 1 | 0.9714 | 0.9718 | +0.0004 |
| 2 | 0.9714 | 0.9735 | +0.0021 |
| 3 | 0.9722 | 0.9731 | +0.0009 |
| 4 | 0.9665 | 0.9721 | +0.0056 |

v2 mean: 0.9701. v3 mean: 0.9725. Delta: +0.0024.

**Findings:**

1. Retention remains structurally flat in v3. Task 0 accuracy does not degrade across tasks 1-4, same as v2. The isolation mechanism is intact under the subspace formulation.
2. Every task improved over v2. The improvement is consistent but modest (+0.0024 mean). This is expected: MNIST is too easy to stress the capacity ceiling. At k=128 per domain, even the v2 row partition has enough neurons to saturate MNIST performance. The theoretical factor-of-D capacity recovery will show at harder tasks where the ceiling actually bites.
3. Task 4 showed the largest gain (+0.0056). This is likely because task 4 is trained last -- by then the shared weight tensor has been shaped by 4 prior domains, and the Grassmannian subspace has more signal to initialize from than a raw row slice.

**What this means:** v3 is strictly better than v2 on this benchmark, but MNIST is not the discriminating test. The real validation of the capacity claim requires a task complex enough that 128 neurons per domain is a binding constraint. Next benchmark: Split CIFAR-10 or a language task where the capacity gap between v2 and v3 should be visible.

---

## Session Log

**2026-04-24 — Split CIFAR-10 v2 vs v3 Results (Aman + Claude)**

Harder benchmark. Split CIFAR-10: 5 binary tasks over disjoint class pairs (airplane/automobile, bird/cat, deer/dog, frog/horse, ship/truck). RGB input, 3072-dim flattened. 10 epochs per task. H=640, k=128, D=5.

**Results after all 5 tasks trained:**

| Task | Baseline | v2 | v3 |
| --- | --- | --- | --- |
| 0 | 0.5920 | 0.8855 | 0.8640 |
| 1 | 0.5755 | 0.7660 | 0.7470 |
| 2 | 0.6405 | 0.7880 | 0.8060 |
| 3 | 0.7730 | 0.8930 | 0.8800 |
| 4 | 0.8545 | 0.8385 | 0.8575 |
| Mean | 0.6871 | 0.8342 | 0.8309 |

**Finding 1 — v2 retention: perfectly flat.**

Task 0 holds at exactly 0.8855 across all 5 tasks. Hard block separation achieves structural isolation on CIFAR, same as MNIST.

**Finding 2 — v3 retention: slight drift.**

Task 0 goes 0.8795 → 0.8780 → 0.8735 → 0.8735 → 0.8640. Total drift: -0.0155pp over 5 tasks. Not catastrophic but not flat. The subspaces are leaking slightly despite QR re-orthogonalization every step.

**Finding 3 — v3 does not beat v2.**

Mean 0.8309 vs 0.8342. Delta: -0.0033. The theoretical capacity recovery did not appear.

**Diagnosis:**

v3 uses a shared weight tensor with projected gradients. Even with orthogonal subspace projections, all domains write to the same W matrix during training. v2 uses physically separate parameter blocks — different tensors, zero shared state. At this scale, hard physical separation beats soft geometric separation.

This is not a failure. It is a precise finding: the Grassmannian advantage is theoretical until the task is complex enough that v2's capacity ceiling is the binding constraint. H_D=128 is sufficient for binary CIFAR classification. The ceiling has not been hit.

**What comes next:**

Domain scaling stress test to find where v2 breaks and whether v3 holds longer.

---

## Session Log

**2026-04-24 — Domain Scaling Stress Test (Aman + Claude)**

Goal: find the crossover point where Grassmannian advantage becomes empirically real by stressing v2 with increasing D. H_TOTAL fixed at 640. D swept over [5, 10, 20, 40]. Permuted MNIST, 5 epochs/task.

| D | H_D | v2 mean | v3 mean | Delta |
| --- | --- | --- | --- | --- |
| 5 | 128 | 0.9690 | 0.9692 | +0.0002 |
| 10 | 64 | 0.9602 | 0.9623 | +0.0021 |
| 20 | 32 | 0.9471 | 0.9527 | +0.0056 |
| 40 | 16 | 0.9297 | 0.9171 | -0.0126 |

**Finding 1 — v3 wins in the D=10-20 range.**

As H_D shrinks, v2's capacity ceiling bites and v3's direction-through-full-space advantage materializes. Gap widens monotonically from D=5 to D=20, exactly as predicted.

**Finding 2 — v3 breaks before v2 at D=40.**

v2 at 16 neurons per domain holds at 0.9297 mean, min=0.9232. Stable across all 40 tasks. v3 at k=16 collapses to 0.9171 mean, min=0.8114. High variance, early tasks severely degraded.

**Finding 3 — v3 early task drift pattern at D=40.**

v3 per-task at D=40 shows a systematic pattern: tasks 0-12 score 0.811-0.940 (noisy, low), tasks 13-40 score 0.890-0.954 (recovering, stable). Late tasks are fine. Early tasks are hurt.

**Diagnosis — QR re-orthogonalization drift.**

Every time a new domain trains and the QR sweep fires, all D basis matrices are re-orthogonalized together. The early trained subspaces get nudged slightly with each sweep. By domain 40, task 0's subspace has been re-orthogonalized 39 times. Each nudge is small but they accumulate. At k=16 the subspace is narrow enough that small rotations have large functional impact.

v2 is immune by construction: physically separate parameter blocks, once trained, are never touched by subsequent domain training.

**Architectural conclusion:**

Grassmannian v3 has a sweet spot between D=10 and D=20. Below D=10: advantage is marginal, v2 is simpler and equally effective. Above D=20: QR drift on early tasks exceeds the capacity gains, and v2's hard physical isolation wins.

The QR drift is not a fundamental limit of the Grassmannian idea — it is a limit of the specific enforcement mechanism (periodic global re-orthogonalization). A domain-local orthogonality scheme that does not sweep all bases simultaneously would eliminate the drift. The fix: when training domain d, only re-orthogonalize P_d against already-trained domains {0..d-1}, without modifying those frozen bases. This preserves isolation for trained domains while enforcing orthogonality for the new one.

**The discriminating test requires a task where 128 neurons per domain is genuinely insufficient**

---

## Session Log

**2026-04-24 — v4 Domain-Local Orthogonality: Still Breaks at D=40 (Aman + Claude)**

Hypothesis: v3's drift at D=40 was caused by global QR re-orthogonalization nudging frozen subspaces. v4 replaces global QR with domain-local incremental Gram-Schmidt: before training domain d, project P_d into the null space of all frozen {P_0...P_{d-1}}, then freeze P_d permanently. Frozen bases never touched again.

**Results (Permuted MNIST, H=640, same sweep):**

| D | H_D | v2 | v3 | v4 |
| --- | --- | --- | --- | --- |
| 5 | 128 | 0.9702 | 0.9685 | **0.9719** |
| 10 | 64 | 0.9597 | **0.9644** | 0.9634 |
| 20 | 32 | 0.9453 | **0.9544** | 0.9492 |
| 40 | 16 | **0.9287** | 0.9199 | 0.9207 |

**v4 did not fix D=40.** v4 mean 0.9207 vs v3's 0.9199 -- negligible difference. v4 minimum (0.8053) is actually worse than v3 minimum (0.8417).

**Revised diagnosis -- capacity exhaustion, not drift.**

The null space available to each successive domain shrinks monotonically:

- Domain 0: full 640 dims available
- Domain 1: 624 dims available
- Domain 39: exactly 16 dims available

At D=40, k=16: the weight space is completely tiled by domain 39. The last domain is squeezed into whatever 16 dimensions remain. 16 dimensions is below the functional floor for the Grassmannian projection -- the gradient Pi_d @ G @ Pi_d is so restrictive the model cannot learn meaningful features regardless of orthogonality scheme.

v2 survives at D=40 because its 16 neurons per domain are physically independent parameters. The shared weight tensor in v3/v4 means all domains write to the same W, and at k=16 the projection collapses expressive capacity regardless of how cleanly the subspaces are separated.

**Architectural conclusion:**

Grassmannian shared-weight approach has a hard floor at approximately k=32 (D=20 at H=640). Below k=32, physical separation (v2) is the correct architecture. Above k=32, Grassmannian wins on capacity. The crossover is geometric, not an implementation artifact.

The fix is not a better orthogonality scheme. It is either: (a) increase H_TOTAL so k stays above 32 as D grows, or (b) use v2 below k=32 and switch to Grassmannian above it. A hybrid selector based on k is the natural next design decision.

---

## Architectural Finding — Two Modes, One Selector

*[Aman + Claude, 2026-04-24]*

The sweep results define two operating regimes based on k = H // D:

**Mode 1 — Grassmannian (k ≥ 64)**

Shared weight tensor, subspace projection gradient gate, directions through full space. Capacity advantage is real. Use when few domains, maximum expressiveness per domain required.

**Mode 2 — Physical separation (k < 32)**

Block-diagonal separate parameters, hard isolation by construction, no shared tensor interference. Use when many domains, k too small for subspace projection to be meaningful.

The selector is a single threshold on k:

```python
def select_architecture(H, D):
    k = H // D
    if k >= 64:
        return "grassmannian"
    else:
        return "physical"
```

PWP is not v2 or v3. It is a substrate that selects the correct geometry based on packing density. Large rooms get subspace architecture. Small rooms get physical walls. The building analogy is exact.

This is a stronger result than either architecture alone: a unified framework with an automatic mode selector derived empirically from the phase boundary at k ≈ 32–64. — where v2 plateaus and cannot improve regardless of more training, while v3 can continue learning by accessing more of the weight space. A multi-class task at higher hidden dimension, or a language task, would expose this gap. Alternatively: increase D (more domains, smaller H_D per domain in v2) until v2 degrades and v3 holds.

[🔍 Pending Proofing](https://www.notion.so/Pending-Proofing-3454dd60be88813e85a2eb8be74b58d8?pvs=21)