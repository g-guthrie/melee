# Pretraining with a frozen in-weights tool (pilot)

Unrelated to the rest of this repository; parked on this branch until it has its own repo.

**Question.** If an exact, frozen tool sits inside a transformer from the first step of
training, does gradient descent alone teach the model to route to it?

**Setup.** `train.py` trains a small causal transformer from scratch on a mix of two tasks:

- *tool task*: bind K names to values, then compute `f(v_i, v_j)` for two named values;
- *filler task*: memorise a table of random facts (competes for capacity).

The tool is a parameter-free multilinear lookup of `f`, exact on one-hot inputs and
differentiable on soft ones. It reads two operand distributions from the residual stream
through trainable linear readers and writes its answer back as a token embedding.
20% of `(a, b)` pairs are never queried in training.

**Arms.** `none` (baseline), `tool` (no supervision of the interface), `tool_aux`
(readers are told the operands), `mlp` (trainable MLP in the tool's slot), `oracle`
(tool is fed the true operands).

**Readouts, fixed before running.**

| Readout | Meaning | Pass |
|---|---|---|
| held-out pair accuracy, `tool` arm | the model found and uses the tool unaided | > 90%, with `none` and `mlp` near chance on the random table |
| reader accuracy | readers decode the true operands | > 90% |
| accuracy with the tool ablated | the model depends on the tool | collapses on held-out pairs |
| filler accuracy vs `none` at equal steps | capacity freed by offloading | higher in tool arms, larger gap in the narrow model |

Kill criterion: if `tool_aux` cannot reach `oracle` accuracy, the interface is broken and
nothing else is interpretable.

**Run.** `python sweep.py --name pilot --workers 8` (one GPU), results in `results/`.
