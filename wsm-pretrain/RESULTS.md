# Results: exact tools inside a model's forward pass

One day of experiments (26 November 2026). Unrelated to the rest of this repository.

## Questions

1. Can a model learn to use exact tools that live inside its forward pass?
2. Is it better to train with the tools from the start or to bolt them onto a trained model?
3. Does a small learned interface match a hand-built in-weights calculator on a real model?

## Design

- **Bus** (`bus/`, `bus2/`): every position holds one exact value register. Two trainable pointer heads pick
  earlier positions, a trainable gate picks a tool, and frozen multilinear tools compute exactly
  (copy, add, sub, mul, max, lt, eq, inc on values 0..96). Tool outputs feed other tools through the registers.
- **Toy task**: short straight-line programs whose intermediate results are sometimes hidden, so values must be
  carried internally. `bus2` adds counting (chains of increments) and a branch on a computed value.
- **Qwen bus** (`qbus/`): a 0.58M-parameter reader on a frozen Qwen3-0.6B-Base reads one hidden layer and outputs a
  gate and a pointer. The expression is then read exactly from the token text, evaluated with the paper's own
  exact arithmetic (`calc_ref`), and the answer's tokens are forced. The host's weights never change.

## Toy results (12,000 steps, 3 repeats each; mean and range)

Four tools (`results/bus-v1`):

| Version | Hidden multi-step chains | Never-trained operations | Bus switched off |
|---|---|---|---|
| No tools | 43.0% (42.5-43.3) | 55.9% | n/a |
| Tools, guidance always on | 99.9% | 100% | 1.4% |
| Tools, guidance faded out by 30% of training | 100% | 100% | 0.7% |
| Tools, no guidance | 97.9% (97.6-98.2) | 99.9% | 1.2% |
| Faded + looped twice | 100% | 100% | 1.1% |
| Faded + attention residuals | 100% | 100% | 0.0% |
| Faded + looped + attention residuals | 98.8% (96.3-100) | 99.7% | 0.0% |

Eight tools with counting and branching (`results/bus-v2`):

| Version | Hidden chains | Counting | Branching |
|---|---|---|---|
| No tools | 59.1% | 54.5% | 75.3% |
| Faded guidance, one pass | 97.5% | 99.9% | 96.8% |
| Faded guidance, two separate passes | 99.9% | 99.8% | 99.9% |
| Faded guidance, looped twice | 99.6% | 98.9% | 99.9% |
| Faded guidance, looped + attention residuals | 99.8% | 99.9% | 99.9% |
| No guidance, one pass | 94.7% (93.6-95.5) | 98.8% | 99.8% |
| No guidance, looped twice | 97.2% (95.8-97.9) | 99.3% | 99.9% |

- Models trained with the tools use them almost perfectly, and depend on them (accuracy collapses with the bus off).
- They find the tools without any routing guidance, though guidance is a little better on long chains.
- Unguided models wire themselves differently from the designed routes (pointer agreement about 33%): they fire a
  tool at nearly every position and relay values in short hops.
- A second pass helps branching for guided models (96.8% to 99.9%). Reusing the same weights (looping) and using
  two separate passes do equally well. Attention residuals neither help nor hurt here.
- **Bolting the bus onto a frozen toy model did not work** (`results/bolt`, `results/x1`): 58-61% on hidden chains
  against 56% without tools, with pointers only 43-70% correct even under guidance.

## Qwen3-0.6B-Base results (frozen host, same machine, 1,319 problems each)

| Setup | GSM-Hard | GSM8K |
|---|---|---|
| No tools | 25.78% (340) | 52.39% (691) |
| Paper's calculator implant (reproduced) | 34.04% (449) | 53.37% (704) |
| Our bus, two-number calls only | 29.42% (388) | 50.19% (662) |
| **Our bus, whole expressions** | **35.03% (462)** | **54.13% (714)** |

The paper reports 27.29 to 35.03 on GSM-Hard and 52.24 to 53.37 on GSM8K for its calculator.

- The learned reader gets 99.4% of calls fully right on held-out text, with 0.3 false fires per 1,000 tokens. It
  trains in about two minutes on 4,700 examples.
- Layer 12 of 28 is the best place to read; layers 6 and 18 are close, layer 24 is worse. Giving the reader its own
  attention layers did not help.
- The first version handled only "a op b =" and lost to the paper's calculator because of chained expressions
  (57 of the 72 problems it lost). It also fired on the last two numbers of a chain, which hurt ordinary GSM8K.
  Teaching it whole expressions was a change of training labels plus about 30 lines of code.
- Arithmetic written in the answers is 96.4% correct on GSM-Hard with our bus (97.6% with the paper's calculator,
  78.6% with no tools).

## Limits

- Single runs on Qwen: leads of 10-13 problems out of 1,319 are within noise. The fair reading is "matches".
- Our write side forces tokens in the generation loop; the paper's implant writes through weights.
- Results written after a dollar sign are our weak spot (86-91% correct against 98-99% otherwise).
- Every toy win is on a task designed around the tools. Nothing here tests whether in-pass tools make a model
  generally more capable, which was the original motivation.
- No real-text pretraining run was done.

## Cost

About $8 of RunPod GPU time and under $1 on Lambda.

## Reproduce

`rp.py` drives a RunPod GPU over HTTPS. Toy sweeps: `python bus/sweep.py 12000 7 results/bus-v1`,
`python bus2/sweep.py 12000 3 results/bus-v2`. Qwen: clone `anadim/weight-space-micromachines`, install its
requirements, then from its root run `qbus/build_data.py`, `qbus/train_reader.py` and `qbus/run_gsm_bus.py`
(see `results/qwen/summary.txt` for the raw result lines).
