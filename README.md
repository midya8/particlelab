# ParticleLab

**Full weight ESS does not mean recovered uncertainty.** A bootstrap particle filter from scratch, with log-domain weights, systematic resampling, complete weighted/carried particle histories, initial-ancestry audits and an exact scalar Kalman reference. Dependency-free Python 3.10+; no paid API, GPU or data download.

![Particle filtering and finite-support failure](assets/particle-audit.svg)

The static example ends with weight ESS 32 out of 32, one initial ancestor and zero particle variance. The exact posterior still has variance .000499972. Resampling resets weights, not missing state support. A no-resampling comparison exposes the same poor initial finite support; this is not a claim that resampling alone causes the error.

## Run locally

From the repository root:

```sh
python -m unittest discover -s tests -v
python -m particlelab demo > /tmp/particle-demo.json
python examples/render_demo.py /tmp/particle-demo.json --check
python -m particlelab filter examples/sequence.json --particles 512 --seed 0
python -m particlelab filter examples/sequence.json --ess-fraction 0
python -m particlelab kalman examples/sequence.json
python -m particlelab filter --help
```

Every model command reads exactly these six JSON fields:

```json
{
  "data": [0.1, 0.3, -0.2, 0.4, 0.7, 0.9, 0.2, -0.3, -0.6, -0.4, 0.1, 0.2],
  "a": 0.9,
  "process_std": 0.4,
  "observation_std": 0.5,
  "initial_mean": 0.0,
  "initial_std": 2.0
}
```

`filter` also accepts `--observation square`, meaning h(x)=x² rather than x. `kalman` is only the linear identity-observation reference. The supplied linear fixture is not a square-model validation study. `demo` needs no file. Output is JSON; invalid inputs exit 2 with a prefixed stderr error and no traceback. Python APIs are `filter_particles`, `kalman` and `systematic_resample` from `particlelab`.

## Explicit model and timing

- Initial x₀ ~ Normal(initial_mean, initial_std²). The first observation updates this distribution directly: **there is no initial transition**.
- For t>0, `xₜ = a*xₜ₋₁ + Normal(0, process_std²)`, with independent process noise. Process std may be zero.
- yₜ | xₜ ~ Normal(h(xₜ), observation_std²), with supplied fixed noise. h is identity or square.
- The transition is the proposal. Weight each propagated particle by its observation likelihood and incoming normalized weight. The log normalizer is the next evidence increment.
- Normalize in log space. Retain finite normalized log weights across skipped resampling even if their stored probabilities underflow to zero. A later observation can revive such a particle's relative mass.
- If weight `ESS < N*ess_fraction`, resample with one systematic offset and reset carried weights to 1/N. The inequality is strict; fraction 0 disables resampling. Finite rounding can matter near the threshold.

This is filtering p(xₜ | y₀:ₜ), not smoothing p(xₜ | y₀:T). Nothing learns parameters, proposes guided particles, rejuvenates states or corrects an observation model.

## Inspect the output, not one score

Each time step retains filtered particles, probability/log weights, incoming probability/log weights, initial-root labels, filtered mean/variance and ESS, the resampling decision and parent indices, carried particles/weights/root labels, carried moments/ESS, unique initial ancestors, distinct carried state values and evidence increments. Filtered summaries are **before** resampling; carried summaries are **after**. Resampling can change moments by sampling noise. Ancestry counts track labels, not the number of currently distinct state values. Positive process noise can create distinct values without recovering lost initial ancestors.

Systematic resampling uses points `(j+offset)/N`, offset uniform in [0,1). CDF bins are right-open: a point exactly on a boundary skips that bin, including zero-mass bins. Particles are dependent after resampling; they are not independent posterior draws. Float CDF roundoff is guarded at the tail, but the implementation is not exact rational arithmetic.

The accumulated evidence estimate is random. Its log is not an unbiased log-evidence estimate. Log-evidence errors here are finite-run diagnostics, not a likelihood-estimator theorem or a Bayes-factor reliability claim. `ESS = 1/sum(w²)` measures weight concentration only. Neither ESS nor ancestry count proves convergence, accuracy or calibrated uncertainty.

## Reproducible development examples

### Linear filtering, with exact reference

The twelve observations in the example payload use a=.9, process std=.4, observation std=.5, initial mean=0/std=2. `demo` runs 512 particles for **all sixteen seeds 0..15**. The figure shows seed 0, not a run selected for fit. The mean of per-run time-averaged mean RMSEs is **.0161016904**; the corresponding variance RMSE average is **.00794734279**. Mean log-evidence error across those runs is **+.00876794367**. Complete per-seed errors are retained. These are different summaries, not pooled RMSE or a confidence interval.

The sequence was fixed during implementation. The .1 mean-RMSE threshold in the demo is a development smoke check, not a convergence guarantee, particle-budget recommendation or held-out benchmark. No real data is used. The exact reference uses scalar Gaussian conditioning and the supplied model, not the true quality of that model.

### Static-support failure, including a no-resampling comparison

Twenty zero observations, a=1, process std=0, observation std=.1, initial mean=0/std=3, 32 particles, seed 0. This was the first seed inspected; additional exploratory seeds 1/7/19/23 also ended with one initial ancestor under resampling. With Q=0 the state support comes entirely from the finite initial sample. Reweighting cannot create a particle at the true posterior center.

| Final quantity | ESS threshold fraction 1 | Resampling disabled | Exact Kalman |
|---|---:|---:|---:|
| Mean | -.2163600684 | -.2163533277 | 0 |
| Latent variance | 0 | .00000308085 | .000499972224 |
| Weight ESS | 32 | 1.00002950 | not applicable |
| Initial labels still carried | 1 | 32 | not applicable |

The no-resampling run retains all initial labels but almost all weight sits on one. The resampling run has equal weights over duplicate descendants. Both have poor effective state support. This comparison does not isolate a general causal effect of resampling, and it says nothing about a different proposal, rejuvenation move, particle budget or process model.

The four-panel SVG shows all 12 linear and all 20 static points, with fixed numeric domains. Its shaded mean +/- one-standard-deviation summaries are not coverage bands. All plotted values, model settings, seed errors and domains are embedded in SVG metadata. Regeneration fails on changed fixtures or out-of-domain data instead of clipping it silently.

## Validation and bounds

**38 tests** cover one-step hand weights/moments/evidence, no-resampling evidence against full products over the initial sample, recovery of relative log mass after probability underflow, exact resampling boundary/zero-bin/offspring audits, RNG isolation, parent/root mappings, state-diversity versus ancestry, closed-form Gaussian/static-sequence references, extreme precision, strict CLI payloads, both success/failure demo checks and all plotted coordinates/values/SD polygon vertices. The expected-offspring test integrates a fixed rational-weight fixture over a systematic-offset grid; it is not a proof for all floats. CI runs Python 3.10..3.13 plus the demo and byte-identical SVG check.

Inputs: 1..80 finite observations in [-1e4,1e4]; a in [-1,1]; process std in [0,100]; observation/initial std in [1e-4,100]; initial mean in [-1e4,1e4]. Particles: integer 1..4096; `N*T <= 65536`. Seed: integer `0..2**32-1`. ESS fraction: [0,1]. Booleans are rejected as numbers. Full history uses `O(T*N)` time/output/memory, including `O(N)` systematic resampling, plus `O(T)` for the Kalman reference. The demo runs sixteen additional particle filters. All quantities use floating point and a finite local RNG. Tiny weights can underflow; finite logs survive only before resampling. Random draws are reproducible in the tested environment, not a cross-language bitstream contract.

The square observation branch has hand-checked even-likelihood and propagation tests, not a nonlinear tracking benchmark. There is no smoothing, parameter learning, high-dimensional proposal, robustness claim, convergence certificate, anomaly test, real-world accuracy or coverage guarantee. No external service receives the local input.

## Attribution

Independent implementation inspired by Gordon, Salmond and Smith, **"Novel approach to nonlinear/non-Gaussian Bayesian state estimation" (1993)**, especially Section 3's transition propagation, likelihood weighting and bootstrap resampling. [Primary paper](https://people.bordeaux.inria.fr/pierre.delmoral/gordon-salmond-smith-1993.pdf). The paper also discusses loss of distinct samples. This project uses systematic/adaptive ESS resampling, scalar linear/square observation models and its own synthetic fixtures; it does **not** reproduce the paper's bearings-only experiments, roughening, prior editing or performance comparisons. No code or paper figures were copied.

MIT license.
