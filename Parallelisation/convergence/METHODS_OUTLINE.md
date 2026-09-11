# Methods Outline — Par-4 Approach-Strategy Simulation, Birdie Model, and Convergence Study

Grounded directly in the current implementation: `Parallelisation/convergence/core.py`
(ESHO scheme), `Parallelisation/convergence_birdie/core_birdie.py` (birdie scheme),
`Parallelisation/convergence/equivalence.py` / `convergence_worker.py` (convergence
scheme), and the original research notebook `PART 3 / Simulating Par 4/scriptpar4.py`
(source of the tee-shot layer, not yet ported into `core.py`). Every parameter value
below is read from the code, not recalled from memory — flagged where the two
schemes (ESHO vs. birdie) have drifted apart, and where a stage exists in the
notebook but not in the productionised pipeline.

---

## PART 1 — Simulation / Optimisation Mechanism (Tee to Green)

### 1.1 Pipeline overview

```
hole geometry (data) --> coordinate alignment --> strategy grid
club dispersion models (data) --> per-club (Side, Carry) ~ N(mu, Sigma)
putting green data --> GPR #1: E[putts | x, y on green]     (ESHO scheme)
                    --> GPR #1': P(1-putt | x, y on green)   (birdie scheme)
Broadie baseline table --> off-green expected-strokes lookup

for each grid point (candidate approach-shot origin):
  for each candidate club x aim-offset:
    Monte Carlo shots --> landing points --> lie classification -->
      ESHO (or birdie prob) per shot --> per-combo mean/var
  argmin ESHO (or argmax P(birdie)) over combos --> optimal_results[grid point]

[notebook only, not in core.py] GPR #2 fit on optimal_results surface -->
  tee-shot evaluation: club x aim -> predicted ESHO of landing spot -> best tee play
```

### 1.2 Course geometry and the coordinate system

**Data sources** (`Parallelisation/data/`):
- `hole_9_data.csv` — WKT polygons for green / bunker / water / fairway / rough / tee, tagged by `lie`.
- `newshapes.geojson` — a second, independently-digitised fairway and water-hazard shape (`water_hazard_3`), used to extend/patch the hole beyond the original digitisation.

**Alignment pipeline** (`build_hole`, `core.py:100-235`) — a fixed sequence of shapely
affine transforms, each with a specific anchor and angle (this is the "transformation
matrix" layer of the pipeline; all rotations are 2D rotation matrices `R(θ)` applied via
`shapely.affinity.rotate`, all shifts are translation vectors via `shapely.affinity.translate`):

1. **Tee point**: centroid of the tee-box polygon *farthest* (Euclidean) from the green centroid — `tee_point = argmax_{teebox} ||centroid(teebox) - centroid(green)||`.
2. **New-geometry pass 1** (`new_fairway`, `new_hazard3`, from the GeoJSON): translate so the fairway centroid lands at `(0, 100)`; rotate by `-θ` where `θ = atan2(Δx, Δy)` of the fairway's own long axis (its bounding-box top vs. bottom at the centroid x) — this straightens the new fairway to run along +y before any further placement.
3. **Original hole shapes** (`hole_9`): translate green/bunker/water/fairway by `+160` in y (`y_shift_base = 160`). Pin location fixed at `hole_pin = (5, 174 + 160) = (5, 334)`.
4. **New-fairway final placement**: translate centroid to `(20, 175)`, rotate `-68°`.
5. **New-hazard3 final placement**: translate centroid to `(0, 210)`, rotate `+110°`.
6. Polygon lists assembled: `green_polygon` (1), `bunker_polygons`, `rough_polygons`, `fairway_polygons = hole_9 fairway ∪ new_fairway`, `water_polygons = hole_9 water_hazard ∪ new_hazard3`.
7. **[New, this session] OB + water-flush touch-up** (`core.py:212-227`): OB thresholds fixed at `x < -40`, `x > +60`, `y > max(bunker.bounds.ymax) + 10` (≈ y = 393.4, i.e. ~10 yd past the back edge of the bunkers behind the green). The rightmost water polygon is unioned with a rectangle `box(x_max-5, y_min-5, 60, y_max+5)` so its right edge sits flush against `x = +60` with no playable gap.

**Strategy grid** (`core.py:227-233`): `ht_length = ||pin - tee||`; `x ∈ linspace(-40, 60, 10)` (10 columns, ~11.1 yd pitch), `y ∈ linspace(50, ht_length-50, (ht_length-50)/10)` (~8.6 yd pitch) → **280 candidate approach-shot origins**, i.e. every (club, aim) decision the model searches over is anchored to one of these 280 (x, y) points.

### 1.3 Player shot-dispersion model

**Source**: `simulated_lpga_shot_data2.csv`, grouped by `Club`.

- Per club: `mu = mean(Side, Carry)`, `Sigma = cov(Side, Carry)` (empirical 2×2 covariance) — i.e. each club's shot outcome is modelled as **bivariate normal** `(Side, Carry) ~ N(mu_club, Sigma_club)` in *club-local* coordinates (Side = lateral miss, Carry = distance).
- **Correlation-sign fix** (`core.py:261-266`): the off-diagonal of `Sigma` is forced positive (`|Sigma_01|`) because `rotation_translator` negates the x-side term for this (northward) hole orientation — without the fix, left/right misses would map to the wrong carry-distance correlation.
- **Tunable global perturbations** (`build_hole` args, used for sensitivity analysis, not the convergence study itself): `carry_shift_yards` (additive, shifts every club's mean carry), `variance_scale` (multiplicative, scales every covariance matrix).
- **Rough-lie adjustment** (`core.py:271-286`): clubs ranked by mean carry; interpolation parameter `t = rank / (n_clubs - 1) ∈ [0,1]`. `carry_loss = 0.05 + t*(0.17-0.05)` (5–17% carry reduction, longer clubs lose more from rough), `var_increase = 0.10 + t*(0.40-0.10)` (10–40% variance increase). Applied as `mean_carry *= (1-carry_loss)`, `Sigma *= (1+var_increase)`.

### 1.4 Green putting model — GPR #1 (ESHO scheme)

**Class** `_PuttGPModel(gpytorch.models.ExactGP)` (`core.py:44-60`):
- Mean function: `ConstantMean()`.
- Covariance: `ScaleKernel(RBFKernel())` — isotropic RBF with a learned output-scale, default (untouched) lengthscale initialisation.
- Likelihood: `GaussianLikelihood()`.
- **Training data**: `gpr_green_dataset.csv` (`x, y, simulated_strokes`), `y` shifted `+160` to match game coordinates. Target: `E[strokes to hole out | landing (x,y) on the green]`.
- **Optimiser**: `Adam(lr=0.1)` on `model.parameters()`; loss = negative `ExactMarginalLogLikelihood`; **100 iterations** (`gp_training_iter` default, full-batch, no early stopping / validation split).
- **Inference**: `evaluate_on_green` — point prediction, `likelihood(model(x)).mean`, no uncertainty is propagated forward into ESHO (posterior mean only).

### 1.5 Off-green expected strokes — Broadie baseline

**Source**: `strokes_by_lie_yards_broadie.csv` (Mark Broadie's strokes-gained tables), melted long by lie (`fairway`, `rough`, `sand`), interpolated with `scipy.interpolate.interp1d(kind="linear", fill_value="extrapolate")` per lie — a 1D lookup `strokes(distance | lie)`, linearly extrapolated beyond the table's range.

### 1.6 Shot geometry — rotation/translation and aim parameterisation

`rotation_translator` (`core.py:439-461`): given a club's local `(x_side, y_carry)` draw and an `angle_deg` (the *aim* offset expressed as an angle), the shot is rotated by the 2×2 rotation matrix
```
R(angle_deg) = [[cos θ, -sin θ], [sin θ, cos θ]]
```
applied to `(x_side, y_carry)`, then re-expressed in the global frame using the unit vector from `starting_point` to `target`, and added to `starting_point`. Aim is stored as a **lateral yard offset at the target** (`aim_offset`), converted to an angle via `angle_deg = degrees(atan(aim_offset / total_distance))` — so the same `aim_offset` value means a larger angular adjustment for a short shot than a long one.

### 1.7 Lie classification and hazard handling

`get_lie_category` (`core.py:338-348`): point-in-polygon containment tests, in priority order **green → water → bunker → fairway → else rough** (fairway/bunker/water are exclusive regions; anything uncategorised defaults to rough).

- **Water**: `_get_water_drop` finds where the straight line from the shot's origin to the (water) landing point first crosses a water polygon boundary; the ball is treated as dropping there. Evaluated as `1.0 (penalty) + Broadie(drop point → target, "rough")`.
- **[New, this session] Out of bounds**: `is_out_of_bounds(point, hole)` — three half-plane/line tests (`x < ob_x_left`, `x > ob_x_right`, `y > ob_y_far`). Checked on every simulated landing point *before* lie classification. An OB shot's value is `Broadie(playing_location → target, playing_location's lie) + 1.0` — computed once per grid point (constant across all clubs/aims at that point, not resimulated per shot), i.e. "as if you'd replayed from where you stood, plus a one-stroke penalty." Non-recursive by construction (depends only on the fixed origin, not on the yet-unknown optimum), and because it is always ≥ a normally-played shot's value from the same lie, an OB-prone (club, aim) combination can never be the argmin.

### 1.8 Approach-shot Monte Carlo evaluation and per-point optimisation

`simulate_approach_shots` (`core.py:475-609`):
- **Club shortlist per grid point**: ranked by `|club mean carry − distance to pin|`; for `distance ≤ 150 yd`, the **5** closest clubs are evaluated; for `distance > 150 yd`, **every** non-Driver club is evaluated (the carry-proximity heuristic is unreliable at long range). Driver is excluded everywhere (approach shots only).
- **Aim grid**: `aim_offset ∈ arange(-40, 40+5, 5)` yards (17 values), `aim_range`/`aim_step` are configurable per run.
- **Sampling**: for each (grid point, club, aim), `n_new` i.i.d. draws from the club's (possibly rough-adjusted) bivariate normal, rotated to a landing point, evaluated to a stroke value as above.
- **Incremental accumulation**: shots are *not* redrawn from scratch each sweep step — an `accumulator` dict keyed by `(x, y, club, aim)` carries forward all prior draws, so the total sample size per combo grows monotonically across the `N`-sweep (this is what makes the convergence study possible without re-simulating history).
- **Per-combo statistic**: `R(s,θ) = mean(strokes) + penalty` (`penalty = 1.0`, or `2.0` if the *grid point itself* starts in water), `var(s,θ) = var(strokes)`, `SE(s,θ) = sqrt(var / n_total)`.
- **Per-point optimum**: `argmin_θ R(s,θ)` — a single best (club, aim), used for the arg-min diagnostics (plots, match-rate); the **equivalence-set** definition (Part 2) is the actual object of study, not this arg-min.

### 1.9 [Notebook only — not yet ported to `core.py`] Tee-shot layer via a second GPR

Present in `PART 3 / Simulating Par 4/scriptpar4.py` (lines ~830-978), absent from the
modular `convergence/` pipeline — flagged as the literal "missing section":

1. Fit a **second** `ExactGP` (`GPModelApp`) — same `ConstantMean + ScaleKernel(RBFKernel)` structure, but RBF lengthscale explicitly initialised to `15.0` (`requires_grad=True`, so it's a warm start, not fixed) — on `X = (x, y)` of every grid point's optimal approach result, `y = mean ESHO` at that point. `Adam(lr=0.1)`, 100 iterations, `GaussianLikelihood`. This produces a **smooth ESHO surface** over the whole fairway from the 280 discrete optimal points.
2. `evaluate_tee_shot`: for each candidate tee club × aim (aim grid `(-30, 30)`, step 2, `n_samples=50`), simulate tee shots, rotate to a landing point, and query the *fitted ESHO surface* (not a fresh Broadie/putting evaluation) for the predicted ESHO at that landing spot. Best (club, aim) by mean predicted ESHO is the recommended tee shot.
3. This is the concrete instance of CLAUDE.md's stated goal #3 ("another GPR is fit to these optimal points... in Par 4s") — it exists and runs in the notebook but has no equivalent module, CLI, or test coverage in `convergence/`.

### 1.10 Libraries, methods, and parameter settings (ESHO scheme, summary)

| Component | Library / method | Key settings |
|---|---|---|
| Geometry | `shapely` (affine transforms), `geopandas` (CRS reprojection, `EPSG:32611`) | fixed shift/rotate sequence, §1.2 |
| Green GPR | `gpytorch.models.ExactGP`, `ConstantMean`, `ScaleKernel(RBFKernel)`, `GaussianLikelihood` | Adam(lr=0.1), 100 iters, full-batch exact GP |
| Off-green baseline | `scipy.interpolate.interp1d` | linear, extrapolated |
| Shot dispersion | empirical bivariate normal per club (`numpy.cov`) | `np.random.multivariate_normal` sampling |
| Optimisation | brute-force grid search over (club, aim) per grid point | Monte Carlo mean/var per combo, incremental accumulator |
| Tee-shot layer (notebook only) | second `ExactGP` fit on optimal-point surface | lengthscale warm-start 15.0, Adam(lr=0.1), 100 iters |

---

## Birdie scheme (mirrors §1, differences called out)

`Parallelisation/convergence_birdie/core_birdie.py`. Geometry, tee/pin location,
club dispersion, and rough adjustment are **identical** to §1.2–1.3 (same functions,
same parameters). Differences:

### B.1 Green model — GPR #1', binary classification instead of regression

**Class** `_BirdieGreenModel(gpytorch.models.ApproximateGP)`:
- **Sparse/variational GP**, not exact: `CholeskyVariationalDistribution` + `VariationalStrategy(learn_inducing_locations=True)`. Inducing points are initialised to the *entire training set* (`inducing_points = X_train.clone()`, not a subsample) — so at `gp_training_iter` default, this is a variational approximation with as many inducing points as data, mainly buying the Bernoulli likelihood rather than sparsity.
- Mean/covariance structure: same `ConstantMean + ScaleKernel(RBFKernel)` as GPR #1.
- **Likelihood**: `BernoulliLikelihood()` — implicitly a probit link (GPyTorch's default `BernoulliLikelihood` uses the standard normal CDF, not a logistic/logit link — worth stating explicitly in a methods section since CLAUDE.md's stated goal says "Logit/Probit" generically).
- **Target**: binary, `y = 1[simulated_strokes == 1]` (a 1-putt, i.e. a birdie make from that spot), from the same `gpr_green_dataset.csv`.
- **Objective**: `VariationalELBO(likelihood, model, num_data=len(y_train))`, optimised jointly over `birdie_model.parameters()` and `birdie_likelihood.parameters()` (both listed as separate Adam param groups), `lr=0.1`, `gp_training_iter` default **200** (vs. 100 for GPR #1 — undocumented why the default differs; worth reconciling or justifying explicitly).

### B.2 Off-green shots

No Broadie table is used at all in the birdie scheme: any landing spot not classified `"green"` contributes `P(birdie) = 0` directly (`simulate_approach_shots_birdie:484-488`) — birdie is only possible by holing out in one putt, and only "on the green" is modelled.

### B.3 Objective and water/OB handling

- **Maximise**, not minimise: `argmax_θ mean(P(birdie))`, mirroring §1.8's argmin structure exactly but with the comparison flipped.
- **Water-starting grid points**: hard-coded `P(birdie) = 0`, club/aim recorded as a placeholder (`"Driver"`, `0.0`) rather than actually searched — this both differs from and is cruder than the ESHO scheme's water handling (which computes a real drop-point-based ESHO).
- **No OB logic**: the birdie scheme has not received the §1.7 OB update — same "gaming at long range" failure mode this session's OB fix addressed in the ESHO scheme is presumably still present here. **Flag as follow-up work**, not yet done.
- **Aim grid**: still the *old* `(-20, 20)`, step `2` default — not updated to the ESHO scheme's current `(-40, 40)`, step `5`. **Flag as a drift to reconcile** if the two schemes are meant to stay comparable.
- **Club shortlist**: always top-5 by carry proximity, no `>150 yd → all clubs` branch (that fix, made earlier this session to the ESHO scheme, has not been mirrored here either).

### B.4 Libraries, methods, and parameter settings (birdie scheme, summary)

| Component | Library / method | Key settings |
|---|---|---|
| Green model | `gpytorch.models.ApproximateGP`, `CholeskyVariationalDistribution`, `VariationalStrategy` | inducing points = full training set, `learn_inducing_locations=True` |
| Likelihood | `BernoulliLikelihood` (probit link) | — |
| Objective | `VariationalELBO` | Adam(lr=0.1), 200 iters, joint model+likelihood params |
| Target | binary 1-putt indicator | from `gpr_green_dataset.csv` |
| Optimisation | brute-force grid search, maximise mean P(birdie) | same incremental-accumulator structure as ESHO scheme |

---

## PART 2 — Convergence Scheme

### 2.1 Motivating problem

A single arg-min `(club*, aim*)` per grid point is not a well-defined statistic
when several (club, aim) combinations are statistically indistinguishable in
expected outcome — at many grid points the arg-min can oscillate indefinitely
as `N` grows even once the model has converged in every practically meaningful
sense (confirmed empirically before this session's OB fix: long-range points
never stabilised because an unpenalised "gap" beyond the water let noise
relabel the arg-min every sweep step). Convergence is therefore defined and
tested on the **set of near-optimal strategies**, not the single arg-min.

### 2.2 Definitions

For grid point `g`, seed `k`, sample size `N`:
- `R(s,θ)` — mean ESHO for combination `(club=s, aim=θ)`, §1.8 (lie penalty included).
- `SE(s,θ) = sqrt(var(s,θ) / n_total(s,θ))` — standard error of that mean.
- `R_min = min_θ R(s,θ)`, `SE_min` = the standard error **of the arg-min combination**, not each candidate's own SE.
- **Equivalence set**: `E*(g,k,N) = { (s,θ) : R(s,θ) ≤ R_min + e·SE_min }`, `e = 1` (a one-sided, one-standard-error band around the best observed combination).
- Set membership uses **exact** `(club, aim)` identity (aim rounded to 1 dp to absorb floating-point noise) — no aim-tolerance binning (removed this study; the coarser 5 yd aim grid made binning redundant and was actively hiding real disagreement).

### 2.3 Test 1 — within-seed stabilisation

Computed **live**, inside the HPC worker (`equivalence.SeedStabilityTracker`), not
as post-processing:
- At each `N`, compute Jaccard similarity `J = |E*_N ∩ E*_{N-1}| / |E*_N ∪ E*_{N-1}|` between consecutive equivalence sets for the same grid point.
- A grid point is **stabilised** at the first `N` where `J = 1` (`jaccard_threshold`) for `k_consecutive = 3` snapshots in a row (i.e. 2 consecutive unchanged transitions).
- The worker logs, per `N`: `% points stabilised (ever)`, `% points currently stable`, `% points with |E*| = 1`, mean Jaccard vs. the previous `N`, and a `all_points_stable` boolean — plus, per seed, the first `N` at which 100% of points were stabilised (`reached_100pct_stable`, `first_N_100pct_stable`).
- The sweep always runs to `N_max = 500` regardless of stabilisation — no early stopping — so non-converged points/seeds are *reported*, not silently dropped.

### 2.4 Test 2 — cross-seed agreement

100 independent seeds each run the identical sweep (`Parallelisation/convergence/submit_hpc.sh`, a Slurm job array, seed = array task ID). At fixed `(g, N)`:
- **Pairwise Jaccard**: for every seed pair, `J(E*_i, E*_j)`, averaged over all `100·99/2` pairs per grid point (`cross_seed_jaccard.py`).
- **Core / union**: `core = ∩_k E*(g,k,N)` (elements *every* seed agrees belong in the set), `union = ∪_k E*(g,k,N)`; `full_agreement_frac` = fraction of seeds whose `E*` exactly equals the core.
- A grid point is reported "converged" (majority rule) if ≥ 50% of its seeds passed Test 1.

### 2.5 Theoretical framing

- The equivalence-set approach is a form of the classic **selection-uncertainty /
  multiple-comparisons-with-the-best (MCB)** problem in ranking-and-selection: rather
  than reporting a single winner (which is statistically unstable when several
  arms have overlapping confidence intervals), report the *confidence set* of
  arms not significantly worse than the observed best. The `e·SE_min` band is a
  crude (non-simultaneous, non-Bonferroni-corrected) one-standard-error
  selection rule — `e ≈ 1` in strict Gaussian terms is a ~68% one-sided band per
  *comparison*, not a joint coverage guarantee across the whole candidate set;
  this should be stated as a limitation (a true simultaneous-inference correction,
  e.g. Rinott's procedure or a Bonferroni/Šidák adjustment over the ~5–19
  candidate clubs × 17 aims, would tighten or loosen `E*` depending on the
  correction and is a natural extension).
- Test 1 (within-seed) answers "has *this* random stream of Monte Carlo draws
  settled" — a question about **simulation budget**. Test 2 (cross-seed) answers
  "would a *different* random stream reach the same conclusion" — a question
  about **estimator variance / reproducibility**, orthogonal to Test 1. A point
  can pass Test 1 in every seed while failing Test 2 (each seed individually
  stable, but stable on a *different* set) — this is exactly the diagnostic value
  of running both, and (per the analysis in `MyScripts/convergence_stabilisation.qmd`)
  is the dominant pattern observed on this hole: most cross-seed disagreement is
  driven by aim-offset instability (seeds agree on the club, disagree on the
  exact aim within the grid), not club-level disagreement.

### 2.6 Execution model (HPC)

- `submit_hpc.sh`: Slurm array `0..N_SEEDS-1`, one task per seed, each running `run_hpc_worker.py --seed $SLURM_ARRAY_TASK_ID`.
- `convergence_worker.py::run_convergence`: per seed, sweeps `N = n_start(10), n_start+n_step, ..., n_max(500)` (`n_step=10`), calling `simulate_approach_shots` each step with `n_new = n_step` new shots merged into the accumulator (§1.8).
- Per-`N` outputs, per seed: `seedNNNN_NXXXX_equivset.csv` (only the equivalence-set members — not the full candidate table, to keep the HPC output small: ~5,000 files / 100 seeds instead of tens of GB), `seedNNNN_stabilisation.tsv` (Test 1 time series), `seedNNNN_result.json` (final summary + 100%-stable flag).
- Retrieval: `pack_hpc_results.sh` (tar the small per-seed files on the HPC) → `fetch_hpc_results.sh` (scp/rsync to a laptop, since the HPC requires Duo and has no browser/SSH reachable off-campus — Open OnDemand's file browser is the fallback download path).
- Post-processing: `run_equivalence_analysis.py` (aggregate Test 1 across seeds, spatial map) and `cross_seed_jaccard.py` (Test 2 pairwise matrices) — both pure Python/pandas, no resimulation. Final reporting/visualisation in R (`MyScripts/convergence_stabilisation.qmd`, Quarto).

### 2.7 The out-of-bounds fix, as convergence-scheme methodology (this session)

Framed as a methods point, not just an implementation note: prior to this
change, long approach shots (grid points near the tee, where the `>150 yd → all
clubs` branch already fires) had a genuine unmodelled escape — landing right of
the water hazard or left of the fairway carried **no stroke penalty at all**,
so a wide-dispersion club could occasionally "get lucky" and land in a
penalty-free zone with a better expected outcome than a controlled shot,
inflating that combination's mean and destabilising its rank across Monte
Carlo draws (directly causing the Test-1 non-convergence this session set out
to diagnose). The fix (§1.7) closes that gap with a hard penalty region on
three sides, and squares the water hazard off against the right-hand OB line
so there is no remaining unpenalised strip between "water" and "OB." This is a
**model-fidelity correction**, expected to materially reduce `|E*|` and reduce
`converged_N` at the long-range grid points specifically — the natural
before/after comparison for the paper is `equiv_set_size_final` and
`converged_N` at those points, pre- vs. post-OB, on matched seeds.

---

## Open items / explicitly flagged as missing or inconsistent

1. Tee-shot GPR layer (§1.9) — exists only in the notebook, not in `core.py`/`convergence/`. No CLI, no HPC worker, no tests.
2. Birdie scheme drift from the ESHO scheme (§B.3): no OB logic, old aim grid, old club-shortlist rule, cruder water handling, different `gp_training_iter` default (200 vs 100) with no stated justification.
3. `BernoulliLikelihood`'s link function (probit, GPyTorch default) should be stated explicitly rather than left as "Logit/Probit" per CLAUDE.md's goal wording — the two links are not interchangeable and only one is actually implemented.
4. The `e·SE_min` equivalence band is not a simultaneous-inference correction (§2.5) — worth either justifying as an intentional simplification or extending.
5. No confidence-bound propagation from GPR #1's posterior *variance* into ESHO — only the posterior mean is used (§1.4); CLAUDE.md goal #3 ("keep the full distribution... to allow for confidence bound estimation") is partially met (raw shot distributions are retained in the accumulator) but the green-model uncertainty itself is discarded at the point of evaluation.
