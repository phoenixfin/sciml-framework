# Research Plan: SINDy-Based Discovery of Dengue Transmission Dynamics
*Two-Paper Series — Working Document*

---

## Background & Motivation

Dengue fever is a vector-borne disease with complex, partially observable dynamics. Public health
surveillance typically yields only **weekly case counts** — no data on susceptible population $S(t)$,
exposed $E(t)$, or mosquito density. Standard compartmental models (SIR, SEIR, SEIR-SEI) require
these unobserved variables as inputs, forcing analysts to either assume them or reconstruct them
with additional assumptions.

Sparse Identification of Nonlinear Dynamics (SINDy) offers an alternative: discover the governing
equations *inductively from data*, without assuming model structure in advance. Recent work has applied
SINDy to measles, chickenpox, and COVID-19 — but **no SINDy paper on dengue exists**. More
importantly, the existing approaches each have a critical limitation in the dengue context:

- **SPADE4** (Saha et al., 2022): works from $I(t)$ only via delay embedding, but produces
  uninterpretable random feature models — no mechanistic insight.
- **Horrocks & Bauch** (2020): produces interpretable epidemiological equations, but requires
  explicit reconstruction of $S(t)$ via a 2-parameter grid search — an additional layer of
  assumptions on top of the data.
- **DDE-SINDy** (Breda et al., 2025/2026): formalizes delay differential equation discovery, but
  has only been applied to tick-borne diseases, not dengue.

This two-paper series fills these gaps through a progression from method development (Paper 1) to
mechanistic formalization (Paper 2).

---

## Paper 1

### Title (working)
*Interpretable Data-Driven Discovery of Dengue Transmission Dynamics from Infected-Only Data via
Delay-Embedded SINDy*

### Core Contribution

A new SINDy variant that combines:
1. **Takens delay embedding** — reconstructs hidden state dimensions from lagged values of $I(t)$
   alone, eliminating the need for $S(t)$ reconstruction
2. **Epidemiological candidate library** — polynomial and seasonally-forced terms, as in Horrocks,
   applied to delay coordinates rather than $(S, I)$
3. Applied to **dengue** for the first time

This bridges SPADE4 (single-variable input) and Horrocks (interpretable library) into a single
framework that requires fewer assumptions than either alone.

### Key Research Questions

1. Can SINDy recover interpretable epidemic structure (mass-action incidence, seasonal forcing)
   from dengue incidence data alone, without reconstructing $S(t)$?
2. Which delay lags $\tau$ are selected by the sparse regression — and do they correspond to known
   dengue biology (intrinsic incubation ~1 week, extrinsic incubation in *Aedes* ~2 weeks)?
3. Does the delay-embedded approach outperform SPADE4 (in interpretability) and Horrocks (in
   data requirements) on the same dengue dataset?

### Methodology

**Step 1 — Input:** Weekly dengue incidence $j(t)$, population size $N$. No other inputs required.

**Step 2 — Preprocessing:**
- Smooth $j(t)$ with Savitzky-Golay filter (window = 7–13 weeks)
- Convert incidence to prevalence: $I(t) = \sum_{s \geq 0} j(t-s)\,e^{-\gamma s}$, $\gamma = 1/T_{\inf}$

**Step 3 — Delay Embedding:**
Build the Takens delay matrix with embedding dimension $p$ and lag $\tau$:
$$\mathbf{h}(t) = [I(t),\; I(t-\tau),\; I(t-2\tau),\; \ldots,\; I(t-(p-1)\tau)]$$
By Takens' theorem, $\mathbf{h}(t)$ is diffeomorphic to the full state $[S(t), I(t), R(t), \ldots]$
under mild conditions.

**Step 4 — Epidemiological Library:**
Construct candidate function library on delay coordinates $\mathbf{h}$:
$$\Theta = \left[\,1,\; h_1,\; h_2,\; \ldots,\; h_1 h_2,\; h_1 h_3,\; \ldots,\; \beta(t)\cdot h_i h_j,\; \ldots\,\right]$$

Key terms to include:
- Polynomial terms up to degree 2 in delay coordinates
- Seasonal forcing: $\beta(t) = 1 + \alpha\sin(2\pi t/52 + \phi)$
- Cross terms $h_i \cdot h_j$ (analog of mass-action $S \cdot I$, but in delay space)
- Optional: climate covariates if available (rainfall, temperature)

**Step 5 — Sparse Regression (STLS):**
$$\hat{\Xi} = \arg\min_{\Xi} \|\Theta\,\Xi - \dot{\mathbf{h}}\|_2^2 \quad \text{s.t.} \quad \|\Xi\|_0 \text{ small}$$
Use Sequentially Thresholded Least Squares (STLS). Model selection via AIC over grid of
$(p, \tau, \alpha, \phi)$.

**Step 6 — Validation:**
- In-sample trajectory fit
- Out-of-sample forecast (hold-out last season)
- Power spectral density comparison (captures annual/multi-annual cycle structure)
- Ablation: compare to SPADE4 (interpretability) and Horrocks (data requirements)

### Novelty Statement

> *No existing method recovers interpretable epidemic structure from a single incidence time series
> without reconstructing unobserved compartments. This paper fills that gap by replacing
> susceptible reconstruction with delay embedding while preserving the epidemiological
> interpretability of the candidate library.*

### Data Requirements

| Data | Required | Source |
|------|----------|--------|
| Weekly dengue incidence | ✅ Yes | In hand |
| Population size $N$ | ✅ Yes | BPS / Dinas Kesehatan |
| Infectious period $T_{\inf}$ | Approximate (3–7 days) | Literature |
| Climate data (temp, rainfall) | Optional (strengthens paper) | BMKG |

### Target Journals
- *Bulletin of Mathematical Biology* (primary)
- *Chaos: An Interdisciplinary Journal of Nonlinear Science*
- *Journal of Theoretical Biology*

---

## Paper 2

### Title (working)
*Sparse Identification of Delay Differential Equations for Dengue: Recovering Vector-Host Dynamics
from Case Notification Data*

### Core Contribution

Formalization of dengue dynamics as a **delay differential equation (DDE)** discovered directly
from incidence data using the Breda et al. (2025) DDE-SINDy framework. The key biological
hypothesis: the delays discovered in Paper 1 empirically can be explained by the vector-host
transmission cycle, and a formal DDE model can encode this.

The dengue transmission chain has two known biological delays:
- **Intrinsic incubation** (human): $\tau_1 \approx 4$–$7$ days ($\approx 1$ week)
- **Extrinsic incubation** (mosquito): $\tau_2 \approx 8$–$12$ days ($\approx 1$–$2$ weeks)

Paper 2 asks whether these delays can be *recovered from case data alone*, without entomological
observations.

### Key Research Questions

1. Can DDE-SINDy discover the two-delay structure ($\tau_1$, $\tau_2$) of dengue from incidence
   data, and do the values match known biology?
2. Does a DDE model provide better long-term trajectory fits than the ODE model from Paper 1?
3. Can the discovered DDE be used to estimate the basic reproduction number $R_0$ and its
   sensitivity to delay parameters?

### Methodology

**Foundation:** Breda et al. (2025) framework for sparse identification of DDEs using quadrature
approximation of distributed delay terms.

**DDE candidate library extension:**
$$\dot{I}(t) = \Xi^T \cdot \left[\,I(t),\; I(t-\tau_1),\; I(t-\tau_2),\; I(t)\cdot I(t-\tau_2),\; \ldots\,\right]$$
where $\tau_1$, $\tau_2$ are either fixed from biology or optimized jointly with $\Xi$.

**Delay identification strategy:**
- Grid search over discrete delays $\tau \in \{1, 2, \ldots, 8\}$ weeks
- Bayesian optimization over continuous delay values (Pecile et al., 2024)
- Compare AIC of DDE models with 1, 2, and 3 delay terms

**Connection to vector dynamics:**
Map the discovered DDE structure back to a reduced SEIR-SEI (human-mosquito) model to interpret
coefficients in terms of $\beta_h$, $\beta_v$, $\gamma$, $\mu_v$.

### Builds on Paper 1

Paper 1 establishes *empirically* which delay lags $\tau$ are selected by sparse regression on
dengue data. Paper 2 takes those lags as biological hypotheses and formalizes them as a DDE system.
The narrative connection:

> *"In Paper 1, we found that delay coordinates at $\tau \approx 2$ weeks dominated the discovered
> dynamics. Here we show this corresponds to the extrinsic incubation period in Aedes aegypti,
> and formalize it as a biologically interpretable DDE."*

### Data Requirements

| Data | Required | Notes |
|------|----------|-------|
| Weekly dengue incidence | ✅ Yes | Same as Paper 1 |
| Entomological indices (Breteau, HI) | Strongly preferred | For delay validation |
| Mosquito lifespan estimates | From literature | Validates $\tau_2$ |

### Target Journals
- *Journal of Mathematical Biology* (primary)
- *Mathematical Biosciences*
- *PLOS Computational Biology*

---

## Logical Progression Between Papers

```
Paper 1                             Paper 2
─────────────────────────────       ────────────────────────────────────
Method: Delay-embedded SINDy   →    Method: Formal DDE-SINDy
Input:  I(t) only              →    Input:  I(t) [same] + validate with
                                            entomological data
Output: Interpretable ODE      →    Output: DDE with biological delays
        in delay coordinates
Finding: τ ≈ 2 weeks selected  →    Finding: τ matches extrinsic
         by sparse regression           incubation in Aedes
```

---

## Suggested Timeline

| Milestone | Paper 1 | Paper 2 |
|-----------|---------|---------|
| Finalize methodology | Month 1 | Month 4 |
| Run on real dengue data | Month 2 | Month 5 |
| Benchmarking / ablation | Month 2–3 | Month 5–6 |
| Draft manuscript | Month 3–4 | Month 6–7 |
| Submission | Month 5 | Month 8–9 |

---

## Open Questions (to resolve before writing)

1. **Data granularity:** Is the dengue data weekly or monthly? Weekly is needed for delay
   resolution at $\tau = 1$–$2$ weeks. Monthly data limits delay identification.
2. **Data length:** How many years? Ideally $\geq 3$ years for seasonal pattern + holdout.
3. **Climate covariates:** Is BMKG data available for the same region and period?
4. **Entomological data:** Any mosquito index data for Paper 2 validation?
5. **Reporting rate:** Is underreporting magnitude known (e.g., from seroprevalence studies)?
   This affects normalization and should be discussed as a limitation in both papers.

---

## Key References

| Paper | Relevance |
|-------|-----------|
| Horrocks & Bauch (2020), *Sci. Rep.* | Epidemiological SINDy library; basis for Paper 1 library design |
| Saha et al. (2022), *Bull. Math. Biol.* | SPADE4; benchmark for Paper 1 |
| Breda et al. (2025), preprint | DDE-SINDy framework; theoretical basis for Paper 2 |
| Breda et al. (2026), preprint | DDE-SINDy on vector-borne disease (SFTS); closest existing work to Paper 2 |
| Pecile et al. (2024) | Bayesian optimization for delay identification; tool for Paper 2 |
| Wang et al. (2025) | Climate-forced SINDy for influenza; template for climate extension |
| Bakarji et al. (2022) | Deep delay autoencoder + SINDy; optional advanced baseline |
