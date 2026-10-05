# In-Depth Theoretical Foundations: Initial State Perturbations for Coastal Ocean Ensembles in Norkyst-ROMS

## 1. Introduction & Physical Motivation

In numerical simulations of the Norwegian Coastal Current (NCC) using the Regional Ocean Modeling System (ROMS; e.g., the Norkyst-800 implementation with horizontal resolution $\Delta x \approx 800\text{ m}$), the circulation is characterized by intense mesoscale and submesoscale eddy fields. The NCC is a buoyant, topographically steered boundary current driven by freshwater discharge from Norwegian fjords and Baltic outflow, flowing northward alongside the saline Atlantic Inflow across the continental slope.

When investigating subtle regional hydrodynamic impacts of offshore wind farm parameterizations—specifically increased atmospheric drag, turbine structural drag, and enhanced vertical mixing within the water column—the deterministic difference between a single unperturbed wind farm run and a control run:

$$\Delta \mathbf{x}(t) = \mathbf{x}_{\text{farm}}(t) - \mathbf{x}_{\text{ctrl}}(t)$$

rapidly becomes overwhelmed by chaotic divergence in the eddy field. Because the atmosphere-ocean dynamical system exhibits sensitive dependence on initial conditions (Lorenz 1963, 1969), two model runs initiated from nearly identical states diverge along distinct trajectories on the model's strange attractor. In an eddy-rich coastal regime, the local eddy phase difference $|\mathbf{x}'_{\text{eddy}}|$ easily reaches $0.2\text{--}0.5\text{ m s}^{-1}$ in velocity and $0.5\text{--}2.0\text{ }^\circ\text{C}$ in temperature, while the direct far-field footprint of wind turbine mixing might be an order of magnitude smaller ($0.01\text{--}0.05\text{ m s}^{-1}$ and $0.05\text{--}0.2\text{ }^\circ\text{C}$).

To isolate the forced anthropogenic signal from the turbulent unforced background, an ensemble approach is necessary. By generating an ensemble of $K$ realizations with perturbed initial states:

$$\bar{\mathbf{x}}(t) = \frac{1}{K} \sum_{k=1}^K \mathbf{x}^{(k)}(t)$$

the unforced, uncorrelated eddy fluctuations cancel out at a rate scaling with $\mathcal{O}(K^{-1/2})$, permitting the robust detection of the forced wind farm response.

However, perturbing an initial ocean state is fundamentally different from perturbing an atmospheric model. The ocean is strongly stratified, laterally bounded by complex bathymetry and coastlines, and governed by a split-explicit free-surface barotropic/baroclinic time-stepping scheme. Arbitrary or poorly constructed initial perturbations fail catastrophically: they either radiate away as spurious high-frequency acoustic/inertia-gravity wave shocks (spin-up shocks), violate the Courant-Friedrichs-Lewy (CFL) numerical stability criteria, or trigger unphysical convective mixing at $t=0$.

This note provides a rigorous theoretical exposition of the geophysical fluid dynamics governing coastal perturbations, evaluates the mathematical foundations of candidate perturbation methods from literature, and details the dynamical justification for the EnOI/Historical Anomaly approach adopted for Norkyst-ROMS.

---

## 2. Geophysical Fluid Dynamics of the Coastal Ocean Attractor

### 2.1 Spatial and Temporal Scales of Instability
The Norwegian coastal current exhibits both baroclinic and barotropic instabilities:
1. **First Baroclinic Rossby Radius of Deformation ($R_D$):**
   $$R_D = \frac{N H_p}{\pi f_0} = \frac{1}{\pi f_0} \int_{-H}^0 N(z) \, dz$$
   where $N(z) = \sqrt{-\frac{g}{\rho_0} \frac{\partial \rho}{\partial z}}$ is the Brunt-Väisälä buoyancy frequency, $H_p$ is pycnocline depth, and $f_0 \approx 1.2\text{--}1.4 \times 10^{-4}\text{ s}^{-1}$ is the Coriolis parameter along the Norwegian shelf ($58^\circ\text{N}\text{--}71^\circ\text{N}$). Along the Norwegian coast, strong shallow freshwater stratification yields $R_D \approx 5\text{--}15\text{ km}$.
2. **Mesoscale Eddy Scales:**
   The fastest growing baroclinic waves (Eady 1949, Charney 1947, Pedlosky 1987) have wavelengths $\lambda \approx 3.9 R_D \approx 20\text{--}60\text{ km}$. With an 800 m grid, Norkyst resolves these modes with $25\text{--}75$ grid points per wavelength, placing it firmly in the eddy-resolving regime.
3. **E-folding Growth Time (Lyapunov Horizon):**
   The maximum baroclinic growth rate $\sigma_{\max}$ is given by:
   $$\sigma_{\max} = 0.31 \frac{f_0}{\sqrt{\text{Ri}}} = 0.31 \frac{f_0}{N} \left| \frac{\partial \mathbf{u}}{\partial z} \right|$$
   where $\text{Ri} = N^2 / (\partial_z u)^2$ is the gradient Richardson number. In the Norwegian Coastal Current, typical vertical shears of $0.3\text{ m s}^{-1}$ over $30\text{ m}$ yield e-folding times:
   $$\tau_e = \sigma_{\max}^{-1} \approx 2\text{--}5\text{ days}$$
   Consequently, after $2\text{--}3$ weeks of integration, any dynamically balanced initial perturbation will cause the eddy field to decorrelate completely from the unperturbed trajectory, generating an independent realization of the coastal eddy field.

### 2.2 The Manifold of Balanced Ocean States
In geophysical fluid dynamics, high-frequency motions (inertia-gravity waves, barotropic Kelvin waves, edge waves) and low-frequency, turbulent motions (Rossby waves, geostrophic eddies) coexist. The slow, balanced motions lie on an invariant "slow manifold" (Leith 1980, Lorenz 1986).

When an ocean model is initialized with a state $\mathbf{x}_0 + \delta \mathbf{x}$, if $\delta \mathbf{x}$ has a projection onto the fast gravity wave modes, the model undergoes **geostrophic adjustment** (Rossby 1937, 1938; Gill 1982):
- Energy projected onto scales smaller than the Rossby deformation radius ($L \ll R_D$) radiates away as inertia-gravity waves, leaving the mass/density field largely unchanged while the velocity field adjusts.
- Energy projected onto scales larger than the Rossby deformation radius ($L \gg R_D$) forces the density field to adjust to the velocity field through vertical motions and acoustic/gravity wave radiation.

In a hydrostatic primitive equation model like ROMS:
- High-frequency oscillations generated by unbalanced initial perturbations excite the external (barotropic) mode.
- In ROMS's split-explicit formulation (Shchepetkin & McWilliams 2005), barotropic waves travel at phase speed $c_{\text{ext}} = \sqrt{g H} \approx 30\text{--}200\text{ m s}^{-1}$ along the shelf. Large unbalanced pressure gradients cause rapid oscillations in the free surface $\zeta$ and barotropic velocities $\bar{u}, \bar{v}$, triggering limiter activations or violating the barotropic CFL criterion:
  $$\Delta t_{\text{ext}} \le \frac{\min(\Delta x, \Delta y)}{\sqrt{2 g H_{\max}}}$$
- Even if the simulation survives numerically, numerical diffusion and sponge layers damp this spurious energy over a 48–72 hour "spin-up" period, leaving the eddy field essentially unperturbed. Thus, **perturbations must be in approximate balance to survive and trigger genuine eddy divergence.**

---

## 3. Comprehensive Evaluation of Perturbation Methodologies

The oceanographic and meteorological literature offers several distinct paradigms for generating initial ensemble perturbations. Below is a detailed mathematical comparison of these methods in the context of high-resolution coastal ROMS simulations.

```
                                Initial Perturbation Methods
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
   Dynamic Cycling                    Data-Driven Covariance               Synthetic / Noise
         │                                    │                                    │
  ├── Bred Vectors (BV)                ├── Historical Anomaly (EnOI)        ├── White / Coloured Noise
  └── Singular Vectors (TLM/ADM)       └── Multivariate EOF / PCA           └── Geostrophic Random Fields
```

---

### 3.1 Historical Anomaly / Ensemble Optimal Interpolation (EnOI) Sampling

#### Mathematical Formulation
Let $\mathcal{A} = \{ \mathbf{x}_{\text{hist}}(t_1), \mathbf{x}_{\text{hist}}(t_2), \dots, \mathbf{x}_{\text{hist}}(t_M) \}$ be an ensemble of $M$ historical ocean states produced by the exact model configuration (Norkyst-800) during the same calendar season across previous days or years. The sample mean state is:

$$\bar{\mathbf{x}} = \frac{1}{M} \sum_{m=1}^M \mathbf{x}_{\text{hist}}(t_m)$$

The anomaly vector for historical state $m$ is:

$$\mathbf{x}'_m = \mathbf{x}_{\text{hist}}(t_m) - \bar{\mathbf{x}}$$

The ensemble perturbation for member $k \in \{1, \dots, K\}$ is synthesized as a scaled linear combination:

$$\delta \mathbf{x}^{(k)} = \alpha \sum_{m=1}^M w_{m,k} \, \mathbf{x}'_m$$

$$\mathbf{x}^{(k)}(t_0) = \mathbf{x}_0(t_0) + \delta \mathbf{x}^{(k)}$$

where $\mathbf{w}_k = [w_{1,k}, \dots, w_{M,k}]^T \in \mathbb{R}^M$ is a weight vector, and $\alpha \in [0.05, 0.15]$ is a dimensionless scaling factor.

To ensure that ensemble members explore mutually independent directions in phase space, the weight matrix $\mathbf{W} \in \mathbb{R}^{K \times M}$ is constructed to be orthonormal:

$$\mathbf{W} \mathbf{W}^T = \mathbf{I}_K \quad (M \ge K)$$

This is achieved via thin QR decomposition of an $M \times K$ standard normal random matrix $\mathbf{G} \sim \mathcal{N}(0, 1)$:

$$\mathbf{G} = \mathbf{Q} \mathbf{R} \implies \mathbf{W} = \mathbf{Q}^T$$

#### Theoretical Advantages
1. **Strict C-Grid Boundary Adherence:**
   On the Arakawa C-grid, coastline boundaries require $u = 0$ on east-west boundaries and $v = 0$ on north-south boundaries. Because each historical snapshot $\mathbf{x}_{\text{hist}}(t_m)$ satisfies $\mathbf{u}_m \cdot \mathbf{n} = 0$, any linear combination:
   $$\mathbf{u}_{\text{pert}} \cdot \mathbf{n} = \mathbf{u}_0 \cdot \mathbf{n} + \alpha \sum_{m=1}^M w_{m,k} (\mathbf{u}_m \cdot \mathbf{n} - \bar{\mathbf{u}} \cdot \mathbf{n}) = 0$$
   strictly vanishes at every land-sea interface.
2. **Preservation of T-S Relationships and Steric Balance:**
   Ocean water masses exhibit tight Temperature-Salinity correlations (e.g., Norwegian Coastal Water, Atlantic Water, Norwegian Sea Deep Water). Because the anomalies $\mathbf{x}'_m$ are drawn from actual model realizations on the attractor, temperature and salinity perturbations covary naturally. This preserves the local baroclinic Rossby radius and steric sea surface height gradients.
3. **Zero Computational Overhead on HPC:**
   Unlike methods requiring model integrations, the sampling is evaluated completely offline via memory-mapped NetCDF I/O in Python.

#### Limitations
- Anomalies represent climatological/seasonal variability rather than the instantaneous, fastest-growing local instability mode of day $t_0$. However, for multi-month or annual simulations where the eddy predictability horizon is $\sim 15\text{ days}$, this distinction is dynamically negligible.

---

### 3.2 Bred Vector (BV) / Nonlinear Breeding Cycle

#### Mathematical Formulation
Introduced by Toth & Kalnay (1993, 1997) for atmospheric ensembles and adapted to ocean modeling by Baehr & Piontek (2014) and Shu et al. (2011), the breeding method mimics the operational assimilation cycle to identify nonlinearly growing dynamical modes.

Given the nonlinear ROMS propagator $\mathcal{M}_{t \to t + \Delta t}$:
1. Start with control state $\mathbf{x}_0(t_0)$ and perturbed state $\mathbf{x}^{(k)}(t_0) = \mathbf{x}_0(t_0) + \delta \mathbf{x}_0$.
2. Integrate both states forward for a breeding interval $\Delta t$ (typically $24\text{--}48\text{ hours}$ in coastal ocean models):
   $$\mathbf{x}_0(t_0 + \Delta t) = \mathcal{M}(\mathbf{x}_0(t_0)), \quad \mathbf{x}^{(k)}(t_0 + \Delta t) = \mathcal{M}(\mathbf{x}^{(k)}(t_0))$$
3. Compute the raw difference:
   $$\delta \mathbf{x}_{\text{raw}} = \mathbf{x}^{(k)}(t_0 + \Delta t) - \mathbf{x}_0(t_0 + \Delta t)$$
4. Compute the total ocean energy norm $\|\delta \mathbf{x}\|$:
   $$\|\delta \mathbf{x}\|^2 = \frac{1}{2} \int_V \left[ \rho_0 (\delta u^2 + \delta v^2) + \frac{g^2}{N^2 \rho_0} \delta \rho^2 \right] dV + \frac{1}{2} \int_A g \rho_0 \delta \zeta^2 \, dA$$
5. Rescale the perturbation to the target amplitude $\varepsilon_0$:
   $$\delta \mathbf{x}_{\text{bred}} = \varepsilon_0 \frac{\delta \mathbf{x}_{\text{raw}}}{\|\delta \mathbf{x}_{\text{raw}}\|}$$
6. Add back to control: $\mathbf{x}^{(k)}_{\text{next}} = \mathbf{x}_0(t_0 + \Delta t) + \delta \mathbf{x}_{\text{bred}}$, and repeat for $J \approx 3\text{--}5$ breeding cycles.

#### Theoretical Pros & Cons
- **Pros:** Isolates the true local Lyapunov vectors (the fastest-growing baroclinic modes) of the instantaneous day-$t_0$ flow. Highly effective for short-range eddy path forecasting (1–4 weeks).
- **Cons:** Extremely expensive on HPC resources. For $K=10$ members and 4 breeding cycles of 24h, one must run $4 \times 10 = 40$ full model days of high-resolution Norkyst simulations *before* the annual run even starts. In a 1-year run, the chaotic eddy field decorrelates by week 3 regardless of whether the perturbation was a bred vector or an EnOI anomaly, making the compute cost unjustifiable.

---

### 3.3 Tangent Linear and Adjoint Singular Vectors (SVs)

#### Mathematical Formulation
Singular vectors (Farrell 1989, Palmer 1993, Moore et al. 2004) maximize linear perturbation growth over an optimization interval $\tau$ under a chosen initial norm $\mathbf{S}_0$ and final norm $\mathbf{S}_\tau$. Let $\mathbf{L}$ be the Tangent Linear Model (TLM) of ROMS, and $\mathbf{L}^*$ its Adjoint (ADM):

$$\frac{\|\mathbf{L} \delta \mathbf{x}_0\|_{\mathbf{S}_\tau}^2}{\|\delta \mathbf{x}_0\|_{\mathbf{S}_0}^2} = \frac{\delta \mathbf{x}_0^T \mathbf{L}^T \mathbf{S}_\tau \mathbf{L} \delta \mathbf{x}_0}{\delta \mathbf{x}_0^T \mathbf{S}_0 \delta \mathbf{x}_0}$$

The optimal initial perturbations are the leading eigenvectors of the propagator $\mathbf{S}_0^{-1/2} \mathbf{L}^T \mathbf{S}_\tau \mathbf{L} \mathbf{S}_0^{-1/2}$, computed using Lanczos iterations.

#### Theoretical Pros & Cons
- **Pros:** Mathematically optimal linear growth modes. Available in ROMS via `ARPACK` drivers.
- **Cons:** Requires maintaining and compiling ROMS with TLM and ADM CPP options, which often clash with custom physics parameterizations (such as structural wind farm drag and non-standard vertical mixing schemes in `alpsjur/roms`). Furthermore, linearization is only valid for short time horizons ($\tau \sim 3\text{--}7\text{ days}$).

---

### 3.4 Geostrophically Balanced Synthetic Random Fields

#### Mathematical Formulation
Generate a 2D pseudo-random Gaussian field for sea surface height perturbation $\delta \zeta(x, y)$ with spatial covariance $C(r) = \exp\left(-\frac{r^2}{2 L_h^2}\right)$, where $L_h \approx R_D \approx 10\text{ km}$ (Evensen 1994, 2003). 

From $\delta \zeta$, compute surface geostrophic velocity perturbations:
$$u_g' = -\frac{g}{f} \frac{\partial (\delta \zeta)}{\partial y}, \quad v_g' = \frac{g}{f} \frac{\partial (\delta \zeta)}{\partial x}$$
and project downward through the water column using baroclinic dynamical modes $F_m(z)$ derived from the vertical Sturm-Liouville problem:
$$\frac{d}{dz}\left( \frac{f^2}{N^2(z)} \frac{d F_m}{dz} \right) + \lambda_m^2 F_m = 0$$

#### Theoretical Pros & Cons
- **Pros:** Completely analytical; does not rely on prior archived simulations.
- **Cons:** At intricate coastlines, islands, and narrow fjords, synthetic spatial filtering does not naturally respect $\mathbf{u} \cdot \mathbf{n} = 0$. Imposing boundary masks a posteriori breaks geostrophic balance, generating violent gravity waves along the coast. Furthermore, baroclinic modal projection over steep, variable bathymetry (where $H$ changes by 400 m over 3 grid cells) creates severe artificial density steps.

---

### 3.5 Comparative Summary Matrix

| Metric / Attribute | Historical Anomaly (EnOI) | Bred Vectors (BV) | Singular Vectors (TLM/ADM) | Balanced Synthetic Noise |
| :--- | :--- | :--- | :--- | :--- |
| **Dynamical Balance** | **High** (on model attractor) | **Highest** (cyclically balanced) | **High** (optimal linear modes) | **Moderate** (degrades at boundaries) |
| **Coastal Boundary Match** | **Exact** ($\mathbf{u} \cdot \mathbf{n} = 0$) | **Exact** | **Exact** | **Poor** (requires boundary damping) |
| **Water Mass Conservation** | **Exact** (preserves $T$-$S$ curve) | **High** | **Moderate** | **Poor** (risks $N^2 < 0$) |
| **Spin-up Gravity Shocks** | **Minimal** ($< 24\text{ h}$) | **Negligible** | **Minimal** | **High** ($> 48\text{--}72\text{ h}$) |
| **Computational Cost** | **Zero model hours** (Offline Python) | **High** ($3\text{--}5$ model cycles) | **Very High** (Lanczos TLM/ADM) | **Zero model hours** |
| **Suitability for Annual Runs**| **Optimal** | **Redundant** (memory lost in 3 wks)| **Infeasible** (code divergence) | **Sub-optimal** |

---

## 4. Discrete Formulation & Technical Invariants in ROMS

When synthesizing perturbed initial condition NetCDF files for ROMS, three specific discrete numerical invariants must be satisfied.

### 4.1 Arakawa C-Grid Staggering and Land Masking
ROMS uses an Arakawa C-grid where variables are staggered horizontally:
- **$\rho$-points:** Temperature (`temp`), salinity (`salt`), free-surface height (`zeta`), and bathymetry (`h`) reside at cell centers $(j, i)$.
- **$u$-points:** Zonal momentum (`u`, `ubar`) resides at western/eastern cell faces $(j, i - 1/2)$, dimensioned $(N_\eta, N_\xi - 1)$.
- **$v$-points:** Meridional momentum (`v`, `vbar`) resides at southern/northern cell faces $(j - 1/2, i)$, dimensioned $(N_\eta - 1, N_\xi)$.

```
        (j, i) rho-point [temp, salt, zeta, h]
             ▲
             │
             │   v(j+1/2, i) [v-point]
             │         ▲
             │         │
 ───► u(j, i-1/2) ────┼──── u(j, i+1/2) ───►
      [u-point]        │
                       │   v(j-1/2, i)
                       ▼
```

If an initial condition file contains non-zero velocity values on land points (even with land-masking enabled), boundary stencil evaluations during horizontal advection (e.g., 3rd-order upstream-biased advection) will draw spurious momentum into coastal water columns. The generator must apply:
$$\mathbf{u}_{\text{pert}} = \mathbf{u}_{\text{pert}} \odot \text{mask}_u, \quad \mathbf{v}_{\text{pert}} = \mathbf{v}_{\text{pert}} \odot \text{mask}_v, \quad \mathbf{T}_{\text{pert}} = \mathbf{T}_{\text{pert}} \odot \text{mask}_\rho$$

### 4.2 Barotropic-Baroclinic Diagnostic Re-Integration
In ROMS, the total 3D horizontal velocity is split into a vertically averaged barotropic component $(\bar{u}, \bar{v})$ and a baroclinic component $(u_{\text{bc}}, v_{\text{bc}})$:

$$\bar{u}(x, y) = \frac{1}{D(x, y)} \int_{-h}^\zeta u(x, y, z) \, dz, \quad \bar{v}(x, y) = \frac{1}{D(x, y)} \int_{-h}^\zeta v(x, y, z) \, dz$$

where $D(x, y) = h(x, y) + \zeta(x, y)$ is the instantaneous total water column thickness.

In discrete terrain-following $s$-coordinates:
$$z(x, y, s, t) = \zeta(x, y, t) + [h(x, y) + \zeta(x, y, t)] \cdot z_0(x, y, s)$$
where the layer thicknesses at cell centers are $\Delta z_k = z_w(k) - z_w(k-1) = H_z(k)$.

Because the barotropic mode is stepped independently with a small acoustic time step $\Delta t_{\text{fast}}$:
$$\Delta t_{\text{fast}} = \frac{\Delta t_{\text{slow}}}{\text{NDTFAST}}$$
if the initial barotropic fields $\bar{u}_0, \bar{v}_0$ stored in the initial condition file do not exactly match the discrete vertical integral:

$$\bar{u}_{\text{file}} \neq \frac{\sum_{k=1}^N u_k \, H_{z,u}(k)}{\sum_{k=1}^N H_{z,u}(k)}$$

the model experiences a barotropic mass-flux discrepancy on time step $n=0$. This forces the mode-coupling filter to generate an immediate barotropic adjustment pulse, corrupting sea surface height $\zeta$ across the domain. The framework prevents this by diagnostically re-evaluating the vertical integral from the perturbed 3D field:

$$\bar{u}^{(k)} = \frac{1}{D_u} \sum_{k=1}^N u_k^{(k)} \left[ \frac{H_{z}(j, i, k) + H_z(j, i+1, k)}{2} \right]$$

### 4.3 Static Stability ($N^2 \ge 0$) and Water Column Stratification
The perturbation synthesis must not produce static density inversions (heavy water lying over light water). In ROMS vertical index convention:
- $k = 0$ is the seabed ($s = -1$).
- $k = N-1$ is the free surface ($s = 0$).

Static stability requires:
$$\frac{\partial \rho}{\partial z} \le 0 \iff \rho(k) \ge \rho(k+1) \quad \forall k \in [0, N-2]$$

If $\rho(k) < \rho(k+1)$ at $t=0$, the turbulent closure scheme (e.g. GLS, Mellor-Yamada, or KPP) evaluates the gradient Richardson number $\text{Ri} < 0$. This immediately forces the eddy vertical diffusivity $K_v$ and viscosity $A_v$ to their maximum limits ($K_v \sim 10^{-1}\text{ m}^2\text{ s}^{-1}$), destroying the pycnocline via massive instantaneous numerical mixing in the first few hours of integration.

The framework computes in situ density $\rho(S, T, z)$ using the standard non-linear equation of state and applies convective adjustment (Rahmstorf 1993) to any column where a perturbation causes a local inversion, guaranteeing static stability prior to model launch.

---

## 5. Statistical Framework for Signal Extraction in Chaotic Coastal Flow

### 5.1 The Signal vs. Noise Formulation
Let $\mathbf{x}_{\text{farm}}^{(k)}(t)$ and $\mathbf{x}_{\text{ref}}^{(k)}(t)$ denote the state vectors of the $k$-th ensemble member under the wind farm parametrization and the reference configuration, respectively, sharing the identical initial perturbation $\delta \mathbf{x}^{(k)}$ and identical atmospheric/boundary forcing.

The instantaneous difference for realization $k$ decomposes into:

$$\Delta \mathbf{x}^{(k)}(t) = \mathbf{s}(t) + \mathbf{n}^{(k)}(t)$$

where:
- $\mathbf{s}(t) = \mathbb{E}[\mathbf{x}_{\text{farm}}(t) - \mathbf{x}_{\text{ref}}(t)]$ is the deterministic, forced physical signal of the offshore wind farm (e.g. localized velocity deficit, enhanced bottom boundary layer mixing, upstream upwelling/downwelling).
- $\mathbf{n}^{(k)}(t)$ is the stochastic eddy noise realization resulting from internal chaotic dynamics, satisfying $\mathbb{E}[\mathbf{n}^{(k)}(t)] = 0$ with covariance $\mathbf{\Sigma}_{\text{eddy}}$.

The ensemble mean difference:

$$\widehat{\Delta \mathbf{x}}(t) = \frac{1}{K} \sum_{k=1}^K \left( \mathbf{x}_{\text{farm}}^{(k)}(t) - \mathbf{x}_{\text{ref}}^{(k)}(t) \right) = \mathbf{s}(t) + \frac{1}{K}\sum_{k=1}^K \mathbf{n}^{(k)}(t)$$

has variance:

$$\operatorname{Var}\left( \widehat{\Delta \mathbf{x}}(t) \right) = \frac{\mathbf{\Sigma}_{\text{eddy}}}{K}$$

With an ensemble size of $K = 10$, the standard error of the eddy noise is reduced by $\sqrt{10} \approx 3.16$ ($\sim 68\%$ noise reduction) compared to a single deterministic simulation. Over an annual integration, temporal averaging combined with ensemble averaging provides sufficient statistical power to resolve wind farm drag and mixing signals even within the turbulent core of the Norwegian Coastal Current.

---

## 6. References

1. **Baehr, J., & Piontek, R. (2014).** Ensemble initialization of the oceanic component of a coupled model through bred vectors at seasonal-to-interannual timescales. *Geoscientific Model Development*, 7(1), 453–461.
2. **Charney, J. G. (1947).** The dynamics of long waves in a baroclinic westerly current. *Journal of Meteorology*, 4(5), 135–162.
3. **Deng, Z., et al. (2022).** Comparison of Perturbation Strategies for the Initial Ensemble in Ocean Data Assimilation. *Journal of Marine Science and Engineering*, 10(3), 412.
4. **Eady, E. T. (1949).** Long waves and cyclone waves. *Tellus*, 1(3), 33–52.
5. **Evensen, G. (1994).** Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics. *Journal of Geophysical Research: Oceans*, 99(C5), 10143–10162.
6. **Evensen, G. (2003).** The Ensemble Kalman Filter: theoretical formulation and practical implementation. *Ocean Dynamics*, 53(4), 343–367.
7. **Gill, A. E. (1982).** *Atmosphere-Ocean Dynamics*. Academic Press, New York.
8. **Lorenz, E. N. (1963).** Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences*, 20(2), 130–141.
9. **Moore, A. M., et al. (2004).** A comprehensive ocean prediction and analysis system based on the tangent linear and adjoint of a regional ocean model (ROMS). *Ocean Modelling*, 7(1–2), 227–258.
10. **Oke, P. R., et al. (2008).** The Bluelink ocean data assimilation system (BODAS). *Ocean Modelling*, 21(1–2), 46–68.
11. **Pedlosky, J. (1987).** *Geophysical Fluid Dynamics*. Springer-Verlag.
12. **Rahmstorf, S. (1993).** A fast and complete convection scheme for ocean models. *Ocean Modelling*, 101, 9–11.
13. **Rossby, C.-G. (1938).** On the mutual adjustment of pressure and velocity distributions in certain simple current systems, II. *Journal of Marine Research*, 1(3), 239–263.
14. **Shchepetkin, A. F., & McWilliams, J. C. (2005).** The regional oceanic modeling system (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model. *Ocean Modelling*, 9(4), 347–404.
15. **Shu, Y., et al. (2011).** Bred vectors in the South China Sea. *Journal of Oceanography*, 67(4), 487–498.
16. **Toth, Z., & Kalnay, E. (1993).** Ensemble forecasting at NMC: The generation of perturbations. *Bulletin of the American Meteorological Society*, 74(12), 2317–2330.
17. **Toth, Z., & Kalnay, E. (1997).** Ensemble forecasting at NCEP and the breeding method. *Monthly Weather Review*, 125(12), 3297–3319.
