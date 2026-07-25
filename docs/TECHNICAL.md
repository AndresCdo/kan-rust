# KAN Core Technical Contract

This document describes the validated fixed-grid KAN core introduced in 0.2.0.
It is intentionally narrower than the original project claims.

## Scope

Implemented and tested:

- canonical degree-three B-spline edge functions;
- sequential KAN forward propagation;
- full multilayer reverse-mode MSE gradients;
- deterministic initialization and full-batch SGD;
- validated, versioned JSON persistence.

Deferred: regularization, grid refinement, pruning, symbolic fitting, and
legacy MLP/vector/matrix remediation.

## Spline convention

For `G >= 1` uniform intervals over `[a, b]` and degree `p = 3`, let
`h = (b - a) / G`. The knot vector is exterior extended:

```text
t_i = a + (i - p) * h, for i = 0 .. G + 2p
```

There are `G + 2p + 1 = G + 7` knots and `G + p = G + 3` coefficients. This
is the convention used by the KAN paper and pykan; it is not the repeated-end
clamped-knot convention.

The Cox-de Boor recurrence is:

```text
B[i, 0](x) = 1 when t[i] <= x < t[i + 1], otherwise 0
B[i, p](x) = (x - t[i]) / (t[i + p] - t[i]) * B[i, p - 1](x)
           + (t[i + p + 1] - x) / (t[i + p + 1] - t[i + 1]) * B[i + 1, p - 1](x)
```

Zero-denominator terms contribute zero. For inputs outside the extended
interval `[t[0], t[last])`, the spline basis and its derivative are zero; the
SiLU residual branch remains active.

## Forward and reverse pass

Every edge is

```text
phi(x) = w_base * SiLU(x) + w_spline * sum_r c[r] * B[r, 3](x)
```

Nodes sum incoming edge outputs. Given an incoming node adjoint `delta`, the
parameter and input derivatives are:

```text
dL/dw_base   = delta * SiLU(x)
dL/dw_spline = delta * spline(x)
dL/dc[r]     = delta * w_spline * B[r, 3](x)
dL/dx        = delta * (w_base * SiLU'(x) + w_spline * spline'(x))
```

The trainer caches node values, walks layers from output to input, accumulates
all edge gradients, and applies one simultaneous full-batch SGD update.

Initialization is reproducible across platforms through the in-tree SplitMix64
sequence. In layer/output/input traversal order, base weights are sampled from
a signed uniform distribution and divided by the square root of the input
width, spline weights start at `1`, and coefficients use signed uniform noise
scaled by `0.01`. The seed defaults to `0` and can be set with `with_seed`.

## Validation and persistence

`KanConfig` requires at least two non-zero widths, non-zero grid intervals,
and a finite domain with `min < max`. Forward and training validate exact
dimensions, nonempty equal-sized batches, finite values, and positive finite
learning rates. P0 rejects configurations above 250,000 trainable parameters,
JSON source-state documents above 8 MiB, and documents above a bounded JSON
node budget before typed deserialization or model allocation.

Persistence writes only source state in a `kan-rust` format-version `1`
envelope: shape, degree, intervals, domain, and per-edge parameters. Knots,
caches, and gradients are reconstructed rather than trusted from JSON. The
loader rejects unknown fields, unsupported versions, malformed dimensions,
incorrect coefficient counts, duplicate fields, excessive JSON complexity, and
non-finite parameters. The CLI checks metadata on the opened regular-file
handle and retains an in-band limit-plus-one read. Hostile special files that
can block during `open` are outside the portable CLI threat model.

## Verification strategy

Tests cover exterior knot counts, basis partition of unity, left-boundary
values, coefficient/base/spline-weight finite differences across multiple
layers, deterministic `[2, 5, 1]` training loss reduction, invalid inputs, and
persistence round trips. Gradient checks use interior points away from knots;
they are not claims about nonsmooth future features such as pruning.

## References

- Liu et al., [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/html/2404.19756v5), Eq. 2.5 and 2.10-2.16.
- [pykan `spline.py`](https://github.com/KindXiaoming/pykan/blob/master/kan/spline.py).
- [SciPy `BSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html), used only with an identical knot vector when comparing values.
