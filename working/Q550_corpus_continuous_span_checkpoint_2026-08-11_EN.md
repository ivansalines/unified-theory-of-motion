# Q=550 — Corpus Continuous-Span Checkpoint

**Checkpoint date:** 2026-08-11  
**Scope:** frozen natural-cubic-spline linear cocycle at Q=550  
**Status:** continuous endpoint distinction certified across the full dense-start span.

---

## 1. Final status

The dense grid contains 81 start nodes, hence 80 inter-node cells.

Using the bidirectional cocycle-reset corpus architecture v7.1 with:

- `half-step sweep = [512]`
- `H/K residual subdivisions = 8`
- `local paired pieces = 8`

all cells `0..79` pass continuously:

\[
\boxed{80/80\ \text{cells PASS}}
\]

Every cell satisfies:

\[
badL=-1,\qquad badR=-1.
\]

The worst continuous lower margin over the complete 80-cell span is:

\[
\boxed{\min lower = +7.319723\times10^{-4}}
\]

at **cell 49**.

Final classification:

`BIDIRECTIONAL COCYCLE-RESET CORPUS PILOT PASSED AT LARGEST HALF-STEP RESOLUTION`

Therefore, for the frozen natural-cubic-spline linear cocycle at Q=550, the endpoint distinction is no longer certified only at the 81 dense nodes: it is certified **continuously over the complete interval** from the first to the last dense start.

---

## 2. Geometry that unlocked the proof

The long endpoints are nested:

\[
P_{\rm line}=0.22777777777777777,
\]

\[
P_{\rm oriented}=0.45416666666666666,
\]

\[
2P_{\rm line}=0.45555555555555555.
\]

Hence:

\[
\Delta P
=
2P_{\rm line}-P_{\rm oriented}
=
0.001388888888888884
\approx \frac{1}{720}.
\]

With

\[
G_{\rm edge}(s)
=
\Phi(s+2P_{\rm line},\,s+P_{\rm oriented}),
\]

we have the exact nesting identity:

\[
T_{2P}(s)=G_{\rm edge}(s)T_O(s),
\]

and therefore:

\[
\boxed{
D(s)
=
T_O(s)-T_{2P}(s)
=
(I-G_{\rm edge}(s))T_O(s)
}.
\]

This yields the central common-body factorization:

\[
\boxed{D=C\,T,\qquad C=I-G_{\rm edge},\quad T=T_O.}
\]

The long-body uncertainty is charged only once and is attenuated by the small edge factor, instead of comparing two independent long transports.

---

## 3. Path to closure

### A. Relative-defect microbox

The successful transport validator uses:

\[
R=U'-BU,\qquad X=U(I+E),
\]

with:

\[
E'=-U^{-1}R(I+E).
\]

A floating-point knot-touch bug in the local cubic envelope was fixed in pilot v1.4 by skipping every genuinely empty interval intersection:

```python
if segment_right < segment_left:
    continue
```

This removed the false `lower endpoint exceeds upper` failure at low-index starts without changing bounds on true overlaps.

### B. v4 — local endpoint nesting

The endpoint-nesting/common-body architecture was first certified on cell 67→68.

Instead of paying for two independent long-body errors, it used the exact small terminal cap.

**Structural lesson:** preserve common-body correlation before tightening numerics.

### C. v5 — all 81 dense start nodes

The direct dense-start audit independently validated every start.

Results:

- `m=8`: 81/81 pass
- `m=16`: 81/81 pass
- `m=32`: 81/81 pass

At `m=32`:

\[
\min lower=+1.064161\times10^{-2},
\]

\[
\max R=4.120767\times10^{-4}.
\]

Thus all 81 dense start nodes were certified, but continuous inter-node coverage was still open.

### D. v6 — first continuous corpus attempt

The start-coordinate dynamics were written as:

\[
T'=B_1T-TB_0,
\]

\[
C'=(B_1-B_2)+B_2C-CB_1.
\]

This correctly retained the small paired difference \(B_1-B_2\), but the first enclosure still widened too much.

A subdivision scan showed:

\[
rq:
0.3633
\to
0.1815
\to
0.09076
\to
0.04539
\to
0.02271
\to \cdots
\]

so the paired \(B_1-B_2\) radius behaved correctly, approximately like \(1/M\).

The obstruction was elsewhere.

### E. v6.1 — full-path scaling diagnosis

The full-path test showed that refining the corpus tube moved the first failure deeper into cell 39, but the long-body tube radius remained the dominant source of inflation.

The key diagnosis was:

**physical motion of \(T\) was being partly paid as uncertainty width.**

This was numerical memory, not evidence of a physical singularity.

### F. v7 — one-sided cocycle reset

Instead of chaining the Euler/Picard endpoint ball, v7 reconstructed each slab from the original rigorous anchor through short cocycle transports.

For left anchor \(a\):

\[
H_0(x)=\Phi(x,a),
\]

\[
H_1(x)=\Phi(x+P_O,a+P_O),
\]

\[
H_2(x)=\Phi(x+2P_{\rm line},a+2P_{\rm line}).
\]

Then:

\[
\boxed{
T(x)=H_1(x)T(a)H_0(x)^{-1}
}
\]

and:

\[
\boxed{
G(x)=H_2(x)G(a)H_1(x)^{-1}.
}
\]

The local Picard tube covers only one slab and is discarded at the endpoint.

The cocycle centers matched independently reconstructed right-node centers down to roughly \(10^{-11}\)–\(10^{-13}\), confirming that the coordinate was correct.

But one anchor still had to carry the short-transport enclosure across the whole cell.

### G. v7.1 — bidirectional cocycle reset

The final articulation uses both rigorous v5 endpoint anchors.

Left half:

\[
T(x)=H_1T_LH_0^{-1},
\qquad
G(x)=H_2G_LH_1^{-1}.
\]

Right half, with forward transports \(K_j\) from \(x\) to the right endpoint:

\[
\boxed{
T(x)=K_1^{-1}T_RK_0
}
\]

\[
\boxed{
G(x)=K_2^{-1}G_RK_1.
}
\]

Thus each endpoint covers only its nearest half-cell:

\[
\boxed{
\text{left anchor}
\rightarrow
\text{midpoint}
\leftarrow
\text{right anchor}
}
\]

No local Picard endpoint uncertainty is inherited by the next slab.

This first closed cell 39:

- `Nhalf=512`: `min=+7.460753e-04`, `pass=True`
- `Nhalf=1024`: `min=+4.868956e-03`, `pass=True`

Then all seven deliberately hard cells passed at 512.

Finally all 80 cells passed at 512.

---

## 4. Full-span result

Architecture:

\[
D=C\,T,
\qquad
C=I-G_{\rm edge}.
\]

Continuous cover:

\[
[s_0,s_{80}]
=
\bigcup_{i=0}^{79}[s_i,s_{i+1}].
\]

At `Nhalf=512`:

\[
\boxed{
\forall i\in\{0,\dots,79\},
\quad
lower_i(s)>0
\quad\text{throughout the certified cell cover.}
}
\]

Global worst cell:

\[
\boxed{
i=49,\qquad
\min lower=+7.319723\times10^{-4}.
}
\]

Hence:

\[
\boxed{
81\ \text{rigorous nodes}
\;\longrightarrow\;
80\ \text{rigorous inter-node cells}
\;\longrightarrow\;
\text{one continuous certified span}.
}
\]

---

## 5. Points that must not be lost

1. **Do not return to independent long-body differences.**  
   They destroy common-mode correlation and create artificial radius growth.

2. **Do not interpret rotation or physical motion as uncertainty.**  
   The cocycle center must carry the physical motion; the radius must carry only validation uncertainty.

3. **Keep the small endpoint cap explicit.**  
   \[
   \Delta P\approx1/720
   \]
   is not incidental numerical smallness; it is the exact nesting geometry behind the successful factorization.

4. **Use both endpoint anchors for continuous cells.**  
   The bidirectional reset is what turned the node certificate into a continuous-span certificate.

5. **Do not change the v7.1 architecture for this layer unless a new inconsistency appears.**  
   It has now passed all 80 cells.

---

## 6. Scope boundary still open

This result is rigorous only for the:

**frozen natural-cubic-spline linear cocycle at Q=550.**

It does **not** yet interval-validate the cached nonlinear orbit as an exact trajectory of the nonlinear system.

Therefore the next genuinely higher-level target is:

\[
\boxed{
\text{exact nonlinear-orbit interval validation}
}
\]

or an equivalent rigorous bridge proving that the spline/orbit cache encloses an exact nonlinear trajectory.

The current continuous-span result should remain untouched as a completed lower layer while that higher layer is investigated.

---

## 7. Key files at this checkpoint

- `test_support_q550_relative_defect_microbox_pilot_v1_4.py`
- `test_support_q550_endpoint_nesting_dense_span_audit_v5.py`
- `test_support_q550_corpus_inter_node_bridge_v6.py`
- `test_support_q550_corpus_inter_node_bridge_v6_1.py`
- `test_support_q550_cocycle_reset_corpus_bridge_v7.py`
- `test_support_q550_bidirectional_cocycle_reset_v7_1.py`
- `support_q550_endpoint_nesting_dense_span_0_80_report.txt`
- `support_q550_bidirectional_full80_v7_1_report.txt`
- `support_q550_bidirectional_full80_v7_1.json`
- `support_q550_bidirectional_full80_v7_1_summary.csv`
- `support_q550_bidirectional_full80_v7_1_steps.csv`

---

## 8. Resume sentence

If this work is resumed after a pause, start from:

> **At Q=550 the frozen-spline linear cocycle endpoint distinction is continuously certified across all 80 dense inter-node cells using the bidirectional cocycle-reset common-body architecture v7.1; the next open layer is exact nonlinear-orbit interval validation.**

This is the current state of the corpus.
