import Palladium.Compiler.Correctness.WellFormed
import Palladium.Compiler.Correctness.RepresentsPMF
import Palladium.Compiler.Correctness.Correct
import Palladium.Compiler.Correctness.Lemmas
import Palladium.Compiler.Correctness.Expr

import Palladium.Compiler.Correctness.Sample
import Palladium.Compiler.Correctness.Marginalize
import Palladium.Compiler.Correctness.Stmt
import Palladium.Compiler.Correctness.Program
import Palladium.Compiler.Correctness.Block

/-!
# Compiler Correctness

Aggregate imports for the correctness proof modules. The proof establishes that the
denotation of a compiled PAL program represents the PMF of the program.

Import chain: `WellFormed (RepresentsParams) → RepresentsPMF → Correct → Lemmas → Expr →
Marginalize → Sample → Condition → Stmt → Block → Program`

The correctness property is stated as a commutative diagram:

```
  PAL.Expr/Stmt/Program ────────── compile ────────────▶ JAX.Expr/Program
         │                                                    │
       ⟦·⟧                                                  ⟦·⟧
         │                                                    │
         ▼                                                    ▼
  PMF Δ                    ◀── represents ──▶  I(𝒞⟦Δ⟧) → Float
  (⟦Δ⟧ → Float)
```

A Jay expression denotes, for each valuation `γ`, an indexing function over its shape.
Each axis of the shape corresponds to a random variable, and the indices `idx(δ)` of a
random environment `δ` select the entry for `δ`:

* expressions: `⟦𝒞⟦e⟧⟧ γ [proj(idx(δ))] = ↑(⟦e⟧ θ δ)` (`Expr.correctness`);
* statements and blocks: `⟦e⟧ γ ~ μ → ⟦𝒞⟦s⟧ e⟧ γ ~ ⟦s⟧ θ μ` (`Stmt.correctness`,
  `Block.correctness`);
* programs: `⟦𝒞⟦prog⟧⟧ γ ~ ⟦prog⟧ θ` (`Program.correctness`).

`Program.correctness_eval` states the program theorem for the evaluation of Jay in a
tensor layout, using the adequacy of `eval` (`JAX.Adequacy`).
-/
