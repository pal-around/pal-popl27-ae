/-
The paper's results.

This file states the paper's three correctness theorems and prints the axioms each one
depends on. Building it (`lake build Results`) re-checks every proof.
The `#print axioms` output lets a reviewer confirm that no result depends on `sorryAx`
(no admitted proofs) and see the exact axiom basis: Lean's classical basis (`propext`,
`Classical.choice`, `Quot.sound`). The development declares no axioms.

`Extras.lean` states the results of the development beyond these theorems.

The last section evaluates a few small Jay and PAL terms as sanity checks.
-/

import Palladium

open Palladium.Compiler

/-!
### Theorem "Program compilation preserves semantics"

For a PAL program `prog` and a valuation `γ` of its parameters that represents `θ`,
indexing the denotation of the compiled Jay program at the indices of any random
environment `δ` gives the probability that the PAL program assigns to `δ`:
`⟦𝒞⟦prog⟧⟧ γ [idx(δ)] = ⟦prog⟧ θ δ`.
-/
#check @Program.correctness
#print axioms Program.correctness

/-!
### Theorem "Statement and block compilation preserves semantics"

For a Jay expression `e` whose denotation represents the PMF `μ`, the denotation of the
compiled statement (block) applied to `e` represents the PMF of the statement (block)
applied to `μ`: `⟦e⟧ γ ~ μ → ⟦𝒞⟦s⟧ e⟧ γ ~ ⟦s⟧ θ μ`.
-/
#check @Stmt.correctness
#print axioms Stmt.correctness

#check @Block.correctness
#print axioms Block.correctness

/-!
### Theorem "Expression compilation preserves semantics"

The denotation of the compiled expression, read at the indices of a random environment
`δ` projected to its shape, is the value of the PAL expression at `δ`:
`⟦𝒞⟦e⟧⟧ γ [proj(idx(δ))] = ↑(⟦e⟧ θ δ)`.
-/
#check @Expr.correctness
#print axioms Expr.correctness

/-!
### Sanity checks

Each `#eval` prints its result during the build; the expected value is stated in the
comment above it.
-/

namespace Sanity

open Palladium.Util Palladium.TensorLib Palladium.PAL Palladium.Compiler

-- `0 / 0 = 0`.
#eval Float.div0 0 0

/-- The paper's projection example: for `σ = [3, 1, 4]` broadcast to `σₒ = [2, 3, 5, 4]`,
    `proj(i) = (i₁, 0, i₃)`; here `i = (1, 2, 4, 3)`. -/
example :
    ((Indices.cons 1 (by decide) (Indices.cons 2 (by decide) (Indices.cons 4 (by decide)
      (Indices.cons 3 (by decide) Indices.nil))) : Indices [2, 3, 5, 4]).proj
      (σ := [3, 1, 4]) (by decide)).values = [2, 0, 3] := rfl

/-- `[1.0, 2.0] + 3.0` in Jay. -/
def jayAdd : Palladium.JAX.Expr (L := Tabulated) [] (Palladium.JAX.Ty.tensor [2] .float) :=
  .bop .add (.array .float (Tensor.ofTabulated (σ := [2]) #v[1.0, 2.0]))
    (.array .float (Tensor.ofTabulated (σ := []) (3.0 : Float))) .add_ff (by decide)

/-- The index `[1]`. -/
def i1 : Indices [2] := Indices.cons 1 (by decide) Indices.nil

-- `⟦[1.0, 2.0] + 3.0⟧ [1] = 5.0`.
#eval jayAdd.denote (fun _ _ h => nomatch h) i1

-- The tensor computed by `eval` holds the same entry.
#eval Tensor.index (L := Tabulated) (σ := [2]) (α := Float) (jayAdd.eval (fun _ _ h => nomatch h)) i1

/-- Two binary random variables `x, y`. -/
def xy : WellFormedRandomContext := ⟨[⟨"x", .fin 2⟩, ⟨"y", .fin 2⟩], by decide⟩

/-- The context `x, y` after sampling from the empty context. -/
def xyΔ : WellFormedRandomContext := ⟨xy.ctx ++ (∅ : WellFormedRandomContext).ctx, by decide⟩

/-- `x, y ← sample (wpp w)` with a constant weight `w`. -/
def sampleConst (w : Float) : Stmt (Θ := ∅) (∅ : WellFormedRandomContext) xyΔ :=
  .sample xy xyΔ.wf (.wpp (.float w))

/-- The valuation `x = 1, y = 0`. -/
def δ10 : RandomEnvironment xyΔ.ctx :=
  Indices.toRandomEnvironment _ ⟨[1, 0], rfl, by decide⟩

-- With weight `1.0`, each of the four valuations has mass `0.25`.
#eval (sampleConst 1.0).eval (θ := default) (fun _ => 1.0) δ10

-- With weight `0.0`, normalization is `0 / 0 = 0`.
#eval (sampleConst 0.0).eval (θ := default) (fun _ => 1.0) δ10

/-- `condition false` on the empty context. -/
def conditionFalse : Stmt (Θ := ∅) (∅ : WellFormedRandomContext) ∅ :=
  .condition (Expr.fin (hn := by decide) (0 : Fin 2))

-- `condition false` sets the mass to zero, also when the mass is NaN.
#eval conditionFalse.eval (θ := default) (fun _ => (0.0 : Float) / 0.0) default

end Sanity
