/-
Worked examples: writing a PAL program, compiling it to Jay, and running both.

For every valuation `δ` of the program's random variables, each example prints
- the probability `⟦prog⟧ θ δ` that the PAL program assigns to `δ` (`Program.eval`),
- the entry at `δ` of the tensor computed by the compiled Jay program `𝒞⟦prog⟧` in the
  `Tabulated` layout (`JAX.Program.eval`),
- the index-wise denotation `⟦𝒞⟦prog⟧⟧ γ` of the compiled program at `δ`
  (`JAX.Program.denote`).

`Program.correctness` and `Program.correctness_eval` guarantee that the three columns
agree; `joint_correct` and `coin_correct` state these instances, and `#print axioms`
shows that they rest on no axioms beyond Lean's classical basis. Building this file (`lake build Examples`)
prints the tables.

To try a program of your own, copy one of the examples and change its weights,
conditions or contexts. Programs are built from the constructors of `PAL.Expr`,
`PAL.Stmt`, `PAL.Block` and `PAL.Program` (`Palladium/PAL/Syntax.lean`); membership
proofs and proofs that names are distinct close with `by decide`.
-/

import Palladium

namespace Examples

open Palladium.Util Palladium.TensorLib Palladium.PAL Palladium.Compiler

/-- All indices of shape `σ`, in lexicographic order. -/
def allIndices : (σ : Shape) → List (Indices σ)
  | [] => [Indices.nil]
  | n :: σ => (List.finRange n).flatMap fun i =>
      (allIndices σ).map (Indices.cons i.val i.isLt)

/-- For each valuation `δ` of `Δ`: the values of `δ` (newest variable first), the PAL
    semantics, the entry of the evaluated Jay tensor, and the Jay denotation at `δ`. -/
def table {Θ : ValidStaticContext} {Δ : WellFormedRandomContext}
    (prog : Program (Θ := Θ) Δ) (θ : ⟦Θ⟧)
    (γ : Palladium.JAX.Environment (L := Tabulated) 𝒞⟦Θ⟧) :
    List (List Nat × Float × Float × Float) :=
  let jay := 𝒞⟦prog⟧ (L := Tabulated)
  let t := jay.eval γ
  (allIndices Δ.ctx).map fun i =>
    let δ := Indices.toRandomEnvironment Δ.ctx i
    (i.values, prog.eval (θ := θ) δ, t[δ],
      jay.denote (Palladium.JAX.Environment.toValuation γ) δ.toIndices)

/-! ### Example 1: joint sampling

```
x, y ← sample (wpp (1 + x · y))      -- x, y : fin 2
```
The weights of `(x, y) = (0, 0), (0, 1), (1, 0), (1, 1)` are `1, 1, 1, 2`, so the
probabilities are `0.2, 0.2, 0.2, 0.4`.
-/

/-- The random context `x, y`. -/
def xy : WellFormedRandomContext := ⟨[⟨"x", .fin 2⟩, ⟨"y", .fin 2⟩], by decide⟩

/-- The context after sampling `x, y` in the empty context. -/
def Δxy : WellFormedRandomContext := ⟨xy.ctx ++ (∅ : WellFormedRandomContext).ctx, by decide⟩

/-- `x, y ← sample (wpp (1 + x · y))`. -/
def sampleXY : Stmt (Θ := ∅) (∅ : WellFormedRandomContext) Δxy :=
  .sample xy Δxy.wf (.wpp (Expr.add (.float 1.0)
    (Expr.mul (.var "x" (.fin 2) (by decide)) (.var "y" (.fin 2) (by decide)))))

/-- The program: its body samples `x, y` and returns them. -/
def joint : Program (Θ := ∅) Δxy :=
  .mk "joint" (Block.emptyInitialContext sampleXY (List.Perm.refl _))

-- Columns: `[x, y]`, `⟦joint⟧`, the evaluated Jay tensor at `δ`, `⟦𝒞⟦joint⟧⟧` at `δ`.
#eval table joint default default

-- The tensor computed by the compiled program (nested vectors, indexed by `x`, then `y`).
#eval (𝒞⟦joint⟧ (L := Tabulated)).eval default

/-- Correctness for `joint`, an instance of `Program.correctness_eval`. -/
theorem joint_correct (δ : RandomEnvironment Δxy.ctx) :
    ((𝒞⟦joint⟧ (L := Tabulated)).eval default)[δ] = joint.eval (θ := default) δ :=
  Program.correctness_eval (L := Tabulated) joint default (fun _ _ h => nomatch h) δ

/-! ### Example 2: a parameter, conditioning and marginalization

```
param p : float                                   -- p = 0.3
x ← sample (wpp (if x = 1 then p else 1 - p))     -- x : fin 2
y ← sample (wpp (1 + x · y))                      -- y : fin 3
condition (y ≠ 0)
marginalize x
```
Given `x = 0`, `y` is uniform; given `x = 1`, `y = 0, 1, 2` have weights `1, 2, 3`.
Conditioning keeps the mass where `y ≠ 0` without renormalizing, so the result is
`0, 0.7/3 + 0.3 · 2/6, 0.7/3 + 0.3 · 3/6`, that is, `0, 0.333333, 0.383333`.
-/

/-- The parameter context `p : float`. -/
def Θp : ValidStaticContext := ValidStaticContext.ofList [⟨"p", .float⟩] (by decide) (by decide)

/-- The parameter valuation `p = 0.3`. -/
def θp : ⟦Θp⟧ := (default : StaticEnvironment []).cons "p" .float (0.3 : Float)

/-- `p = 0.3` as a Jay rank-0 tensor. -/
def pTensor : Palladium.JAX.Ty.denote (L := Tabulated) (.tensor [] .float) :=
  Tensor.ofTabulated (L := Tabulated) (σ := []) (0.3 : Float)

/-- The Jay environment of the compiled program: `p` is the tensor `0.3`. -/
def γp : Palladium.JAX.Environment (L := Tabulated) 𝒞⟦Θp⟧ :=
  (default : Palladium.JAX.Environment (L := Tabulated) []).cons "p" (.tensor [] .float) pTensor

/-- The Jay environment represents the parameter valuation. -/
theorem γp_represents : Palladium.JAX.Environment.RepresentsParams (Γ := []) γp θp := by
  intro x T h
  cases h with
  | head => rfl
  | tail _ h => cases h

/-- The random variable `x`. -/
def vx : WellFormedRandomContext := ⟨[⟨"x", .fin 2⟩], by decide⟩
/-- The random variable `y`. -/
def vy : WellFormedRandomContext := ⟨[⟨"y", .fin 3⟩], by decide⟩

/-- The context `x`. -/
def Δx : WellFormedRandomContext := ⟨vx.ctx ++ (∅ : WellFormedRandomContext).ctx, by decide⟩
/-- The context `y, x`. -/
def Δyx : WellFormedRandomContext := ⟨vy.ctx ++ Δx.ctx, by decide⟩

/-- `x ← sample (wpp (if x = 1 then p else 1 - p))`. -/
def sampleX : Stmt (Θ := Θp) (∅ : WellFormedRandomContext) Δx :=
  .sample vx Δx.wf (.wpp (.ite
    (Expr.eq (.var "x" (.fin 2) (by decide)) (Expr.fin (hn := by decide) (1 : Fin 2)))
    (.param "p" .float (by decide))
    (Expr.sub (.float 1.0) (.param "p" .float (by decide)))))

/-- `y ← sample (wpp (1 + x · y))`. -/
def sampleY : Stmt (Θ := Θp) Δx Δyx :=
  .sample vy Δyx.wf (.wpp (Expr.add (.float 1.0)
    (Expr.mul (.var "x" (.fin 2) (by decide)) (.var "y" (.fin 3) (by decide)))))

/-- `condition (y ≠ 0)`. -/
def conditionY : Stmt (Θ := Θp) Δyx Δyx :=
  .condition (Expr.ne (.var "y" (.fin 3) (by decide)) (Expr.fin (hn := by decide) (0 : Fin 3)))

/-- `marginalize x`. -/
def marginalizeX : Stmt (Θ := Θp) Δyx (Δyx.kerase "x") :=
  .marginalize "x" (.fin 2) (by decide)

/-- The program: its body runs the four statements and returns `y`. -/
def coin : Program (Θ := Θp) (Δyx.kerase "x") :=
  .mk "coin" (Block.emptyInitialContext
    (.seq sampleX (.seq sampleY (.seq conditionY marginalizeX))) (List.Perm.refl _))

-- Columns: `[y]`, `⟦coin⟧ θp`, the evaluated Jay tensor at `δ`, `⟦𝒞⟦coin⟧⟧ γp` at `δ`.
#eval table coin θp γp

/-- Correctness for `coin`, an instance of `Program.correctness_eval`. -/
theorem coin_correct (δ : RandomEnvironment (Δyx.kerase "x").ctx) :
    ((𝒞⟦coin⟧ (L := Tabulated)).eval γp)[δ] = coin.eval (θ := θp) δ :=
  Program.correctness_eval (L := Tabulated) coin γp γp_represents δ

#print axioms joint_correct
#print axioms coin_correct

end Examples
