import Palladium.Compiler.Correctness.RepresentsPMF
import Palladium.PAL.Semantics

/-!
# Compiler Correctness Predicates

The correctness predicates state what it means for each compilation unit to be correct.
They are stated for the index-wise semantics `⟦·⟧` of Jay (`JAX.Expr.denote`), with a
valuation `γ` of the Jay context.
-/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL
open Palladium.JAX
open Tensor

section

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-!
## Expression Correctness

The compiled expression, read at the indices of `δ` projected to its shape, gives the
value of the PAL expression at `δ`.
-/

/-- `γ ~ θ → ⟦𝒞⟦e⟧⟧ γ [proj(idx(δ))] = ↑(⟦e⟧ θ δ)`. -/
def Expr.Correct
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext} {T : PAL.Ty}
    {Γ : JAX.Context}
    (e : PAL.Expr (Θ := ↑Θ) ↑Δ T)
    {θ : ⟦Θ⟧}
    (δ : ⟦Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧)) : Prop :=
  γ ~ θ →
  let ⟨_, jexpr, hbc⟩ := 𝒞⟦e⟧ (L := L) (Γ := Γ)
  jexpr.denote γ (δ.toIndices.proj hbc) = ↑(δ |> ⟦e⟧ (θ := θ))

/-!
## Statement Correctness

A PAL statement takes a probability tensor expression and produces a new one.
Correctness means: if the input expression represents μ, then the output represents ⟦s⟧(μ).
-/

/-- `γ ~ θ → ⟦e⟧ γ ~ μ → ⟦𝒞⟦s⟧ e⟧ γ ~ ⟦s⟧ θ μ`. -/
def Stmt.Correct
    {Θ : ValidStaticContext} {Δ Δ' : WellFormedRandomContext}
    {Γ : JAX.Context}
    (s : PAL.Stmt (Θ := Θ) Δ Δ')
    {θ : ⟦Θ⟧}
    (μ : ℙ⟦Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (jexpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float)) : Prop :=
  γ ~ θ →
  jexpr.denote γ ~ μ →
  let jexpr' := jexpr |> 𝒞⟦s⟧
  let μ' := μ |> ⟦s⟧ (θ := θ)
  jexpr'.denote γ ~ μ'

/-!
## Block Correctness

A block takes an input PMF and produces an output PMF.
Correctness means: if the input expression represents μ, then the output represents ⟦blk⟧(μ).
-/

/-- `γ ~ θ → ⟦e⟧ γ ~ μ → ⟦𝒞⟦blk⟧ e⟧ γ ~ ⟦blk⟧ θ μ`.
    The block's output and PMF are both over the combined context
    `↑Δ_ret ++ ↑Δ'` (return ahead of carry). -/
def Block.Correct
    {Θ : ValidStaticContext} {Δ Δ' Δ_ret : WellFormedRandomContext}
    {Γ : JAX.Context}
    (blk : PAL.Block (Θ := Θ) Δ Δ' Δ_ret)
    {θ : ⟦Θ⟧}
    (μ : ℙ⟦Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (jexpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧)
        (JAX.Ty.tensor 𝒞⟦Δ⟧ .float)) : Prop :=
  γ ~ θ →
  jexpr.denote γ ~ μ →
  let jexpr' := PAL.Block.compile blk jexpr
  let μ' := PAL.Block.eval (θ := θ) blk μ
  jexpr'.denote γ ~ μ'


/-!
## Program Correctness

The top-level correctness theorem: the denotation of the compiled program represents the
PMF of the PAL program.
-/

/-- `γ ~ θ → ⟦𝒞⟦prog⟧⟧ γ ~ ⟦prog⟧ θ`, that is, `∀ δ, ⟦𝒞⟦prog⟧⟧ γ [idx(δ)] = ⟦prog⟧ θ δ`. -/
def Program.Correct
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext}
    (prog : PAL.Program (Θ := Θ) Δ)
    {θ : ⟦Θ⟧}
    (γ : JAX.Valuation 𝒞⟦Θ⟧) : Prop :=
  γ ~ θ ‖ [] →
  ⟦𝒞⟦prog⟧ (L := L)⟧ γ ~ ⟦prog⟧ θ

end

end Palladium.Compiler
