import Palladium.Compiler.Correctness.Lemmas

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL

section

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L] [Tensor.LawfulOfTabulated L]

/-!
## Expression Correctness

A PAL expression compiles to a Jay expression whose denotation, read at the indices of a
random environment `δ` projected to its shape, is the value of the PAL expression at `δ`.

`∀ e, ⟦𝒞⟦e⟧⟧ γ [proj(idx(δ))] = ↑(⟦e⟧ θ δ)`
Cases: fin, float, param, var, uop, bop, ite.
-/

/-- `γ ~ θ → ⟦𝒞⟦e⟧⟧ γ [proj(idx(δ))] = ↑(⟦e⟧ θ δ)`.
    Proved by case split on e: fin, float, param, var, uop, bop, ite. -/
theorem Expr.correctness
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext} {T : PAL.Ty}
    {Γ : JAX.Context}
    (e : PAL.Expr (Θ := ↑Θ) ↑Δ T)
    {θ : ⟦Θ⟧}
    (δ : ⟦Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    :
    Expr.Correct (L := L) (θ := θ) e δ γ := by
  unfold Expr.Correct
  intro hpc
  match e with
  | .fin i =>
    dsimp only [PAL.Expr.compile, JAX.Expr.denote]
    rw [Tensor.index_nil]
    exact Tensor.scalar_ofTabulated i
  | .float r =>
    dsimp only [PAL.Expr.compile, JAX.Expr.denote]
    rw [Tensor.index_nil]
    exact Tensor.scalar_ofTabulated r
  | .param x T h =>
    dsimp only [PAL.Expr.compile, JAX.Expr.denote]
    rw [Indices.eq_nil (δ.toIndices.proj _)]
    exact hpc x T h
  | .var x D h =>
    cases D with
    | fin n hn =>
      exact enumExpr_denote h Δ.wf δ γ
  | .uop op e' hv =>
    have ih := Expr.correctness e' δ γ hpc
    dsimp only at ih
    dsimp only [PAL.Expr.compile, PAL.Expr.eval, JAX.Expr.denote]
    rw [ih]
    cases hv <;> rename_i h <;> cases h <;> rfl
  | .bop op a b hv =>
    have ih_a := Expr.correctness a δ γ hpc
    have ih_b := Expr.correctness b δ γ hpc
    dsimp only at ih_a ih_b
    dsimp only [PAL.Expr.compile, PAL.Expr.eval, JAX.Expr.denote]
    rw [Indices.proj_proj _ _ (a.compile (L := L) (Γ := Γ)).broadcastable,
      Indices.proj_proj _ _ (b.compile (L := L) (Γ := Γ)).broadcastable, ih_a, ih_b]
    cases hv <;> rename_i h <;> cases h <;> rfl
  | .ite c t e' =>
    have ih_c := Expr.correctness c δ γ hpc
    have ih_t := Expr.correctness t δ γ hpc
    have ih_e := Expr.correctness e' δ γ hpc
    dsimp only at ih_c ih_t ih_e
    rename_i T₁ T₂
    -- Case split on PAL types to eliminate the `Ty.compile_promote` transport
    -- (for concrete `T₁`, `T₂`, `Ty.compile_promote` is `rfl` and the `▸` disappears).
    cases T₁ <;> (try (rename_i d; cases d)) <;>
      cases T₂ <;> (try (rename_i d; cases d))
    all_goals (
      revert ih_c ih_t ih_e
      dsimp only [PAL.Expr.compile, PAL.Expr.eval, JAX.Expr.denote]
      intro ih_c ih_t ih_e
      rw [Indices.proj_proj _ _ (c.compile (L := L) (Γ := Γ)).broadcastable,
        Indices.proj_proj _ _ (t.compile (L := L) (Γ := Γ)).broadcastable,
        Indices.proj_proj _ _ (e'.compile (L := L) (Γ := Γ)).broadcastable,
        ih_c, ih_t, ih_e]
      rfl
    )

end

end Palladium.Compiler
