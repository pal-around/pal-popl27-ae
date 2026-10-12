import Palladium.Compiler.Correctness.Lemmas
import JAX.Semantics

/-!
# Marginalize Correctness

`⟦ptExpr⟧ γ ~ μ → ⟦compileMarginalize ... ptExpr⟧ γ ~ evalMarginalize(μ)`

where `evalMarginalize(μ)(σ') = ∑_{v ∈ support(D)} μ(σ'.insert(x, v))`.
-/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL
open Palladium.JAX
open Shape Tensor

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Correctness of the marginalize compilation case.

    `⟦ptExpr⟧ γ ~ μ → ⟦compileMarginalize ... ptExpr⟧ γ ~ evalMarginalize(μ)`

    **Proof chain** (for `σ' : ⟦Σ\x⟧`):
    ```
    ⟦compileMarginalize ...⟧ γ [idx(σ')]
    = ⟦sum(PT, axis)⟧ (γ[PT ↦ ⟦ptExpr⟧ γ]) [idx(σ')]          -- letIn, comment
    = ∑_v ⟦ptExpr⟧ γ [ins_axis(idx(σ'), v)]                    -- sum
    = ∑_v μ(σ'.insert(x, v))                                   -- hypothesis
    = evalMarginalize(μ)(σ')                                    -- definition
    ``` -/
theorem Marginalize.correctness
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext} {Γ : JAX.Context}
    (x : String) (D : Discrete) (h : ⟨x, D⟩ ∈ Δ)
    (μ : ℙ⟦↑Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float))
    (hμ : ptExpr.denote γ ~ μ) :
    (PAL.Stmt.compileMarginalize (L := L) x D h ptExpr).denote γ ~
    (PAL.Stmt.evalMarginalize Δ x D h μ) := by
  cases D with
  | fin n hn =>
    intro env'
    unfold PAL.Stmt.compileMarginalize
    rw [denote_shape_cast]
    simp only [JAX.Expr.denote]
    rw [denote_ty_rec, Ty.Sem_rec_apply]
    simp only [JAX.Expr.denote, Environment.get_cons_eq]
    rw [IdxFn.sumAxes_singleton]
    simp only [castDenoteDtype, cast_eq, IdxFn.sum]
    -- Locate the axis of `x`.
    obtain ⟨idx, hIdxLt, hKeysIdx⟩ := List.mem_iff_getElem.mp (show x ∈ Δ.keys by
      rw [List.mem_keys]; exact ⟨_, h⟩)
    have hAxLt : idx < 𝒞⟦Δ⟧.rank := Δ.toShape_length ▸ hIdxLt
    have hShapeAx : 𝒞⟦Δ⟧[idx]? = some n := Δ.toShape_getElem? x idx hIdxLt hKeysIdx n hn h
    have hn_eq : n = 𝒞⟦Δ⟧[idx]'hAxLt := by
      rw [List.getElem?_eq_getElem hAxLt] at hShapeAx; exact (Option.some.inj hShapeAx).symm
    have hRA : randomAxis Δ.ctx x (.fin n hn) h = idx :=
      Stmt.randomAxis_eq_of_getElem Δ.ctx x (.fin n hn) h idx hIdxLt hKeysIdx Δ.wf
    simp only [PAL.Stmt.evalMarginalize, Discrete.support, List.sum_eq_foldr]
    congr 1
    apply List.ext_getElem
    · simp only [List.length_ofFn, List.length_map]
      calc 𝒞⟦Δ⟧[randomAxis Δ.ctx x (.fin n hn) h]'(hRA ▸ hAxLt)
          = 𝒞⟦Δ⟧[idx]'hAxLt := by congr 1
        _ = n := hn_eq.symm
    · intro j hj1 hj2
      simp only [List.getElem_ofFn, List.getElem_map]
      have hj_n : j < n := by
        simp only [List.length_map, List.length_ofFn] at hj2; exact hj2
      rw [← hμ (env'.insert x (.fin n hn) ⟨j, hj_n⟩ h Δ.wf)]
      congr 1
      apply Indices.ext
      show env'.toIndices.values.insertIdx (randomAxis Δ.ctx x (.fin n hn) h) j = _
      rw [hRA]
      exact (Environment.toIndices_insert Δ.ctx x n hn h Δ.wf
        (Δ.kerase x).wf env' ⟨j, hj_n⟩ idx hIdxLt hKeysIdx).symm

end

end Palladium.Compiler
