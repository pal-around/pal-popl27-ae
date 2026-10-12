import Palladium.Compiler.Correctness.Expr
import Palladium.Compiler.Correctness.Lemmas
import Palladium.Compiler.Compile.Basics
import Palladium.Compiler.Compile.Condition

/-!
# Condition Correctness

`⟦ptExpr⟧ γ ~ μ → ⟦compileCondition c ptExpr⟧ γ ~ Stmt.evalCondition c μ`

where `evalCondition` keeps the mass where `c` holds and sets it to zero elsewhere.

## `weakenMask` in `compileCondition`

The compiled condition is built in context `Γ ++ 𝒞⟦Θ⟧`, and `compileCondition` forms
`select(c, probabilityTensor, 0)` inside `letIn probabilityTensor ptExpr …`, where
`probabilityTensor` is bound in `⟨probabilityTensor, ptTy⟩ :: (Γ ++ 𝒞⟦Θ⟧)`.
`JAX.Expr` is indexed by that context, so `PAL.Stmt.weakenMask` re-witnesses every
`var` in the condition for the extended context (same names and semantics; new
membership proofs).

The lifted expression must be `letIn`-free (`hasLetIn = false`): `weakenMask` implements
lifting for that fragment (see the `weakenMask` docstring in `Compile/Condition.lean`).

`weakenMask_denote` proves the denotation commutes with that lift: extending the valuation
with the value bound by `letIn` does not change the condition's denotation. Freshness
`probabilityTensor ∉ (Γ ++ 𝒞⟦Θ⟧).keys` ensures the new binding does not shadow a name
the condition uses.
-/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL
open Palladium.JAX
open Shape Tensor

section

/-- If `probabilityTensor` is not a key of `Γ ++ 𝒞⟦Θ⟧`, no binding uses that name. -/
theorem ne_probabilityTensor_of_probabilityTensor_not_in_keys {Γ : JAX.Context} {Θ : ValidStaticContext}
    (hpt : probabilityTensor ∉ (Γ ++ 𝒞⟦Θ⟧).keys)
    {n : String} {τ : JAX.Ty} (hmem : ⟨n, τ⟩ ∈ Γ ++ 𝒞⟦Θ⟧) : n ≠ probabilityTensor := by
  intro heq; subst heq
  exact hpt (List.mem_keys.mpr ⟨_, hmem⟩)

end

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Helper for `weakenMask_denote` on `sum`: after unfolding `weakenMask` on `sum`,
    the denotation reduces to the child case `ih` (per `η` in `JAX.Expr.denote`). -/
private theorem denote_sum_weakenMask
    {Γ : JAX.Context} {σ : Shape} {η : DType} (dummy : Σ _ : String, JAX.Ty)
    (e' : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ η)) (axes : List (Fin σ.rank)) (hNodup : axes.Nodup)
    (h : (JAX.Expr.sum e' axes hNodup).hasLetIn = false)
    (γ : JAX.Valuation Γ) (v : dummy.snd.Sem)
    (ih :
      (PAL.Stmt.weakenMask (L := L) dummy e' h).denote (γ.cons dummy.fst dummy.snd v) = e'.denote γ) :
    (JAX.Expr.sum (PAL.Stmt.weakenMask (L := L) dummy e' h) axes hNodup).denote (γ.cons dummy.fst dummy.snd v) =
      (JAX.Expr.sum e' axes hNodup).denote γ := by
  cases η with
  | float => simp [JAX.Expr.denote, ih]
  | nat => simp [JAX.Expr.denote, ih]
  | disc d =>
    cases d with
    | fin _ _ => simp [JAX.Expr.denote, ih]

/-- Semantics of `PAL.Stmt.weakenMask dummy e`: the denotation in a valuation that
    only prepends `dummy` with value `v` agrees with `e.denote` on the unextended `γ`.
    Used where `compileCondition` applies `weakenMask` to the mask so it shares a context
    with `var probabilityTensor`; the proof obligation is that the extra binding does
    not affect variables occurring in `e` (`hfresh`). -/
private theorem weakenMask_denote
    {Γ : JAX.Context} {τ : JAX.Ty}
    (dummy : Σ _ : String, JAX.Ty)
    (e : JAX.Expr (L := L) Γ τ) (h : e.hasLetIn = false)
    (γ : JAX.Valuation Γ) (v : dummy.snd.Sem)
    (hfresh : ∀ {n τ}, ⟨n, τ⟩ ∈ Γ → n ≠ dummy.fst) :
    (PAL.Stmt.weakenMask (L := L) dummy e h).denote (γ.cons dummy.fst dummy.snd v) = e.denote γ := by
  revert h
  induction e with
  | var name T hmem =>
      intro h
      have hne : name ≠ dummy.fst := hfresh hmem
      simp [PAL.Stmt.weakenMask, JAX.Expr.denote]
      rw [Environment.get_cons_ne γ v hmem hne]
  | «array» η t =>
      intro h
      simp [PAL.Stmt.weakenMask, JAX.Expr.denote]
  | uop op e' hv ih =>
      intro h
      have he' : e'.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      simp (config := { zeta := true }) [PAL.Stmt.weakenMask, JAX.Expr.denote]
      rw [ih γ hfresh he']
  | bop op e₁ e₂ hv hbc ih₁ ih₂ =>
      intro h
      have ⟨he₁, he₂⟩ : e₁.hasLetIn = false ∧ e₂.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      rw [PAL.Stmt.weakenMask_bop_unfold (L := L) (dummy := dummy) op e₁ e₂ hv hbc h he₁ he₂]
      simp [JAX.Expr.denote]
      rw [ih₁ γ hfresh he₁, ih₂ γ hfresh he₂]
  | «select» e₁ e₂ e₃ hN ih₁ ih₂ ih₃ =>
      intro h
      have ⟨⟨he₁, he₂⟩, he₃⟩ :
          (e₁.hasLetIn = false ∧ e₂.hasLetIn = false) ∧ e₃.hasLetIn = false := by
        simpa [JAX.Expr.hasLetIn] using h
      rw [PAL.Stmt.weakenMask_select_unfold (L := L) (dummy := dummy) e₁ e₂ e₃ hN h he₁ he₂ he₃]
      simp [JAX.Expr.denote]
      rw [ih₁ γ hfresh he₁, ih₂ γ hfresh he₂, ih₃ γ hfresh he₃]
  | «onesLike» e' ih =>
      intro h
      simp [PAL.Stmt.weakenMask, JAX.Expr.denote]
  | «ones» σ =>
      intro h
      simp [PAL.Stmt.weakenMask, JAX.Expr.denote]
  | broadcastTo e' σₒ hbc ih =>
      intro h
      have he' : e'.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      simp (config := { zeta := true }) [PAL.Stmt.weakenMask, JAX.Expr.denote]
      rw [ih γ hfresh he']
  | «reshape» e' σₜ hreshape ih =>
      intro h
      have he' : e'.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      simp (config := { zeta := true }) [PAL.Stmt.weakenMask, JAX.Expr.denote]
      rw [ih γ hfresh he']
  | «sum» e' axes hNodup ih =>
      intro h
      have he' : e'.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      rw [PAL.Stmt.weakenMask_sum_unfold (L := L) (dummy := dummy) e' axes hNodup h]
      have hInner := ih γ hfresh he'
      rw [← PAL.Stmt.weakenMask_proof_irrel e' h he'] at hInner
      exact denote_sum_weakenMask dummy e' axes hNodup h γ v hInner
  | «squeeze» e' ax hsq ih =>
      intro h
      have he' : e'.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      simp (config := { zeta := true }) [PAL.Stmt.weakenMask, JAX.Expr.denote]
      rw [ih γ hfresh he']
  | «transpose» e' permV hperm ih =>
      intro h
      have he' : e'.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      simp (config := { zeta := true }) [PAL.Stmt.weakenMask, JAX.Expr.denote]
      rw [ih γ hfresh he']
  | letIn _ _ _ => intro h; simp [JAX.Expr.hasLetIn] at h
  | «comment» s e' ih =>
      intro h
      have he' : e'.hasLetIn = false := by simpa [JAX.Expr.hasLetIn] using h
      rw [PAL.Stmt.weakenMask_comment_unfold (L := L) (dummy := dummy) s e' h]
      simp [JAX.Expr.denote]
      rw [← PAL.Stmt.weakenMask_proof_irrel e' h he']
      rw [ih γ hfresh he']
  | «scan» _ _ _ _ => intro h; exact absurd h (by simp [JAX.Expr.hasLetIn])

end

section

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L] [Tensor.LawfulOfTabulated L]

/-- Correctness of `condition c`. -/
theorem Condition.correctness
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext} {Γ : JAX.Context}
    (c : PAL.Expr (Θ := Θ) ↑Δ Ty.bool)
    {θ : ⟦Θ⟧}
    (μ : ℙ⟦↑Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float))
    (hμ : ptExpr.denote γ ~ μ)
    (hpc : γ ~ θ)
    (hpt : probabilityTensor ∉ (Γ ++ 𝒞⟦Θ⟧).keys) :
    (PAL.Stmt.compileCondition (L := L) c ptExpr).denote γ ~
    (PAL.Stmt.evalCondition (θ := θ) c μ) := by
  intro δ
  simp only [PAL.Stmt.evalCondition]
  have ih_c := (Expr.correctness (L := L) c δ γ) hpc
  dsimp only [Expr.Correct] at ih_c
  have hfresh : ∀ {n τ}, ⟨n, τ⟩ ∈ Γ ++ 𝒞⟦Θ⟧ → n ≠ probabilityTensor :=
    fun {_ _} hmem => ne_probabilityTensor_of_probabilityTensor_not_in_keys hpt hmem
  let cr := c.compile (L := L) (Γ := Γ)
  let ptTy := JAX.Ty.tensor 𝒞⟦Δ⟧ .float
  let ptBind : Σ _ : String, JAX.Ty := ⟨probabilityTensor, ptTy⟩
  have hw :=
    weakenMask_denote (L := L) (Γ := Γ ++ 𝒞⟦Θ⟧) ptBind cr.expr
      (PAL.Expr.compile_res_expr_hasLetIn_false c) γ (ptExpr.denote γ)
      (fun {_ _} hmem => hfresh hmem)
  let bcd₁ := Shape.Broadcastable.broadcast2_common_data
    (Shape.Broadcastable.refl 𝒞⟦Δ⟧) (Shape.Broadcastable.nil_left 𝒞⟦Δ⟧)
  let bcd₂ := Shape.Broadcastable.broadcast2_common_data cr.broadcastable bcd₁.property.2.2
  let hN := Shape.BroadcastNCompatible.of_triple bcd₁.val bcd₂.val
  have hS : (⊔ₙ #v[cr.σ, 𝒞⟦Δ⟧, []] ‖ hN) = 𝒞⟦Δ⟧ :=
    PAL.Stmt.conditionSelect_shape_eq cr.broadcastable bcd₁.val bcd₂.val
  simp only [PAL.Stmt.compileCondition, JAX.Expr.denote]
  rw [denote_cast_shape hS]
  simp only [JAX.Expr.denote, Environment.get_cons_eq]
  rw [hw, Indices.proj_cast _ hS.symm _ cr.broadcastable, ih_c,
    Indices.proj_cast _ hS.symm _ (Shape.Broadcastable.refl 𝒞⟦Δ⟧), Indices.proj_refl, hμ δ,
    Tensor.index_nil, Tensor.scalar_ofTabulated]
  change _ = if Ty.toBool .bool (Denote.denote c θ δ) then μ δ else 0.0
  generalize Denote.denote c θ δ = v
  change Fin 2 at v
  rcases v with ⟨_ | _ | n, hn⟩
  · rfl
  · rfl
  · omega

end

end Palladium.Compiler
