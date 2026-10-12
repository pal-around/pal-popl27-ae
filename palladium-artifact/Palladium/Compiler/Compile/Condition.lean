import Palladium.Compiler.Compile.Expr

/-! ## Bayesian conditioning compilation -/

namespace Palladium.PAL.Stmt

open Palladium.Util
open Palladium.PAL
open Palladium.Compiler
open Palladium.TensorLib
open Shape

section

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-- Projections for `bop`’s `hasLetIn` (used so `weakenMask` on `bop` unfolds cleanly). -/
def bop_hasLetIn_left {Γ : JAX.Context} (op : JAX.BOp) {σ₁ σ₂ : Shape} {η₁ η₂ ηₒ : DType}
    (e₁ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η₁)) (e₂ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η₂))
    (hv : JAX.BOp.Valid op η₁ η₂ ηₒ) (hbc : σ₁ ⋈ σ₂)
    (h : (JAX.Expr.bop op e₁ e₂ hv hbc).hasLetIn = false) : e₁.hasLetIn = false :=
  (by simpa [JAX.Expr.hasLetIn] using h : e₁.hasLetIn = false ∧ e₂.hasLetIn = false).1

def bop_hasLetIn_right {Γ : JAX.Context} (op : JAX.BOp) {σ₁ σ₂ : Shape} {η₁ η₂ ηₒ : DType}
    (e₁ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η₁)) (e₂ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η₂))
    (hv : JAX.BOp.Valid op η₁ η₂ ηₒ) (hbc : σ₁ ⋈ σ₂)
    (h : (JAX.Expr.bop op e₁ e₂ hv hbc).hasLetIn = false) : e₂.hasLetIn = false :=
  (by simpa [JAX.Expr.hasLetIn] using h : e₁.hasLetIn = false ∧ e₂.hasLetIn = false).2

/-- Projections for `select`’s `hasLetIn` (used so `weakenMask` on `select` unfolds cleanly). -/
def select_hasLetIn_e1 {Γ : JAX.Context} {σ₃ σ₁ σ₂ : Shape} {η₁ η₂ : DType}
    (e₁ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₃ DType.bool))
    (e₂ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η₁)) (e₃ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η₂))
    (hN : ⋈ₙ #v[σ₃, σ₁, σ₂])
    (h : (JAX.Expr.select e₁ e₂ e₃ hN).hasLetIn = false) : e₁.hasLetIn = false :=
  (by simpa [JAX.Expr.hasLetIn] using h : (e₁.hasLetIn = false ∧ e₂.hasLetIn = false) ∧ e₃.hasLetIn = false).1.1

def select_hasLetIn_e2 {Γ : JAX.Context} {σ₃ σ₁ σ₂ : Shape} {η₁ η₂ : DType}
    (e₁ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₃ DType.bool))
    (e₂ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η₁)) (e₃ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η₂))
    (hN : ⋈ₙ #v[σ₃, σ₁, σ₂])
    (h : (JAX.Expr.select e₁ e₂ e₃ hN).hasLetIn = false) : e₂.hasLetIn = false :=
  (by simpa [JAX.Expr.hasLetIn] using h : (e₁.hasLetIn = false ∧ e₂.hasLetIn = false) ∧ e₃.hasLetIn = false).1.2

def select_hasLetIn_e3 {Γ : JAX.Context} {σ₃ σ₁ σ₂ : Shape} {η₁ η₂ : DType}
    (e₁ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₃ DType.bool))
    (e₂ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η₁)) (e₃ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η₂))
    (hN : ⋈ₙ #v[σ₃, σ₁, σ₂])
    (h : (JAX.Expr.select e₁ e₂ e₃ hN).hasLetIn = false) : e₃.hasLetIn = false :=
  (by simpa [JAX.Expr.hasLetIn] using h : (e₁.hasLetIn = false ∧ e₂.hasLetIn = false) ∧ e₃.hasLetIn = false).2

/-- Prepend a dummy binding so an expression typed in `Γ` can be used in `dummy :: Γ`.

    **Restriction (`e.hasLetIn = false`).** Implementation-wise, `weakenMask` matches on
    `JAX.Expr` and has no `letIn` arm—the `letIn` branch is `False.elim` using
    `hasLetIn (letIn …) = true` (see below).

    **Design reason.** A `letIn x e₁ e₂` body `e₂` is typed in `⟨x, T⟩ :: Γ`, not `Γ`.
    Lifting through `dummy` is then not just re-witnessing `var` membership in
    `dummy :: Γ`: one must specify how `dummy` relates to the inner binder (order of
    extensions), and prove `weakenMask_denote`-style lemmas for `letIn` under
    `Environment.cons` (when to extend the valuation with `x` vs `dummy`). This file
    weakens `letIn`-free expressions only.

    **Why compiled conditions satisfy it.** `PAL.Expr.compile` never emits `letIn` in
    expression trees (`compile_res_expr_hasLetIn_false`).-/
def weakenMask {Γ : JAX.Context} {τ : JAX.Ty} (dummy : Σ _ : String, JAX.Ty)
    (e : JAX.Expr (L := L) Γ τ) (h : e.hasLetIn = false) :
    JAX.Expr (L := L) (dummy :: Γ) τ :=
  match e with
  | .var name T hmem => JAX.Expr.var name T (List.mem_cons_of_mem dummy hmem)
  | .array η t => JAX.Expr.array η t
  | .uop op e hv =>
      have he : e.hasLetIn = false := h
      JAX.Expr.uop op (weakenMask dummy e he) hv
  | .bop op e₁ e₂ hv hbc =>
      JAX.Expr.bop op (weakenMask dummy e₁ (bop_hasLetIn_left op e₁ e₂ hv hbc h))
        (weakenMask dummy e₂ (bop_hasLetIn_right op e₁ e₂ hv hbc h)) hv hbc
  | .select e₁ e₂ e₃ hN =>
      JAX.Expr.select
        (weakenMask dummy e₁ (select_hasLetIn_e1 e₁ e₂ e₃ hN h))
        (weakenMask dummy e₂ (select_hasLetIn_e2 e₁ e₂ e₃ hN h))
        (weakenMask dummy e₃ (select_hasLetIn_e3 e₁ e₂ e₃ hN h)) hN
  | .onesLike e =>
      have he : e.hasLetIn = false := h
      JAX.Expr.onesLike (weakenMask dummy e he)
  | .ones σ => JAX.Expr.ones σ
  | .broadcastTo e σₒ hbc =>
      have he : e.hasLetIn = false := h
      JAX.Expr.broadcastTo (weakenMask dummy e he) σₒ hbc
  | .reshape e σₜ hreshape =>
      have he : e.hasLetIn = false := h
      JAX.Expr.reshape (weakenMask dummy e he) σₜ hreshape
  | .sum e axes hNodup =>
      have he : e.hasLetIn = false := h
      JAX.Expr.sum (weakenMask dummy e he) axes hNodup
  | .squeeze e ax hsq =>
      have he : e.hasLetIn = false := h
      JAX.Expr.squeeze (weakenMask dummy e he) ax hsq
  | .transpose e permV hperm =>
      have he : e.hasLetIn = false := h
      JAX.Expr.transpose (weakenMask dummy e he) permV hperm
  | .comment s e =>
      have he : e.hasLetIn = false := h
      JAX.Expr.comment s (weakenMask dummy e he)
  | .letIn .. => False.elim (by simp [JAX.Expr.hasLetIn] at h)
  | .scan .. => False.elim (by simp [JAX.Expr.hasLetIn] at h)

end

section

variable {L : Tensor.Layout}

/-- `weakenMask` does not depend on which proof witnesses `hasLetIn = false`. -/
theorem weakenMask_proof_irrel {Γ : JAX.Context} {dummy : Σ _ : String, JAX.Ty}
    {τ : JAX.Ty} (e : JAX.Expr (L := L) Γ τ) (h₁ h₂ : e.hasLetIn = false) :
    weakenMask dummy e h₁ = weakenMask dummy e h₂ := by
  cases Subsingleton.elim h₁ h₂;   rfl

/-- Unfold `weakenMask` on `bop` (for correctness proofs). -/
theorem weakenMask_bop_unfold {Γ : JAX.Context} {dummy : Σ _ : String, JAX.Ty}
    (op : JAX.BOp) {σ₁ σ₂ : Shape} {η₁ η₂ ηₒ : DType}
    (e₁ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η₁))
    (e₂ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η₂))
    (hv : JAX.BOp.Valid op η₁ η₂ ηₒ) (hbc : σ₁ ⋈ σ₂)
    (h : (JAX.Expr.bop op e₁ e₂ hv hbc).hasLetIn = false)
    (he₁ : e₁.hasLetIn = false) (he₂ : e₂.hasLetIn = false) :
    weakenMask dummy (JAX.Expr.bop op e₁ e₂ hv hbc) h =
      JAX.Expr.bop op (weakenMask dummy e₁ he₁) (weakenMask dummy e₂ he₂) hv hbc := by
  rw [weakenMask_proof_irrel e₁ (bop_hasLetIn_left op e₁ e₂ hv hbc h) he₁,
    weakenMask_proof_irrel e₂ (bop_hasLetIn_right op e₁ e₂ hv hbc h) he₂]
  rfl

/-- Unfold `weakenMask` on `select` (for correctness proofs). -/
theorem weakenMask_select_unfold {Γ : JAX.Context} {dummy : Σ _ : String, JAX.Ty}
    {σ₃ σ₁ σ₂ : Shape} {η₁ η₂ : DType}
    (e₁ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₃ DType.bool))
    (e₂ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η₁))
    (e₃ : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η₂))
    (hN : ⋈ₙ #v[σ₃, σ₁, σ₂])
    (h : (JAX.Expr.select e₁ e₂ e₃ hN).hasLetIn = false)
    (he₁ : e₁.hasLetIn = false) (he₂ : e₂.hasLetIn = false) (he₃ : e₃.hasLetIn = false) :
    weakenMask dummy (JAX.Expr.select e₁ e₂ e₃ hN) h =
      JAX.Expr.select (weakenMask dummy e₁ he₁) (weakenMask dummy e₂ he₂)
        (weakenMask dummy e₃ he₃) hN := by
  rw [weakenMask_proof_irrel e₁ (select_hasLetIn_e1 e₁ e₂ e₃ hN h) he₁,
    weakenMask_proof_irrel e₂ (select_hasLetIn_e2 e₁ e₂ e₃ hN h) he₂,
    weakenMask_proof_irrel e₃ (select_hasLetIn_e3 e₁ e₂ e₃ hN h) he₃]
  rfl

/-- Unfold `weakenMask` on `sum` (for correctness proofs). -/
theorem weakenMask_sum_unfold {Γ : JAX.Context} {dummy : Σ _ : String, JAX.Ty}
    {σ : Shape} {η : DType} (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ η))
    (axes : List (Fin σ.rank)) (hNodup : axes.Nodup)
    (h : (JAX.Expr.sum e axes hNodup).hasLetIn = false) :
    weakenMask dummy (JAX.Expr.sum e axes hNodup) h =
      JAX.Expr.sum (weakenMask dummy e h) axes hNodup :=
  rfl

/-- Unfold `weakenMask` on `comment` (for correctness proofs). -/
theorem weakenMask_comment_unfold {Γ : JAX.Context} {dummy : Σ _ : String, JAX.Ty}
    {τ : JAX.Ty} (s : String) (e : JAX.Expr (L := L) Γ τ)
    (h : (JAX.Expr.comment s e).hasLetIn = false) :
    weakenMask dummy (JAX.Expr.comment s e) h =
      JAX.Expr.comment s (weakenMask dummy e h) :=
  rfl

end

section

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]
variable {Θ : ValidStaticContext}

/-- The output shape of `select(c, PT, 0)` in `compileCondition` is the shape of `PT`. -/
theorem conditionSelect_shape_eq {σc σ : Shape} (hc : σc ≤ σ)
    (h₁ : Shape.BroadcastCompatible σ []) (h₂ : Shape.BroadcastCompatible σc (σ ⊔ [] ‖ h₁)) :
    (⊔ₙ #v[σc, σ, []] ‖ Shape.BroadcastNCompatible.of_triple h₁ h₂) = σ := by
  have hB : (σ ⊔ [] ‖ h₁) = σ :=
    Shape.Broadcastable.antisymm
      (Shape.Broadcastable.broadcast2_common_data (Shape.Broadcastable.refl σ)
        (Shape.Broadcastable.nil_left σ)).property.2.2
      (Shape.Broadcastable.of_broadcastCompatible_left h₁)
  rw [Shape.broadcastN_triple_eq h₁ h₂,
    Shape.Broadcastable.broadcast2_eq_right (hB.symm ▸ hc) h₂, hB]

/-- Compile `condition c` to `letIn PT ptExpr (comment "Condition" (select(c, PT, 0)))`,
    following rule C-Condition of the paper (`where(e, e_□, 0)`).
    The compiled condition is typed in `Γ ++ 𝒞⟦Θ⟧`; `weakenMask` lifts it under the
    `letIn` that binds `PT`. -/
def compileCondition {Δ : WellFormedRandomContext} {Γ : JAX.Context}
    (c : Expr (Θ := Θ) ↑Δ Ty.bool)
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float)) :
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float) :=
  let cr := c.compile (L := L) (Γ := Γ)
  let ptTy := JAX.Ty.tensor 𝒞⟦Δ⟧ .float
  let ptBind : Σ _ : String, JAX.Ty := ⟨probabilityTensor, ptTy⟩
  let ptCtx := ptBind :: (Γ ++ 𝒞⟦Θ⟧)
  let condW := weakenMask ptBind cr.expr (PAL.Expr.compile_res_expr_hasLetIn_false c)
  let ptRef : JAX.Expr (L := L) ptCtx (JAX.Ty.tensor 𝒞⟦Δ⟧ .float) :=
    JAX.Expr.var probabilityTensor _ (.head _)
  let zero : JAX.Expr (L := L) ptCtx (JAX.Ty.tensor [] .float) :=
    JAX.Expr.array .float (Tensor.ofTabulated (L := L) (σ := []) (0.0 : Float))
  let bcd₁ := Shape.Broadcastable.broadcast2_common_data
    (Shape.Broadcastable.refl 𝒞⟦Δ⟧) (Shape.Broadcastable.nil_left 𝒞⟦Δ⟧)
  let bcd₂ := Shape.Broadcastable.broadcast2_common_data cr.broadcastable bcd₁.property.2.2
  let hN := Shape.BroadcastNCompatible.of_triple bcd₁.val bcd₂.val
  let sel : JAX.Expr (L := L) ptCtx (JAX.Ty.tensor (⊔ₙ #v[cr.σ, 𝒞⟦Δ⟧, []] ‖ hN) .float) :=
    JAX.Expr.select condW ptRef zero hN
  JAX.Expr.letIn probabilityTensor ptExpr
    (JAX.Expr.comment "Condition"
      (cast (congrArg (fun s => JAX.Expr (L := L) ptCtx (JAX.Ty.tensor s .float))
        (conditionSelect_shape_eq cr.broadcastable bcd₁.val bcd₂.val)) sel))

end

end Palladium.PAL.Stmt
