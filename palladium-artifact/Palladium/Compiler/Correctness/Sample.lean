import Palladium.Compiler.Correctness.Expr
import Palladium.Compiler.Correctness.Marginalize

/-!
# Sample Correctness

`⟦ptExpr⟧ γ ~ μ → ⟦compileSample vars ... ptExpr⟧ γ ~ evalSample(vars, wppc, μ)`

where `evalSample(vars, wppc, μ)(ρ ++ δ) = μ(δ) * Pr(x⃗ = ρ | δ)` and
`Pr(x⃗ = ρ | δ) = w(ρ, δ) / Σ_{ρ'} w(ρ', δ)` is the conditional probability derived
from the weight program `wppc` (with `0 / 0 = 0`).

The proof reads the denotation of the compiled expression at the indices of `δ' = ρ ++ δ`:
* the weight holds `w(ρ, δ)` (`denote_sampleWeight`);
* `REDUCE(w, m, 0)` holds the sum of `w` over the leading `m` axes (`denote_reduce`), which
  is the sum over all valuations of `x⃗` (`sumAll_eq_sumIdx`);
* the output multiplies the prior mass by the normalized weight
  (`denote_sampleUpdate`, `denote_sampleNormalize`).
-/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL
open Palladium.JAX
open Shape Tensor

/-! ### Environments and indices -/

/-- `Environment.tail` of a cons with a fresh name is the original environment. -/
private theorem tail_cons_of_fresh
    {α : Type} {denote : α → Type} [DecidableEq α]
    {ctx : Context α}
    (env : Environment denote ctx)
    (name : String) (payload : α)
    (value : denote payload)
    (hfresh : name ∉ ctx.keys) :
    (env.cons name payload value).tail = env := by
  funext y τ h
  simp only [Environment.tail, Environment.cons]
  have hne : (⟨y, τ⟩ : Σ _ : String, α) ≠ ⟨name, payload⟩ := by
    intro heq
    have hfst : y = name := congrArg Sigma.fst heq
    subst hfst
    exact hfresh (List.mem_keys.mpr ⟨_, h⟩)
  rw [dif_neg hne]

/-- Indices of a cons environment with a fresh name. -/
private theorem toIndices_cons_of_fresh
    {n : Nat} {hn : 0 < n} {ctx : RandomContext}
    (env : ⟦ctx⟧) (name : String)
    (v : Fin n) (hfresh : name ∉ ctx.keys) :
    (env.cons name (.fin n hn) v).toIndices =
    Indices.cons v.val v.isLt (env.toIndices) := by
  apply Indices.ext
  show Environment.toIndicesRaw (⟨name, .fin n hn⟩ :: ctx) (env.cons name (.fin n hn) v)
     = v.val :: Environment.toIndicesRaw ctx env
  conv_lhs => unfold Environment.toIndicesRaw
  rw [tail_cons_of_fresh env name (.fin n hn) v hfresh]
  simp [Environment.cons]

/-- The index list of an environment over `pre ++ ctx` is the concatenation of the index
    lists of its two parts. -/
theorem toIndicesRaw_append : (pre ctx : RandomContext) → (δ' : ⟦pre ++ ctx⟧) →
    Environment.toIndicesRaw (pre ++ ctx) δ' =
      Environment.toIndicesRaw pre δ'.takePrefix ++
        Environment.toIndicesRaw ctx δ'.dropPrefix
  | [], _, _ => rfl
  | ⟨_, .fin _ _⟩ :: rest, ctx, δ' =>
      congrArg (_ :: ·) (toIndicesRaw_append rest ctx δ'.tail)

/-- Indices of an environment over `pre ++ ctx`, split into its two parts. -/
theorem toIndices_split {pre ctx : RandomContext} (δ' : ⟦pre ++ ctx⟧) :
    δ'.toIndices.cast (RandomContext.toShape_append pre ctx) =
      δ'.takePrefix.toIndices.append δ'.dropPrefix.toIndices :=
  Indices.ext (toIndicesRaw_append pre ctx δ')

/-- Indices of `ρ ++ δ` when the names of `ρ` are fresh for `δ`. -/
theorem toIndices_prepend {pre ctx : RandomContext}
    (hfresh : ∀ x ∈ pre.keys, x ∉ ctx.keys) (ρ : ⟦pre⟧) (δ : ⟦ctx⟧) :
    (ρ.prepend pre δ).toIndices.cast (RandomContext.toShape_append pre ctx) =
      ρ.toIndices.append δ.toIndices := by
  rw [toIndices_split, Environment.takePrefix_prepend,
    Environment.dropPrefix_prepend pre hfresh]

/-- The names of `vars` are fresh for `Δ` when `vars ++ Δ` has distinct keys. -/
theorem fresh_of_nodupKeys_append {pre ctx : RandomContext} (h : (pre ++ ctx).NodupKeys) :
    ∀ x ∈ pre.keys, x ∉ ctx.keys := by
  intro x hx hx'
  have hnd : (pre.keys ++ ctx.keys).Nodup := by
    simpa [List.NodupKeys, List.keys_append] using h
  exact List.disjoint_of_nodup_append hnd hx hx'

/-! ### Sums over valuations and over indices -/

/-- Sum of `f` over all indices of `pre`, the axis-0 index innermost. -/
def sumIdx : (pre : Shape) → (Indices pre → Float) → Float
  | [], f => f Indices.nil
  | n :: pre, f =>
      sumIdx pre (fun i => (List.ofFn (fun v : Fin n => f (Indices.cons v.val v.isLt i))).sum)

/-- Summing over all valuations of `pre` is summing over all indices of `𝒞⟦pre⟧`. -/
theorem sumAll_eq_sumIdx : (pre : RandomContext) → pre.NodupKeys →
    (f : ⟦pre⟧ → Float) → (g : Indices pre.toShape → Float) →
    (∀ ρ, f ρ = g ρ.toIndices) →
    RandomContext.sumAll pre f = sumIdx pre.toShape g
  | [], _, f, g, h => by
      simp only [RandomContext.sumAll, RandomContext.toShape, sumIdx]
      rw [h]
      exact congrArg g (Indices.ext rfl)
  | ⟨x, .fin n hn⟩ :: rest, hnd, f, g, h => by
      have ⟨hx, hrest⟩ := List.nodupKeys_cons.mp hnd
      simp only [RandomContext.sumAll, RandomContext.toShape, sumIdx]
      refine sumAll_eq_sumIdx rest hrest _ _ (fun ρ => ?_)
      simp only [Discrete.support, List.map_ofFn]
      congr 1
      apply List.ofFn_inj.mpr
      funext v
      simp only [Function.comp, h]
      rw [toIndices_cons_of_fresh ρ x v hx]

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-! ### Denotation of the sampling helpers -/

/-- The denotation of a shape-transported JAX expression reads the cast index. -/
theorem denote_castShape {Γ : JAX.Context} {σ₁ σ₂ : Shape} {η : DType} (h : σ₁ = σ₂)
    (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η)) (γ : JAX.Valuation Γ) (i : Indices σ₂) :
    (castShape h e).denote γ i = e.denote γ (i.cast h.symm) := by
  subst h; rfl

/-- `sum(e, 0)` at index `j` is the sum over the axis-0 index. -/
theorem denote_sumAxis0 {Γ : JAX.Context} {n : Nat} {σ : Shape}
    (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor (n :: σ) .float)) (γ : JAX.Valuation Γ)
    (j : Indices σ) :
    (sumAxis0 e).denote γ j =
      (List.ofFn (fun v : Fin n => e.denote γ (Indices.cons v.val v.isLt j))).sum := by
  simp only [sumAxis0]
  rw [denote_cast_ty (sum_axis0_ty n σ), Ty.Sem_cast_apply (sum_axis0_ty n σ)]
  simp only [JAX.Expr.denote, IdxFn.sumAxes_singleton, IdxFn.sum, castDenoteDtype, cast_eq]
  rfl

/-- `REDUCE(e, |pre|, 0)` at index `j` is the sum of `e` over all leading indices. -/
theorem denote_reduce {Γ : JAX.Context} :
    (pre : Shape) → {σ : Shape} → (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor (pre ++ σ) .float)) →
    (γ : JAX.Valuation Γ) → (j : Indices σ) →
    (reduce pre e).denote γ j = sumIdx pre (fun i => e.denote γ (i.append j))
  | [], σ, e, γ, j => by
      simp only [reduce, sumIdx]
      exact congrArg (fun i => e.denote γ i) (Indices.ext rfl)
  | n :: pre, σ, e, γ, j => by
      simp only [reduce, sumIdx]
      rw [denote_reduce pre (sumAxis0 e) γ j]
      congr 1
      funext i
      rw [denote_sumAxis0]
      rfl

/-! ### Denotation of the compiled sample -/

/-- The normalized weights at `δ'` are the weight at `δ'` divided by the sum of the weights
    over all valuations of the sampled variables. -/
theorem denote_sampleNormalize {Δ : WellFormedRandomContext}
    (vars : WellFormedRandomContext) {Γ' : JAX.Context} (w : String)
    (γ : JAX.Valuation Γ') (W : (PAL.Stmt.sampleTy vars Δ).Sem)
    (δ' : ⟦(↑vars ++ ↑Δ : RandomContext)⟧) :
    (PAL.Stmt.sampleNormalize (L := L) (Δ := Δ) vars w).denote (γ.cons w _ W)
        δ'.toIndices =
      Float.div0 (W δ'.toIndices)
        (sumIdx 𝒞⟦(↑vars : RandomContext)⟧ (fun i =>
          W ((i.append δ'.dropPrefix.toIndices).cast
            (PAL.Stmt.sampleShape_eq vars (Δ := Δ)).symm))) := by
  have hshp := PAL.Stmt.sampleShape_eq vars (Δ := Δ)
  have hsuf := PAL.Stmt.sampleShape_suffix vars (Δ := Δ)
  simp only [PAL.Stmt.sampleNormalize]
  rw [denote_castShape]
  simp only [JAX.Expr.denote, Environment.get_cons_eq]
  rw [denote_reduce]
  show Float.div0 _ _ = _
  congr 1
  · rw [Indices.proj_cast _ _ _ (Shape.Broadcastable.refl _), Indices.proj_refl]
  · rw [Indices.proj_cast _ _ _ hsuf,
      ← Indices.proj_cast δ'.toIndices hshp (Broadcastable.suffix _ _) hsuf,
      toIndices_split, Indices.proj_append]
    congr 1
    funext i
    rw [denote_castShape]
    simp only [JAX.Expr.denote, Environment.get_cons_eq]

/-- The output at `δ' = ρ ++ δ` is the prior mass at `δ` times the normalized weight. -/
theorem denote_sampleUpdate {Δ : WellFormedRandomContext}
    (vars : WellFormedRandomContext) {Γ' : JAX.Context} (p w : String)
    (hp : p ≠ probabilityTensor) (hw : w ≠ probabilityTensor)
    (γ : JAX.Valuation Γ') (a : (PAL.Stmt.samplePtTy Δ).Sem)
    (W P : (PAL.Stmt.sampleTy vars Δ).Sem)
    (δ' : ⟦(↑vars ++ ↑Δ : RandomContext)⟧) :
    (PAL.Stmt.sampleUpdate (L := L) (Δ := Δ) vars p w).denote
        (((γ.cons probabilityTensor _ a).cons w _ W).cons p _ P) δ'.toIndices =
      a δ'.dropPrefix.toIndices * P δ'.toIndices := by
  have hshp := PAL.Stmt.sampleShape_eq vars (Δ := Δ)
  have hsuf := PAL.Stmt.sampleShape_suffix vars (Δ := Δ)
  simp only [PAL.Stmt.sampleUpdate]
  rw [denote_castShape]
  simp only [JAX.Expr.denote]
  rw [Environment.get_cons_eq,
    Environment.get_cons_ne _ _ (.tail _ (.head _)) (Ne.symm hp),
    Environment.get_cons_ne _ _ (.head _) (Ne.symm hw), Environment.get_cons_eq]
  show _ * _ = _
  congr 1
  · rw [Indices.proj_cast _ _ _ hsuf,
      ← Indices.proj_cast δ'.toIndices hshp (Broadcastable.suffix _ _) hsuf,
      toIndices_split, Indices.proj_append]
  · rw [Indices.proj_cast _ _ _ (Shape.Broadcastable.refl _), Indices.proj_refl]

variable [Tensor.OfTabulated L] [Tensor.LawfulOfTabulated L]

/-- The weight holds the weight `w(δ')` at the indices of `δ'`. -/
theorem denote_sampleWeight
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext} {Γ : JAX.Context}
    (vars : WellFormedRandomContext) (hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys)
    (e : PAL.Expr (Θ := ↑Θ) (↑vars ++ ↑Δ) Ty.float) {θ : ⟦Θ⟧}
    (γ : JAX.Valuation (⟨probabilityTensor, PAL.Stmt.samplePtTy Δ⟩ :: (Γ ++ 𝒞⟦Θ⟧)))
    (hpc : γ ~ θ ‖ (⟨probabilityTensor, PAL.Stmt.samplePtTy Δ⟩ :: Γ))
    (δ' : ⟦(↑vars ++ ↑Δ : RandomContext)⟧) :
    (PAL.Stmt.sampleWeight (L := L) vars hwf e).denote γ δ'.toIndices =
      e.eval (θ := θ) δ' := by
  simp only [PAL.Stmt.sampleWeight, JAX.Expr.denote]
  exact Expr.correctness (L := L) (Δ := ⟨↑vars ++ ↑Δ, hwf⟩)
    (Γ := ⟨probabilityTensor, PAL.Stmt.samplePtTy Δ⟩ :: Γ) e δ' γ hpc

/-- Correctness of the sample compilation case.

    `⟦ptExpr⟧ γ ~ μ → ⟦compileSample vars ... ptExpr⟧ γ ~ evalSample(vars, wppc, μ)` -/
theorem Sample.correctness
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext} {Γ : JAX.Context}
    (vars : WellFormedRandomContext) (hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys)
    (wppc : PAL.Wppc (Θ := Θ) Δ vars hwf)
    {θ : ⟦Θ⟧}
    (μ : ℙ⟦↑Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float))
    (hμ : ptExpr.denote γ ~ μ)
    (hpc : γ ~ θ) :
    (PAL.Stmt.compileSample (L := L) vars hwf wppc ptExpr).denote γ ~
    (PAL.Stmt.evalSample (θ := θ) vars wppc μ) := by
  obtain ⟨e⟩ := wppc
  intro δ'
  have hshp := PAL.Stmt.sampleShape_eq vars (Δ := Δ)
  have hpc₁ := hpc.cons_reserved probabilityTensor (PAL.Stmt.samplePtTy Δ) (ptExpr.denote γ)
    (by rfl)
  have hW := denote_sampleWeight (L := L) vars hwf e _ hpc₁
  simp only [PAL.Stmt.compileSample, JAX.Expr.denote]
  rw [denote_sampleUpdate vars _ _ probName_ne_probabilityTensor
    weightName_ne_probabilityTensor]
  rw [denote_sampleNormalize]
  simp only [PAL.Stmt.evalSample, PAL.Wppc.eval]
  congr 1
  · exact hμ δ'.dropPrefix
  · congr 1
    · rw [hW, Environment.prepend_takePrefix_dropPrefix]
    · symm
      apply sumAll_eq_sumIdx _ vars.wf
      intro ρ
      rw [← toIndices_prepend (fresh_of_nodupKeys_append hwf) ρ δ'.dropPrefix]
      rw [show ((ρ.prepend (↑vars : RandomContext) δ'.dropPrefix).toIndices.cast
          (RandomContext.toShape_append _ _)).cast hshp.symm =
          (ρ.prepend (↑vars : RandomContext) δ'.dropPrefix).toIndices from Indices.ext rfl]
      rw [hW]

end

end Palladium.Compiler
