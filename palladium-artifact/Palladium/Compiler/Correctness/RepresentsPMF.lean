import Palladium.Compiler.Correctness.WellFormed

namespace Palladium.TensorLib

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL
open Palladium.Compiler
open Tensor

/-!
## The PMF relation

An indexing function `f : I(𝒞⟦Δ⟧) → Float` represents a PMF `μ` over `Δ` (`f ~ μ`) if, for
every random environment `δ`, `f` at the indices of `δ` is the probability of `δ`:

`∀ δ : ⟦Δ⟧,  f [idx(δ)] = μ(δ)`

`Tensor.RepresentsPMF` states the same relation for a tensor in a layout `L`.
-/

/-- `f ~ μ  ≡  ∀ δ, f [idx(δ)] = μ(δ)`. -/
def IdxFn.RepresentsPMF
    {ctx : RandomContext}
    (f : Indices 𝒞⟦ctx⟧ → Float)
    (μ : ℙ⟦ctx⟧) : Prop :=
  ∀ (δ : ⟦ctx⟧), f δ.toIndices = μ δ

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- `t` represents `μ`  ≡  `∀ δ, t[idx(δ)] = μ(δ)`. -/
def Tensor.RepresentsPMF
    {ctx : RandomContext}
    (tensor : Tensor (L := L) 𝒞⟦ctx⟧ Float)
    (μ : ℙ⟦ctx⟧) : Prop :=
  ∀ (δ : ⟦ctx⟧), tensor[δ] = μ δ

end

/-- Tilde notation for `IdxFn.RepresentsPMF`. Write `f ~ μ` in place of
    `IdxFn.RepresentsPMF f μ`, matching the probabilistic convention X ~ P. -/
scoped notation:50 f " ~ " μ:51 => IdxFn.RepresentsPMF f μ

/-- If `f` represents a PMF over `Δ'`, then reading `f` at the inverse-permuted indices
    represents the corresponding PMF over `Δ`, where `Δ.Perm Δ'`. -/
theorem IdxFn.RepresentsPMF.transpose
    {Δ' : RandomContext}
    {hnd_keys : Δ'.keys.Nodup}
    {f : Indices 𝒞⟦Δ'⟧ → Float}
    {μ : ℙ⟦Δ'⟧}
    (hrepr : f ~ μ)
    (hnd' : Δ'.NodupKeys)
    {Δ : RandomContext}
    (hperm : Δ.Perm Δ')
    (retPerm : List Nat)
    (hRetPermEq : retPerm = Block.computeReturnPerm Δ' Δ.keys)
    (hv : Permutation.Valid retPerm 𝒞⟦Δ'⟧.rank)
    (hnd_out : Δ.keys.Nodup)
    (hshape_perm : 𝒞⟦Δ'⟧.permute retPerm hv = 𝒞⟦Δ⟧) :
    (fun i : Indices 𝒞⟦Δ⟧ => f (i.cast hshape_perm.symm).invPermute) ~
      (fun δ => μ (δ.ofPerm hperm)) := by
  subst hRetPermEq
  intro δ
  have hperm_keys : Δ.keys.Perm Δ'.keys := hperm.map Sigma.fst
  have hv' : Permutation.Valid (Block.computeReturnPerm Δ' Δ.keys) 𝒞⟦Δ'⟧.rank :=
    Δ'.toShape_length ▸
      (Block.computeReturnPerm_valid Δ' Δ.keys hnd_keys hperm_keys)
  have h_inv := δ.toIndices_ofPerm Δ' hperm hnd' hnd_out hv'
  have h_eq : (δ.toIndices.cast hshape_perm.symm).invPermute =
      (↑(δ.ofPerm hperm) : Indices _) :=
    Indices.ext h_inv.symm
  show f (δ.toIndices.cast hshape_perm.symm).invPermute = _
  rw [h_eq]
  exact hrepr _

/-- If `f` represents a PMF over `ctx₁`, then `f` read at cast indices represents the
    renamed PMF over `ctx₂`, a context with the same types at each position. -/
theorem IdxFn.RepresentsPMF.of_shape_cast
    {ctx₁ ctx₂ : RandomContext}
    (htypes : ctx₁.map Sigma.snd = ctx₂.map Sigma.snd)
    (hnd₁ : ctx₁.NodupKeys)
    {f : Indices 𝒞⟦ctx₁⟧ → Float}
    {μ : ℙ⟦ctx₁⟧}
    (hrepr : f ~ μ)
    (hshape : 𝒞⟦ctx₁⟧ = 𝒞⟦ctx₂⟧ := RandomContext.toShape_eq_of_map_snd_eq htypes) :
    (fun i : Indices 𝒞⟦ctx₂⟧ => f (i.cast hshape.symm)) ~
      (fun σ => μ (σ.renameByTypes htypes.symm)) := by
  intro σ
  have h_indices : (Environment.toIndices σ).cast hshape.symm =
      Environment.toIndices (Environment.renameByTypes htypes.symm σ) := by
    apply Palladium.TensorLib.Indices.ext
    simp only [Indices.cast, Environment.toIndices_values]
    exact (Environment.toIndicesRaw_renameByTypes ctx₂ ctx₁ htypes.symm hnd₁ σ).symm
  show f ((Environment.toIndices σ).cast hshape.symm) = _
  rw [h_indices]
  exact hrepr _

end Palladium.TensorLib
