import Palladium.Compiler.Compile.Basics

/-! ## Return Permutation Infrastructure

Shared infrastructure for computing and validating the return permutation
that reorders tensor dimensions from body-internal order to declared output order.
Used by both `Block.compile` and `Stmts.compile` (for ite branches). -/

namespace Palladium.PAL.Block

open Palladium.Util Palladium.Compiler Palladium.TensorLib
open Shape

/-! ### Return permutation computation -/

/-- Compute permutation to reorder tensor dimensions from Δ order to returns order. -/
def computeReturnPerm (Δ : RandomContext) (returns : List String) : Permutation :=
  let names := Δ.keys
  returns.filterMap (fun r => names.findIdx? (· == r))

/-! ### Validity of the return permutation -/

private theorem filterMap_length_of_forall_isSome {f : String → Option Nat}
    {l : List String} (h : ∀ x ∈ l, (f x).isSome) :
    (l.filterMap f).length = l.length := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.filterMap_cons]
    have hx := h x (List.mem_cons_self ..)
    rw [Option.isSome_iff_exists] at hx
    obtain ⟨v, hv⟩ := hx
    simp only [hv, List.length_cons]
    exact congrArg (· + 1) (ih (fun a ha => h a (List.mem_cons_of_mem x ha)))

private theorem filterMap_findIdx?_forall_lt (names : List String) (returns : List String)
    (_ : returns ⊆ names) :
    ∀ x ∈ returns.filterMap (fun r => names.findIdx? (· == r)), x < names.length := by
  intro x hx
  rw [List.mem_filterMap] at hx
  obtain ⟨r, _, hr⟩ := hx
  exact (List.findIdx?_eq_some_iff_getElem.mp hr).1

private theorem filterMap_findIdx?_nodup (names : List String) (returns : List String)
    (_ : names.Nodup) (hnd_ret : returns.Nodup) (hSub : returns ⊆ names) :
    (returns.filterMap (fun r => names.findIdx? (· == r))).Nodup := by
  induction returns with
  | nil => exact List.nodup_nil
  | cons x xs ih =>
    have hx_mem : x ∈ names := hSub (List.mem_cons_self ..)
    have hany : names.any (· == x) = true :=
      List.any_eq_true.mpr ⟨x, hx_mem, beq_self_eq_true x⟩
    rw [← List.findIdx?_isSome] at hany
    rw [Option.isSome_iff_exists] at hany
    obtain ⟨i, hi⟩ := hany
    simp only [List.filterMap_cons, hi]
    have hxs_sub : xs ⊆ names := fun a ha => hSub (List.mem_cons_of_mem x ha)
    have hnd_xs : xs.Nodup := (List.nodup_cons.mp hnd_ret).2
    have hx_notin_xs : x ∉ xs := (List.nodup_cons.mp hnd_ret).1
    apply List.nodup_cons.mpr
    constructor
    · intro hmem
      rw [List.mem_filterMap] at hmem
      obtain ⟨r, hr_mem, hr_eq⟩ := hmem
      have ⟨hlt, hp, _⟩ := List.findIdx?_eq_some_iff_getElem.mp hi
      have ⟨_, hp', _⟩ := List.findIdx?_eq_some_iff_getElem.mp hr_eq
      have heq_x : names[i] = x := eq_of_beq hp
      have heq_r : names[i] = r := eq_of_beq hp'
      have : x = r := heq_x.symm.trans heq_r
      exact hx_notin_xs (this ▸ hr_mem)
    · exact ih hnd_xs hxs_sub

private theorem nodup_length_le_of_subset {l₁ l₂ : List String}
    (hnd₁ : l₁.Nodup) (h : l₁ ⊆ l₂) : l₁.length ≤ l₂.length := by
  induction l₁ generalizing l₂ with
  | nil => simp
  | cons x xs ih =>
    have hx_mem : x ∈ l₂ := h (List.mem_cons_self ..)
    have hnd_xs : xs.Nodup := (List.nodup_cons.mp hnd₁).2
    have hx_not_xs : x ∉ xs := (List.nodup_cons.mp hnd₁).1
    have hxs_sub : xs ⊆ l₂.erase x := by
      intro y hy
      have hy_ne : y ≠ x := fun heq => hx_not_xs (heq ▸ hy)
      exact (List.mem_erase_of_ne hy_ne).mpr (h (List.mem_cons_of_mem x hy))
    have ih_le := ih hnd_xs hxs_sub
    have hlen_erase : (l₂.erase x).length = l₂.length - 1 :=
      List.length_erase_of_mem hx_mem
    have hlen_pos : 0 < l₂.length := List.length_pos_of_mem hx_mem
    simp only [List.length_cons]; omega

private theorem nodup_length_eq_of_subsets {l₁ l₂ : List String}
    (hnd₁ : l₁.Nodup) (hnd₂ : l₂.Nodup) (h₁₂ : l₁ ⊆ l₂) (h₂₁ : l₂ ⊆ l₁) :
    l₁.length = l₂.length :=
  Nat.le_antisymm (nodup_length_le_of_subset hnd₁ h₁₂) (nodup_length_le_of_subset hnd₂ h₂₁)

/-- The return permutation is valid when `returns` is a permutation of `Δ.keys`. -/
theorem computeReturnPerm_valid (Δ : RandomContext) (returns : List String)
    (hNodup_rctx : Δ.keys.Nodup)
    (hPerm : returns.Perm Δ.keys) :
    Permutation.Valid (computeReturnPerm Δ returns) Δ.keys.length :=
  .of_fields
    (by
      simp only [computeReturnPerm]
      have hSub := hPerm.subset
      have hNodup_ret := hPerm.nodup_iff.mpr hNodup_rctx
      have hlen : (returns.filterMap (fun r => Δ.keys.findIdx? (· == r))).length =
          returns.length := by
        apply filterMap_length_of_forall_isSome
        intro x hx
        rw [List.findIdx?_isSome]
        exact List.any_eq_true.mpr ⟨x, hSub hx, beq_self_eq_true x⟩
      rw [hlen]
      exact nodup_length_eq_of_subsets hNodup_ret hNodup_rctx hSub hPerm.symm.subset)
    (filterMap_findIdx?_nodup Δ.keys returns hNodup_rctx
      (hPerm.nodup_iff.mpr hNodup_rctx) hPerm.subset)
    (filterMap_findIdx?_forall_lt Δ.keys returns hPerm.subset)

/-! ### Shape lemma for the return permutation -/

/-- Element-level spec: the i-th entry of the filterMap/findIdx? permutation
    maps `returns[i]` to its index in `names`. -/
private theorem filterMap_findIdx?_getElem
    (names returns : List String)
    (hSub : returns ⊆ names)
    (i : Nat) (hi : i < returns.length)
    (hi' : i < (returns.filterMap (fun r => names.findIdx? (· == r))).length) :
    let j := (returns.filterMap (fun r => names.findIdx? (· == r)))[i]'hi'
    ∃ (hj : j < names.length), names[j]'hj = returns[i]'hi := by
  induction returns generalizing i with
  | nil => simp at hi
  | cons r rs ih =>
    have hr_mem := hSub (List.mem_cons_self ..)
    have hany : names.any (· == r) = true :=
      List.any_eq_true.mpr ⟨r, hr_mem, beq_self_eq_true r⟩
    rw [← List.findIdx?_isSome] at hany
    rw [Option.isSome_iff_exists] at hany
    obtain ⟨jr, hjr⟩ := hany
    simp only [List.filterMap_cons, hjr] at hi' ⊢
    cases i with
    | zero =>
      have ⟨hjr_lt, hjr_beq, _⟩ := List.findIdx?_eq_some_iff_getElem.mp hjr
      exact ⟨hjr_lt, eq_of_beq hjr_beq⟩
    | succ k =>
      simp only [List.getElem_cons_succ, List.length_cons] at hi hi' ⊢
      exact ih (fun x hx => hSub (List.mem_cons_of_mem r hx)) k (by omega) (by omega)

/-- `toShape[i]` equals the support size of the discrete type at position `i`. -/
private theorem toShape_getElem_match
    (Δ : RandomContext) (i : Nat) (hi : i < Δ.length) (hi' : i < Δ.toShape.length) :
    Δ.toShape[i]'hi' = match (Δ[i]'hi).2 with | .fin n _ => n := by
  induction Δ generalizing i with
  | nil => simp at hi
  | cons s rest ih =>
    obtain ⟨x, D⟩ := s
    cases D with
    | fin n hn =>
    cases i with
    | zero => rfl
    | succ k =>
      simp only [RandomContext.toShape, List.getElem_cons_succ]
      have hk_len : k < rest.length := by
        simp only [List.length_cons] at hi; omega
      have hk_ts : k < (↑rest : Shape).length := by
        have : k + 1 < (n :: (↑rest : Shape)).length := hi'
        simp only [List.length_cons] at this
        omega
      exact ih k hk_len hk_ts

/-- Permuting `toShape(Δ')` by the return permutation yields `toShape(Δ)`. -/
theorem toShape_permute_computeReturnPerm
    (Δ' Δ : RandomContext)
    (hperm : Δ.Perm Δ')
    (hnd : Δ.keys.Nodup)
    (hv : Permutation.Valid (computeReturnPerm Δ' Δ.keys) 𝒞⟦Δ'⟧.rank) :
    𝒞⟦Δ'⟧.permute (computeReturnPerm Δ' Δ.keys) hv = 𝒞⟦Δ⟧ := by
  have hperm_keys : Δ.keys.Perm Δ'.keys := hperm.map Sigma.fst
  have hnd' : Δ'.keys.Nodup := hperm_keys.nodup_iff.mp hnd
  apply List.ext_getElem
  · -- Length equality
    simp only [Shape.permute, List.length_pmap]
    calc (computeReturnPerm Δ' Δ.keys).length
      _ = 𝒞⟦Δ'⟧.rank := hv.length_eq
      _ = Δ'.keys.length := Δ'.toShape_length
      _ = Δ.keys.length := hperm_keys.length_eq.symm
      _ = 𝒞⟦Δ⟧.rank := Δ.toShape_length.symm
  · intro i h1 h2
    simp only [Shape.permute, List.getElem_pmap]
    -- Goal: 𝒞⟦Δ'⟧[perm[i]] = 𝒞⟦Δ⟧[i]
    have hi_keys : i < Δ.keys.length := by
      rw [← Δ.toShape_length]; exact h2
    have hi_perm : i < (computeReturnPerm Δ' Δ.keys).length := by
      rwa [Shape.permute, List.length_pmap] at h1
    -- perm[i] maps to the index of Δ.keys[i] in Δ'.keys
    have ⟨hj_lt, hj_eq⟩ := filterMap_findIdx?_getElem
      Δ'.keys Δ.keys hperm_keys.subset i hi_keys hi_perm
    -- hj_eq : Δ'.keys[perm[i]] = Δ.keys[i]
    set j := (computeReturnPerm Δ' Δ.keys)[i]'hi_perm
    have hj_ctx : j < Δ'.length := by
      rwa [show Δ'.keys.length = Δ'.length from List.length_map ..] at hj_lt
    have hi_ctx : i < Δ.length := by
      rwa [show Δ.keys.length = Δ.length from List.length_map ..] at hi_keys
    have hj_ts : j < 𝒞⟦Δ'⟧.rank := hv.forall_lt j (List.getElem_mem hi_perm)
    -- Relate toShape elements to context elements
    rw [toShape_getElem_match Δ' j hj_ctx hj_ts]
    rw [toShape_getElem_match Δ i hi_ctx h2]
    -- Goal: match (Δ'[j]).2 with ... = match (Δ[i]).2 with ...
    -- Show (Δ'[j]).2 = (Δ[i]).2 via NodupKeys
    suffices h : (Δ'[j]'hj_ctx).2 = (Δ[i]'hi_ctx).2 by rw [h]
    have hmem_i : Δ[i]'hi_ctx ∈ Δ' := hperm.mem_iff.mp (List.getElem_mem hi_ctx)
    have h_fst : (Δ'[j]'hj_ctx).1 = (Δ[i]'hi_ctx).1 := by
      have h1 : Δ'.keys[j]'hj_lt = (Δ'[j]'hj_ctx).1 := List.getElem_map ..
      have h2 : Δ.keys[i]'hi_keys = (Δ[i]'hi_ctx).1 := List.getElem_map ..
      rw [← h1, ← h2]; exact hj_eq
    -- Decompose sigma pairs and use NodupKeys
    generalize hsi : Δ[i]'hi_ctx = si at hmem_i h_fst ⊢
    generalize hsj : Δ'[j]'hj_ctx = sj at h_fst ⊢
    obtain ⟨ni, di⟩ := si
    obtain ⟨nj, dj⟩ := sj
    simp only at h_fst
    subst h_fst
    exact List.NodupKeys.eq_of_mk_mem hnd' (hsj ▸ List.getElem_mem hj_ctx) hmem_i

end Palladium.PAL.Block
