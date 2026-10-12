import TensorLib.Permutation.Basics
import Batteries.Data.List.Perm

namespace Palladium.TensorLib

namespace Permutation

open List in
/-- If a list has no duplicates, `eraseDups` is the identity. -/
private theorem nodup_eraseDups_eq : ∀ (l : List Nat), l.Nodup → l.eraseDups = l
  | [], _ => rfl
  | a :: as, hnd => by
    rw [eraseDups_cons]
    have ⟨ha_notin, has_nodup⟩ := nodup_cons.mp hnd
    have : as.filter (fun b => !(b == a)) = as := by
      apply List.filter_eq_self.mpr
      intro x hx
      simp
      exact fun heq => ha_notin (heq ▸ hx)
    rw [this]
    congr 1
    exact nodup_eraseDups_eq as has_nodup

/-- `Valid perm n` means `perm` is a valid permutation of `[0, ..., n-1]`,
    i.e. `perm` is a permutation of `List.range n`. -/
structure Valid (perm : Permutation) (n : Nat) : Prop where
  /-- The underlying `List.Perm` witness. -/
  perm : perm.Perm (List.range n)

/-- The permutation has the right length. -/
theorem Valid.length_eq {perm : Permutation} {n : Nat} (h : Valid perm n) :
    perm.length = n :=
  h.perm.length_eq.trans List.length_range

/-- The permutation has no duplicate indices. -/
theorem Valid.nodup {perm : Permutation} {n : Nat} (h : Valid perm n) :
    perm.Nodup :=
  h.perm.nodup_iff.mpr List.nodup_range

/-- All elements are less than `n`. -/
theorem Valid.forall_lt {perm : Permutation} {n : Nat} (h : Valid perm n) :
    ∀ x ∈ perm, x < n :=
  fun _ hx => List.mem_range.mp (h.perm.subset hx)

/-- Construct `Valid` from length, nodup, and forall_lt proofs. -/
theorem Valid.of_fields {perm : Permutation} {n : Nat}
    (hlen : perm.length = n) (hnd : perm.Nodup) (hlt : ∀ x ∈ perm, x < n) :
    Valid perm n :=
  ⟨(List.subperm_of_subset hnd (fun x hx => List.mem_range.mpr (hlt x hx))).perm_of_length_le
    (by simp [hlen])⟩

/-- Construct `Valid` from the boolean validity check. -/
theorem Valid.of_valid {perm : Permutation} {n : Nat}
    (hv : perm.valid n = true) : Valid perm n :=
  .of_fields (valid_length_eq perm hv) (valid_nodup perm hv) (valid_forall_lt perm hv)

/-- Extract the boolean validity check from `Valid`. -/
theorem Valid.to_valid {perm : Permutation} {n : Nat} (h : Valid perm n) :
    perm.valid n = true := by
  simp only [valid, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq, beq_iff_eq]
  exact ⟨⟨h.length_eq, h.length_eq ▸ h.forall_lt⟩, by rw [nodup_eraseDups_eq perm h.nodup]⟩

instance : Decidable (Valid perm n) :=
  if hv : perm.valid n then .isTrue (.of_valid hv)
  else .isFalse (fun h => hv h.to_valid)

/-- Compute the inverse of a valid permutation.
    Note: `getD` is kept inside `find?`'s predicate because `find?` does not pass
    membership proofs to its callback. The outer `Option.getD 0` handles the `find?`
    returning `none` (which never occurs for valid permutations). -/
def invert (perm : Permutation) {n : Nat} (_h : Valid perm n) : Permutation :=
  let n := perm.length
  (List.range n).map (fun j =>
    (List.range n).find? (fun i => perm.getD i 0 == j) |>.getD 0)

/-- Convenience: compute inverse via the validity proof. -/
def Valid.invert {perm : Permutation} {n : Nat} (h : Valid perm n) : Permutation :=
  perm.invert h

/-! ## Inverse permutation properties -/

/-- The inverse of a valid permutation has length `n`. -/
theorem Valid.invert_length {perm : Permutation} {n : Nat} (h : Valid perm n) :
    (perm.invert h).length = n := by
  unfold Permutation.invert
  simp [h.length_eq]

/-- All elements of the inverse permutation are less than `n`. -/
theorem Valid.invert_forall_lt {perm : Permutation} {n : Nat} (h : Valid perm n) :
    ∀ x ∈ (perm.invert h), x < n := by
  intro x hx
  unfold Permutation.invert at hx
  simp only [List.mem_map, List.mem_range] at hx
  obtain ⟨j, hj, rfl⟩ := hx
  have hlen := h.length_eq
  cases hf : (List.range perm.length).find? (fun i => perm.getD i 0 == j) with
  | some val =>
    simp only [Option.getD]
    have hmem := List.mem_of_find?_eq_some hf
    rw [List.mem_range] at hmem
    omega
  | none =>
    simp only [Option.getD]
    omega

/-- `perm[inv[j]] = j` for any `j < n`. -/
theorem Valid.invert_spec {perm : Permutation} {n : Nat} (h : Valid perm n)
    (j : Nat) (hj : j < n) :
    perm[(perm.invert h)[j]'(by rw [h.invert_length]; exact hj)]'(by
      rw [h.length_eq]; exact h.invert_forall_lt _ (List.getElem_mem ..)) = j := by
  -- Suffices to prove the getD-based form, then bridge to bracket indexing
  suffices hs : perm.getD ((perm.invert h).getD j 0) 0 = j by
    have hj_inv : j < (perm.invert h).length := by rw [h.invert_length]; exact hj
    have h1 : (perm.invert h).getD j 0 = (perm.invert h)[j]'hj_inv := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj_inv, Option.getD_some]
    rw [h1] at hs
    have hperm_lt : (perm.invert h)[j]'hj_inv < perm.length := by
      rw [h.length_eq]; exact h.invert_forall_lt _ (List.getElem_mem hj_inv)
    have h2 : perm.getD ((perm.invert h)[j]'hj_inv) 0 =
        perm[(perm.invert h)[j]'hj_inv]'hperm_lt := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hperm_lt, Option.getD_some]
    rw [h2] at hs
    exact hs
  -- Prove the getD form directly
  have hj_mem : j ∈ perm := h.perm.symm.subset (List.mem_range.mpr hj)
  have ⟨idx, hidx_lt, hidx_eq⟩ := List.mem_iff_getElem.mp hj_mem
  -- Reduce (perm.invert h).getD j 0 to the find? expression
  dsimp only [Permutation.invert]
  have hinner : (List.map (fun j =>
      ((List.range perm.length).find? (fun i => perm.getD i 0 == j)).getD 0)
    (List.range perm.length)).getD j 0 =
    ((List.range perm.length).find? (fun i => perm.getD i 0 == j)).getD 0 := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      List.getElem?_range (h.length_eq ▸ hj), Option.map_some, Option.getD_some]
  rw [hinner]
  cases hf : (List.range perm.length).find? (fun i => perm.getD i 0 == j) with
  | some val =>
    simp only [Option.getD]
    have h_find := List.find?_some hf
    simp only [beq_iff_eq] at h_find
    exact h_find
  | none =>
    exfalso
    have hpred : (perm.getD idx 0 == j) = true := by
      simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hidx_lt, hidx_eq]
    exact absurd hpred (List.find?_eq_none.mp hf idx (List.mem_range.mpr hidx_lt))

end Permutation

end Palladium.TensorLib
