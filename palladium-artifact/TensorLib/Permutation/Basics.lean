import Std

/-! # Permutation Basics -/

namespace Palladium.TensorLib

/-- Permutation is a list of axis indices used to reorder a shape. -/
abbrev Permutation := List Nat

namespace Permutation

/-- Check whether `perm` is a valid permutation of `[0, ..., n-1]`. -/
def valid (perm : Permutation) (n : Nat) : Bool :=
  perm.length == n &&
    perm.all (fun i => i < n) &&
    (perm.eraseDups.length == perm.length)

open List in
/-- `eraseDups` is a sublist of the original. -/
private theorem eraseDups_sublist_self : ∀ (l : List Nat), l.eraseDups <+ l
  | [] => .slnil
  | a :: as => by
    rw [eraseDups_cons]
    exact .cons₂ a ((eraseDups_sublist_self (as.filter fun b => !(b == a))).trans filter_sublist)
termination_by l => l.length
decreasing_by
  simp_wf
  have := List.length_filter_le (fun (b : Nat) => !(b == a)) as
  omega

open List in
/-- If `eraseDups` doesn't shorten a list, the list has no duplicates. -/
private theorem nodup_of_eraseDups_length : ∀ (l : List Nat),
    l.eraseDups.length = l.length → l.Nodup
  | [], _ => nodup_nil
  | a :: as, h => by
    rw [eraseDups_cons, length_cons, length_cons] at h
    have h1 : (as.filter (fun b => !(b == a))).eraseDups.length = as.length := by omega
    have hflen := length_filter_le (fun b => !(b == a)) as
    have helen := (eraseDups_sublist_self (as.filter (fun b => !(b == a)))).length_le
    have hfeq : (as.filter (fun b => !(b == a))).length = as.length := by omega
    have hfsame : as.filter (fun b => !(b == a)) = as :=
      filter_sublist.eq_of_length hfeq
    have ha_notin : a ∉ as := by
      intro hmem
      have := (hfsame.symm ▸ hmem : a ∈ as.filter (fun b => !(b == a)))
      simp [mem_filter] at this
    have has_nodup : as.Nodup := nodup_of_eraseDups_length as (by rw [hfsame] at h1; exact h1)
    exact nodup_cons.mpr ⟨ha_notin, has_nodup⟩

/-- A valid permutation has no duplicate indices. -/
theorem valid_nodup {n : Nat} (perm : Permutation) (hvalid : valid perm n = true) :
    perm.Nodup := by
  simp only [valid, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq,
    beq_iff_eq] at hvalid
  exact nodup_of_eraseDups_length perm hvalid.2

/-- All elements of a valid permutation are less than `n`. -/
theorem valid_forall_lt {n : Nat} (perm : Permutation) (hvalid : valid perm n = true) :
    ∀ x ∈ perm, x < n := by
  simp only [valid, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hvalid
  exact hvalid.1.2

/-- A valid permutation has the expected length. -/
theorem valid_length_eq {n : Nat} (perm : Permutation) (hvalid : valid perm n = true) :
    perm.length = n := by
  simp only [valid, Bool.and_eq_true, beq_iff_eq] at hvalid
  exact hvalid.1.1

end Permutation

end Palladium.TensorLib
