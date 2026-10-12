import Mathlib.Order.Defs.PartialOrder
import TensorLib.Shape.BroadcastAligned

namespace Palladium.TensorLib
namespace Shape

/-! ## Section 5: BroadcastCompatible — padded symmetric join predicate -/

/-- Padded symmetric joinability: both shapes are left-padded to a common rank and then checked
    for aligned compatibility. -/
def BroadcastCompatible (lhs rhs : Shape) : Prop :=
  BroadcastAlignedCompatible (pad lhs (Nat.max lhs.rank rhs.rank))
                              (pad rhs (Nat.max lhs.rank rhs.rank))

instance : Decidable (BroadcastCompatible lhs rhs) :=
  inferInstanceAs (Decidable (BroadcastAlignedCompatible ..))

/-! ## Section 6: broadcast2 -/

/-- Total padded broadcast2: left-pad both shapes to the same rank, then compute the aligned join.
    Uses `broadcastDims` so the result reduces for concrete inputs via `decide`. -/
def broadcast2 (lhs rhs : Shape) (_ : BroadcastCompatible lhs rhs) : Shape :=
  broadcastDims (pad lhs (Nat.max lhs.rank rhs.rank))
                (pad rhs (Nat.max lhs.rank rhs.rank))

/-- Bridge: `broadcast2` equals `broadcastAligned` for compatible shapes. -/
theorem broadcast2_eq_broadcastAligned {lhs rhs : Shape} (h : BroadcastCompatible lhs rhs) :
    broadcast2 lhs rhs h = broadcastAligned (pad lhs (Nat.max lhs.rank rhs.rank))
                                           (pad rhs (Nat.max lhs.rank rhs.rank)) h :=
  broadcastDims_eq_broadcastAligned h

/-! ## Section 7: Broadcastable — padded broadcastability -/

/-- `Broadcastable σₛ σₜ` means `σₛ` can be broadcast2 to `σₜ` (after left-padding), i.e. `σₛ ≤ σₜ`. -/
def Broadcastable (σₛ σₜ : Shape) : Prop :=
  BroadcastableAligned (pad σₛ σₜ.rank) σₜ

/-- Every shape is broadcastable with itself. -/
theorem Broadcastable.refl (σₛ : Shape) : Broadcastable σₛ σₛ := by
  unfold Broadcastable
  rw [pad_eq_of_rank_eq σₛ σₛ.rank rfl]
  exact BroadcastableAligned.refl σₛ

/-- Extending an aligned broadcastability proof with leading ones and an arbitrary prefix. -/
private theorem BroadcastableAligned.prepend_ones_trans {pa σₜ : Shape}
    (h1 : BroadcastableAligned pa σₜ) {n : Nat} {out : Shape}
    (h2 : BroadcastableAligned (List.replicate n 1 ++ σₜ) out) :
    BroadcastableAligned (List.replicate n 1 ++ pa) out := by
  induction n generalizing out with
  | zero => simpa using h1.trans h2
  | succ n ih =>
    simp only [List.replicate_succ, List.cons_append] at h2 ⊢
    cases h2 with
    | eq h2' => exact .one (ih h2')
    | one h2' => exact .one (ih h2')

/-- Broadcastability is transitive. -/
theorem Broadcastable.trans {σₛ mid σₜ : Shape}
    (h1 : Broadcastable σₛ mid) (h2 : Broadcastable mid σₜ) :
    Broadcastable σₛ σₜ := by
  unfold Broadcastable at *
  have hlen_ab : σₛ.rank ≤ mid.rank := by
    have := BroadcastableAligned.rank_eq h1; simp [pad_rank] at this; omega
  have hlen_bc : mid.rank ≤ σₜ.rank := by
    have := BroadcastableAligned.rank_eq h2; simp [pad_rank] at this; omega
  show BroadcastableAligned (List.replicate (σₜ.rank - σₛ.rank) 1 ++ σₛ) σₜ
  rw [show σₜ.rank - σₛ.rank = (σₜ.rank - mid.rank) + (mid.rank - σₛ.rank) from by omega,
      ← List.replicate_append_replicate, List.append_assoc]
  exact BroadcastableAligned.prepend_ones_trans h1 h2

/-- Antisymmetry: mutual broadcastability implies equality. -/
theorem Broadcastable.antisymm {σ₁ σ₂ : Shape}
    (h1 : Broadcastable σ₁ σ₂) (h2 : Broadcastable σ₂ σ₁) : σ₁ = σ₂ := by
  have hle1 : σ₁.rank ≤ σ₂.rank := by
    have := BroadcastableAligned.rank_eq h1; simp [pad_rank] at this; omega
  have hle2 : σ₂.rank ≤ σ₁.rank := by
    have := BroadcastableAligned.rank_eq h2; simp [pad_rank] at this; omega
  have heq : σ₁.rank = σ₂.rank := Nat.le_antisymm hle1 hle2
  unfold Broadcastable at h1 h2
  rw [pad_eq_of_rank_eq σ₁ σ₂.rank heq] at h1
  rw [pad_eq_of_rank_eq σ₂ σ₁.rank heq.symm] at h2
  exact BroadcastableAligned.antisymm h1 h2

/-- If σₛ is broadcastable to σₜ, then σₛ and σₜ are broadcast2-compatible. -/
theorem Broadcastable.toBroadcastCompatible {σₛ σₜ : Shape}
    (h : Broadcastable σₛ σₜ) : BroadcastCompatible σₛ σₜ := by
  have hle : σₛ.rank ≤ σₜ.rank := by
    have := BroadcastableAligned.rank_eq h; simp [pad_rank] at this; omega
  have hmax : Nat.max σₛ.rank σₜ.rank = σₜ.rank := Nat.max_eq_right hle
  unfold BroadcastCompatible
  rw [hmax, pad_eq_of_rank_eq σₜ σₜ.rank rfl]
  exact h.toBroadcastAlignedCompatible

/-- Rank of the broadcast2 result equals the max of input ranks. -/
theorem broadcast2_rank {σ₁ σ₂ : Shape} (h : BroadcastCompatible σ₁ σ₂) :
    (broadcast2 σ₁ σ₂ h).rank = Nat.max σ₁.rank σ₂.rank := by
  rw [broadcast2_eq_broadcastAligned]
  rw [broadcastAligned_rank h, pad_of_rank_le σ₁ _ (Nat.le_max_left ..)]

/-- If σₛ is broadcastable to σₜ, broadcasting σₛ with σₜ yields σₜ. -/
theorem Broadcastable.broadcast2_eq_right {σₛ σₜ : Shape}
    (h : Broadcastable σₛ σₜ) (hc : BroadcastCompatible σₛ σₜ) :
    broadcast2 σₛ σₜ hc = σₜ := by
  have hle : σₛ.rank ≤ σₜ.rank := by
    have := BroadcastableAligned.rank_eq h; simp [pad_rank] at this; omega
  have hmax : Nat.max σₛ.rank σₜ.rank = σₜ.rank := Nat.max_eq_right hle
  rw [broadcast2_eq_broadcastAligned]
  revert hc
  unfold BroadcastCompatible
  rw [hmax, pad_eq_of_rank_eq σₜ σₜ.rank rfl]
  exact fun _ => broadcastAligned_eq_right h

/-! ## Section 8: Padded bridging theorems -/

instance : Decidable (Broadcastable σₛ σₜ) :=
  inferInstanceAs (Decidable (BroadcastableAligned ..))

/-! ## Section 8b: Additional broadcast2 theorems -/

/-- If two same-length shapes are both aligned-broadcastable to a common target,
    they are aligned-compatible with each other. -/
theorem BroadcastableAligned.toCompatible_pair {σ₁ σ₂ σₜ : Shape}
    (ha : BroadcastableAligned σ₁ σₜ) (hb : BroadcastableAligned σ₂ σₜ) :
    BroadcastAlignedCompatible σ₁ σ₂ := by
  induction ha generalizing σ₂ with
  | nil => cases hb; exact .nil
  | @eq _ σₜ' d _ ih =>
    cases hb with
    | eq hb' => exact .eq (ih hb')
    | one hb' => exact .right (ih hb')
  | @one _ σₜ' d _ ih =>
    cases hb with
    | eq hb' => exact .left (ih hb')
    | one hb' => exact .eq (ih hb')

/-- Dropping a common `replicate k 1` prefix from both sides preserves aligned compatibility. -/
theorem BroadcastAlignedCompatible.drop_replicate_ones {k : Nat} {σ₁ σ₂ : Shape}
    (h : BroadcastAlignedCompatible (List.replicate k 1 ++ σ₁) (List.replicate k 1 ++ σ₂)) :
    BroadcastAlignedCompatible σ₁ σ₂ := by
  induction k with
  | zero => simpa using h
  | succ n ih =>
    simp only [List.replicate_succ, List.cons_append] at h
    exact ih h.tail

/-- The empty shape is broadcastable to any target. -/
theorem Broadcastable.nil_left (σₜ : Shape) : Broadcastable [] σₜ := by
  unfold Broadcastable pad
  simp only [List.length_nil, Nat.sub_zero]
  induction σₜ with
  | nil => exact .nil
  | cons d σₜ ih => exact .one ih

/-- Broadcastable shapes have source rank at most target rank. -/
theorem Broadcastable.rank_le {σₛ σₜ : Shape}
    (h : Broadcastable σₛ σₜ) : σₛ.rank ≤ σₜ.rank := by
  have := BroadcastableAligned.rank_eq h
  simp [pad_rank] at this; omega

/-- The aligned join of two shapes both broadcastable-aligned to a common target
    is itself broadcastable-aligned to that target. -/
theorem BroadcastableAligned.broadcastAligned_broadcastable {σ₁ σ₂ σₜ : Shape}
    (ha : BroadcastableAligned σ₁ σₜ) (hb : BroadcastableAligned σ₂ σₜ)
    (hc : BroadcastAlignedCompatible σ₁ σ₂) :
    BroadcastableAligned (broadcastAligned σ₁ σ₂ hc) σₜ := by
  induction ha generalizing σ₂ with
  | nil => cases hb; cases hc; exact .nil
  | @eq _ σₜ' d _ ih =>
    cases hb with
    | @eq _ _ _ hb' =>
      cases hc with
      | eq hc' => simp; exact .eq (ih hb' hc')
      | left hc' => simp; exact .eq (ih hb' hc')
      | right hc' => simp; exact .eq (ih hb' hc')
    | @one _ _ _ hb' =>
      cases hc with
      | eq hc' => simp; exact .eq (ih hb' hc')
      | left hc' => simp; exact .eq (ih hb' hc')
      | right hc' => simp; exact .eq (ih hb' hc')
  | @one _ σₜ' d _ ih =>
    cases hb with
    | @eq _ _ _ hb' =>
      cases hc with
      | eq hc' => simp; exact .one (ih hb' hc')
      | left hc' => simp; exact .eq (ih hb' hc')
      | right hc' => simp; exact .one (ih hb' hc')
    | @one _ _ _ hb' =>
      cases hc with
      | eq hc' => simp; exact .one (ih hb' hc')
      | left hc' => simp; exact .one (ih hb' hc')
      | right hc' => simp; exact .one (ih hb' hc')

/-- Stripping a `replicate k 1` prefix from a `BroadcastableAligned` proof. -/
theorem BroadcastableAligned.drop_prefix {k : Nat} {σₛ σₜ : Shape}
    (h : BroadcastableAligned (List.replicate k 1 ++ σₛ) σₜ) :
    BroadcastableAligned σₛ (σₜ.drop k) := by
  induction k generalizing σₜ with
  | zero => simpa using h
  | succ n ih =>
    simp only [List.replicate_succ, List.cons_append] at h
    cases h with
    | eq h' => simp only [List.drop_succ_cons]; exact ih h'
    | one h' => simp only [List.drop_succ_cons]; exact ih h'

/-- Prepending `replicate k 1` to a `BroadcastableAligned` proof. -/
theorem BroadcastableAligned.prepend_ones_of_drop {σₛ σₜ : Shape} {k : Nat}
    (h : BroadcastableAligned σₛ (σₜ.drop k))
    (hlen : σₛ.rank + k = σₜ.rank) :
    BroadcastableAligned (List.replicate k 1 ++ σₛ) σₜ := by
  induction k generalizing σₜ with
  | zero => simpa using h
  | succ n ih =>
    match σₜ with
    | [] => simp at hlen
    | d :: σₜ' =>
      simp only [List.replicate_succ, List.cons_append, List.drop_succ_cons] at h ⊢
      exact .one (ih h (by simp [rank] at hlen ⊢; omega))

/-- If two shapes are both broadcastable to a common target, then they are broadcast2-compatible
    and the broadcast2 result is also broadcastable to that target. -/
theorem Broadcastable.broadcast2_common {σ₁ σ₂ σₜ : Shape}
    (ha : Broadcastable σ₁ σₜ) (hb : Broadcastable σ₂ σₜ) :
    ∃ hc : BroadcastCompatible σ₁ σ₂, Broadcastable (broadcast2 σ₁ σ₂ hc) σₜ := by
  have ha_len := ha.rank_le
  have hb_len := hb.rank_le
  have hm_le : Nat.max σ₁.rank σ₂.rank ≤ σₜ.rank := Nat.max_le.mpr ⟨ha_len, hb_len⟩
  have hm_a : σ₁.rank ≤ Nat.max σ₁.rank σ₂.rank := Nat.le_max_left ..
  have hm_b : σ₂.rank ≤ Nat.max σ₁.rank σ₂.rank := Nat.le_max_right ..
  -- Split pad proofs
  have hpa : pad σ₁ σₜ.rank =
      List.replicate (σₜ.rank - Nat.max σ₁.rank σ₂.rank) 1 ++
      pad σ₁ (Nat.max σ₁.rank σ₂.rank) := by
    simp only [pad]
    rw [show σₜ.rank - σ₁.rank =
        (σₜ.rank - Nat.max σ₁.rank σ₂.rank) + (Nat.max σ₁.rank σ₂.rank - σ₁.rank)
        from by omega,
        ← List.replicate_append_replicate, List.append_assoc]
  have hpb : pad σ₂ σₜ.rank =
      List.replicate (σₜ.rank - Nat.max σ₁.rank σ₂.rank) 1 ++
      pad σ₂ (Nat.max σ₁.rank σ₂.rank) := by
    simp only [pad]
    rw [show σₜ.rank - σ₂.rank =
        (σₜ.rank - Nat.max σ₁.rank σ₂.rank) + (Nat.max σ₁.rank σ₂.rank - σ₂.rank)
        from by omega,
        ← List.replicate_append_replicate, List.append_assoc]
  -- Get aligned compatibility at padded level and strip prefix
  have hc_full := BroadcastableAligned.toCompatible_pair ha hb
  rw [hpa, hpb] at hc_full
  have hcompat : BroadcastCompatible σ₁ σ₂ := BroadcastAlignedCompatible.drop_replicate_ones hc_full
  -- result length
  have hsh_len : (broadcast2 σ₁ σ₂ hcompat).rank = Nat.max σ₁.rank σ₂.rank := by
    rw [broadcast2_eq_broadcastAligned, broadcastAligned_rank hcompat,
        pad_of_rank_le σ₁ _ (Nat.le_max_left ..)]
  -- Extract aligned proofs on σₜ suffix
  have ha_suffix : BroadcastableAligned (pad σ₁ (Nat.max σ₁.rank σ₂.rank))
      (σₜ.drop (σₜ.rank - Nat.max σ₁.rank σ₂.rank)) :=
    BroadcastableAligned.drop_prefix (by rw [← hpa]; exact ha)
  have hb_suffix : BroadcastableAligned (pad σ₂ (Nat.max σ₁.rank σ₂.rank))
      (σₜ.drop (σₜ.rank - Nat.max σ₁.rank σ₂.rank)) :=
    BroadcastableAligned.drop_prefix (by rw [← hpb]; exact hb)
  -- The aligned join is broadcastable to the σₜ suffix
  have hsh_suffix : BroadcastableAligned (broadcast2 σ₁ σ₂ hcompat)
      (σₜ.drop (σₜ.rank - Nat.max σ₁.rank σ₂.rank)) := by
    rw [broadcast2_eq_broadcastAligned]
    exact BroadcastableAligned.broadcastAligned_broadcastable ha_suffix hb_suffix hcompat
  -- Reconstruct full broadcastability
  have hsh_ba : Broadcastable (broadcast2 σ₁ σ₂ hcompat) σₜ := by
    show BroadcastableAligned (pad (broadcast2 σ₁ σ₂ hcompat) σₜ.rank) σₜ
    rw [show pad (broadcast2 σ₁ σ₂ hcompat) σₜ.rank =
        List.replicate (σₜ.rank - Nat.max σ₁.rank σ₂.rank) 1 ++ (broadcast2 σ₁ σ₂ hcompat)
        from by simp only [pad, hsh_len]]
    exact BroadcastableAligned.prepend_ones_of_drop hsh_suffix (by omega)
  exact ⟨hcompat, hsh_ba⟩

/-- A shape broadcastable to `σₜ` is also broadcastable to `d :: σₜ`. -/
theorem Broadcastable.cons_left {σ σₜ : Shape} (d : Nat)
    (h : Broadcastable σ σₜ) : Broadcastable σ (d :: σₜ) := by
  unfold Broadcastable at *
  show BroadcastableAligned (List.replicate (rank (d :: σₜ) - σ.rank) 1 ++ σ) (d :: σₜ)
  simp only [List.length_cons]
  have hle := h.rank_eq
  simp [pad_rank] at hle
  rw [show σₜ.rank + 1 - σ.rank = (σₜ.rank - σ.rank) + 1 from by omega,
      List.replicate_succ, List.cons_append]
  exact .one h

/-- `List.replicate k 1` is broadcastable-aligned to any shape of the same length. -/
theorem BroadcastableAligned.replicate_ones (σₜ : Shape) :
    BroadcastableAligned (List.replicate σₜ.rank 1) σₜ := by
  induction σₜ with
  | nil => exact .nil
  | cons d σₜ ih => exact .one ih

/-- `n :: List.replicate k 1` is broadcastable to `n :: σₜ` when `σₜ.rank = k`. -/
theorem Broadcastable.replicate_ones_cons (n : Nat) (σₜ : Shape) :
    Broadcastable (n :: List.replicate σₜ.rank 1) (n :: σₜ) := by
  unfold Broadcastable
  have hlen : rank (n :: List.replicate σₜ.rank 1) = rank (n :: σₜ) := by
    simp
  rw [pad_eq_of_rank_eq _ _ hlen]
  exact .eq (.replicate_ones σₜ)

/-- `List.replicate k 1` is broadcastable to any shape of length `k`. -/
theorem Broadcastable.replicate_ones (σₜ : Shape) :
    Broadcastable (List.replicate σₜ.rank 1) σₜ := by
  unfold Broadcastable
  rw [pad_eq_of_rank_eq _ _ (by simp)]
  exact BroadcastableAligned.replicate_ones σₜ

/-- Helper: extract left broadcastability from `BroadcastCompatible`. -/
theorem Broadcastable.of_broadcastCompatible_left
    {lhs rhs : Shape}
    (h : BroadcastCompatible lhs rhs) :
    Broadcastable lhs (broadcast2 lhs rhs h) := by
  unfold Broadcastable
  rw [broadcast2_rank h, broadcast2_eq_broadcastAligned]
  exact h.toBroadcastableAligned.1

/-- Helper: extract right broadcastability from `BroadcastCompatible`. -/
theorem Broadcastable.of_broadcastCompatible_right
    {lhs rhs : Shape}
    (h : BroadcastCompatible lhs rhs) :
    Broadcastable rhs (broadcast2 lhs rhs h) := by
  unfold Broadcastable
  rw [broadcast2_rank h, broadcast2_eq_broadcastAligned]
  exact h.toBroadcastableAligned.2

/-! ## Section 8c: Data-level broadcast2 helpers -/

/-- Data-level: if two shapes are broadcastable to a common target, extract the
    `BroadcastCompatible` proof with broadcastability evidence for both inputs and
    the result, as data. -/
def Broadcastable.broadcast2_common_data {σ₁ σ₂ σₜ : Shape}
    (ha : Broadcastable σ₁ σₜ) (hb : Broadcastable σ₂ σₜ) :
    { hc : BroadcastCompatible σ₁ σ₂ //
        Broadcastable σ₁ (broadcast2 σ₁ σ₂ hc) ∧
        Broadcastable σ₂ (broadcast2 σ₁ σ₂ hc) ∧
        Broadcastable (broadcast2 σ₁ σ₂ hc) σₜ } :=
  if hc : BroadcastCompatible σ₁ σ₂ then
    ⟨hc,
     Broadcastable.of_broadcastCompatible_left hc,
     Broadcastable.of_broadcastCompatible_right hc,
     by have ⟨hc', hbc⟩ := broadcast2_common ha hb
        exact broadcast2_common_data.compat_unique hc hc' ▸ hbc⟩
  else
    False.elim (hc (by
      have hcompat := BroadcastableAligned.toCompatible_pair ha hb
      have hpa := broadcast2_common_data.pad_split_replicate ha hb σ₁ (Nat.le_max_left ..)
      have hpb := broadcast2_common_data.pad_split_replicate ha hb σ₂ (Nat.le_max_right ..)
      rw [hpa, hpb] at hcompat
      exact BroadcastAlignedCompatible.drop_replicate_ones hcompat))
  where
    pad_split_replicate {σ₁ σ₂ σₜ : Shape}
        (_ : Broadcastable σ₁ σₜ) (_ : Broadcastable σ₂ σₜ)
        (s : Shape) (hs_le : s.rank ≤ Nat.max σ₁.rank σ₂.rank) :
        pad s σₜ.rank =
        List.replicate (σₜ.rank - Nat.max σ₁.rank σ₂.rank) 1 ++
        pad s (Nat.max σ₁.rank σ₂.rank) := by
      have hm_le : Nat.max σ₁.rank σ₂.rank ≤ σₜ.rank :=
        Nat.max_le.mpr ⟨‹Broadcastable σ₁ σₜ›.rank_le, ‹Broadcastable σ₂ σₜ›.rank_le⟩
      simp only [pad]
      rw [show σₜ.rank - s.rank =
          (σₜ.rank - Nat.max σ₁.rank σ₂.rank) + (Nat.max σ₁.rank σ₂.rank - s.rank)
          from by omega,
          ← List.replicate_append_replicate, List.append_assoc]
    compat_unique (hc₁ hc₂ : BroadcastCompatible σ₁ σ₂) :
        broadcast2 σ₁ σ₂ hc₁ = broadcast2 σ₁ σ₂ hc₂ := rfl

/-! ## Section 9: N-way broadcasting -/

/-- Broadcast a list of shapes pairwise from right to left. -/
private def broadcast2Fold : List Shape → Option Shape
  | [] => none
  | [s] => some s
  | s :: rest => do
      let r ← broadcast2Fold rest
      if h : BroadcastCompatible s r then some (broadcast2 s r h) else none

/-- Broadcast a vector of shapes pairwise from right to left. -/
private def broadcast2FoldN {n : Nat} (shapes : Vector Shape n) : Option Shape :=
  broadcast2Fold shapes.toList

/-- A shape in a successful `broadcast2Fold` is broadcastable to the result. -/
private theorem Broadcastable.of_broadcast2Fold {shapes : List Shape} {join : Shape}
    (h : broadcast2Fold shapes = some join) {s : Shape} (hs : s ∈ shapes) :
    Broadcastable s join := by
  induction shapes generalizing join with
  | nil => simp at hs
  | cons a rest ih =>
    cases rest with
    | nil =>
      simp [broadcast2Fold] at h
      cases hs with
      | head => subst h; exact Broadcastable.refl _
      | tail _ hmem => exact nomatch hmem
    | cons b bs =>
      simp only [broadcast2Fold] at h
      match hrest : broadcast2Fold (b :: bs) with
      | none => simp [hrest] at h
      | some srest =>
        simp only [hrest, bind, Option.bind] at h
        split at h
        · next hc =>
          injection h with h; subst h
          cases hs with
          | head => exact Broadcastable.of_broadcastCompatible_left hc
          | tail _ hmem =>
            exact (ih hrest hmem).trans (Broadcastable.of_broadcastCompatible_right hc)
        · exact Option.noConfusion h

/-- Every shape in a successful `broadcast2FoldN` is broadcastable to the result. -/
private theorem Broadcastable.of_broadcast2FoldN {n : Nat} {shapes : Vector Shape n} {join : Shape}
    (h : broadcast2FoldN shapes = some join) (i : Fin n) :
    Broadcastable shapes[i] join := by
  have hlen : i.val < shapes.toList.length := by simp [i.isLt]
  have heq : shapes.toList[i.val] = shapes[i.val] := Vector.getElem_toList hlen
  have hmem : shapes[i] ∈ shapes.toList := heq ▸ List.getElem_mem hlen
  exact of_broadcast2Fold h hmem

/-- Composing two `BroadcastCompatible` proofs yields `broadcast2FoldN` success for three shapes. -/
private theorem broadcast2FoldN_triple_compat {σ₁ σ₂ σ₃ : Shape}
    (hbc : BroadcastCompatible σ₂ σ₃) (h : BroadcastCompatible σ₁ (broadcast2 σ₂ σ₃ hbc)) :
    broadcast2FoldN #v[σ₁, σ₂, σ₃] = some (broadcast2 σ₁ (broadcast2 σ₂ σ₃ hbc) h) := by
  simp [broadcast2FoldN, broadcast2Fold, hbc, h]

/-! ## Section 10: Total N-way broadcast2 API -/

/-- N-way broadcast2 compatibility predicate. -/
abbrev BroadcastNCompatible (shapes : Vector Shape n) : Prop :=
  (broadcast2FoldN shapes).isSome

instance : Decidable (BroadcastNCompatible shapes) :=
  inferInstanceAs (Decidable (_ = true))

/-- Extract the result shape from an N-way broadcast2 when compatible. -/
@[reducible] def broadcastN (shapes : Vector Shape n) (h : BroadcastNCompatible shapes) : Shape :=
  (broadcast2FoldN shapes).get h

/-- Helper: extract broadcastability of component from `BroadcastNCompatible`. -/
theorem Broadcastable.of_broadcastNCompatible
    {n : Nat} {shapes : Vector Shape n}
    (h : BroadcastNCompatible shapes) (i : Fin n) :
    Broadcastable shapes[i] (broadcastN shapes h) :=
  Broadcastable.of_broadcast2FoldN (by simp [broadcastN, Option.some_get]) i

/-- Composing two `BroadcastCompatible` proofs yields `BroadcastNCompatible`
    for three shapes. -/
theorem BroadcastNCompatible.of_triple
    {σ₁ σ₂ σ₃ : Shape}
    (hbc : BroadcastCompatible σ₂ σ₃)
    (h : BroadcastCompatible σ₁ (broadcast2 σ₂ σ₃ hbc)) :
    BroadcastNCompatible #v[σ₁, σ₂, σ₃] := by
  simp [BroadcastNCompatible, broadcast2FoldN_triple_compat hbc h]

/-- The N-way broadcast2 of three shapes equals the two-step broadcast2. -/
theorem broadcastN_triple_eq {σ₁ σ₂ σ₃ : Shape}
    (hbc : BroadcastCompatible σ₂ σ₃) (h : BroadcastCompatible σ₁ (broadcast2 σ₂ σ₃ hbc)) :
    broadcastN #v[σ₁, σ₂, σ₃] (BroadcastNCompatible.of_triple hbc h) =
    broadcast2 σ₁ (broadcast2 σ₂ σ₃ hbc) h := by
  simp [broadcastN, broadcast2FoldN_triple_compat hbc h]

end Shape
end Palladium.TensorLib
