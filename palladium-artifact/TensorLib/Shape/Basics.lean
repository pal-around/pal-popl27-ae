import TensorLib.Permutation
import Mathlib.Data.List.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.List.Sort
import Mathlib.Deprecated.Sort
import Mathlib.Order.Monotone.Defs

namespace Palladium.TensorLib

/-- Shape is a list of dimension sizes, outermost to innermost. -/
abbrev Shape := List Nat

namespace Shape

/-- Number of axes (rank) of a shape. -/
abbrev rank (σ : Shape) : Nat := σ.length

/-- Remove an axis from a shape. -/
abbrev slice (σ : Shape) (axis : Fin σ.rank) : Shape := σ.eraseIdx axis.val

/-- Sort axis indices by strictly decreasing value (largest axis first).

Uses insertion sort with the total preorder `a.val ≥ b.val`, matching `List.insertionSort`
in Mathlib so `pairwise_insertionSort` applies. -/
def sortAxesDesc {σ : Shape} (axes : List (Fin σ.rank)) : List (Fin σ.rank) :=
  axes.insertionSort (fun a b => a.val ≥ b.val)

/-- Remove every dimension whose index appears in `axes`. -/
def removeAxes (σ : Shape) (axes : List (Fin σ.rank)) : Shape :=
  (List.range σ.length).filterMap (fun i =>
    if hi : i < σ.length then
      if axes.any (fun a => a.val = i) then none
      else some (σ[i]'hi)
    else none)

private theorem any_eq_of_perm {σ : Shape} (l1 l2 : List (Fin σ.rank)) (h : l1.Perm l2)
    (i : Nat) : (l1.any fun a => a.val = i) = (l2.any fun a => a.val = i) := by
  induction h with
  | nil => rfl
  | cons x h ih => simp [List.any_cons, ih]
  | swap x y l => by_cases hx : x.val = i <;> by_cases hy : y.val = i <;> simp [hx, hy, List.any_cons]
  | trans h12 h23 ih12 ih23 => rw [ih12, ih23]

/-- Permuting axis lists does not change `removeAxes`. -/
theorem removeAxes_perm {σ : Shape} (l1 l2 : List (Fin σ.rank)) (h : l1.Perm l2) :
    removeAxes σ l1 = removeAxes σ l2 := by
  simp only [removeAxes]
  refine List.filterMap_congr ?_
  intro i hi
  simp only [List.mem_range] at hi
  have hi' : i < σ.length := hi
  simp only [hi', ↓reduceDIte, any_eq_of_perm l1 l2 h i]

instance axesDescIsTrans {σ : Shape} : IsTrans (Fin σ.rank) (fun a b => a.val ≥ b.val) where
  trans _ _ _ hab hbc := Nat.le_trans hbc hab

instance axesDescIsTotal {σ : Shape} : IsTotal (Fin σ.rank) (fun a b => a.val ≥ b.val) where
  total a b := (Nat.le_total b.val a.val).imp (fun h => h) (fun h => h)

/-- `sortAxesDesc` is a permutation of the input. -/
theorem perm_sortAxesDesc {σ : Shape} (axes : List (Fin σ.rank)) :
    axes.Perm (sortAxesDesc axes) := by
  simpa [sortAxesDesc] using (List.perm_insertionSort (fun a b => a.val ≥ b.val) axes).symm

/-- `sortAxesDesc` preserves `List.Nodup`. -/
theorem nodup_sortAxesDesc {σ : Shape} (axes : List (Fin σ.rank)) (h : axes.Nodup) :
    (sortAxesDesc axes).Nodup :=
  List.Nodup.perm h (perm_sortAxesDesc axes)

private theorem pairwise_gt_of_pairwise_ge_nodup {σ : Shape} (l : List (Fin σ.rank))
    (hge : l.Pairwise (fun a b => a.val ≥ b.val)) (hn : l.Nodup) :
    l.Pairwise (fun a b => a.val > b.val) := by
  induction l with
  | nil => exact List.Pairwise.nil
  | cons a as ih =>
    rcases List.pairwise_cons.mp hge with ⟨hfor, hpwas⟩
    rcases List.nodup_cons.mp hn with ⟨hna, hnas⟩
    refine List.pairwise_cons.mpr ⟨?_, ih hpwas hnas⟩
    intro b hb
    have hle : b.val ≤ a.val := hfor b hb
    have hne : b ≠ a := by
      intro rid
      subst rid
      exact hna hb
    exact Nat.lt_of_le_of_ne hle (fun he => hne (Fin.ext he))

/-- After sorting, consecutive axes are strictly decreasing in value when the input has no
duplicates. -/
theorem pairwise_sortAxesDesc {σ : Shape} (axes : List (Fin σ.rank)) (h : axes.Nodup) :
    (sortAxesDesc axes).Pairwise (fun a b => a.val > b.val) :=
  pairwise_gt_of_pairwise_ge_nodup _ (List.pairwise_insertionSort (fun a b => a.val ≥ b.val) axes)
    (nodup_sortAxesDesc axes h)

private theorem range_map_getD (xs : List Nat) :
    (List.range xs.length).map (xs.getD · 0) = xs := by
  apply List.ext_getElem
  · simp
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_range]
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h2]

private theorem filterMap_range_eraseIdx (σ : List Nat) (k : Nat) (hk : k < σ.length) :
    (List.range σ.length).filterMap (fun i =>
        if hi : i < σ.length then if i = k then none else some (getElem σ i hi) else none) =
      σ.eraseIdx k := by
  induction σ generalizing k with
  | nil => cases hk
  | cons x xs ih =>
    rw [List.length_cons] at hk
    cases k with
    | zero =>
      simp [List.length_cons, List.range_succ_eq_map]
      refine Eq.trans ?_ (range_map_getD xs)
      refine (List.filterMap_eq_map_iff_forall_eq_some (l := List.range xs.length)).mpr ?_
      intro j hj
      simp only [List.mem_range] at hj
      simp [hj]
    | succ k' =>
      have hk' : k' < xs.length := Nat.succ_lt_succ_iff.mp hk
      simp [List.length_cons, List.range_succ_eq_map, List.eraseIdx_cons_succ]
      refine Eq.trans ?_ (ih k' hk')
      refine List.filterMap_congr ?_
      intro j hj
      simp only [List.mem_range] at hj
      have hi : Nat.succ j < (x :: xs).length := Nat.succ_lt_succ hj
      simp [hj, Function.comp_apply]

/-- Removing a single axis matches `slice`. -/
theorem removeAxes_singleton (σ : Shape) (a : Fin σ.rank) :
    removeAxes σ [a] = σ.slice a := by
  rw [slice, removeAxes]
  refine Eq.trans ?_ (filterMap_range_eraseIdx σ a.val a.isLt)
  refine List.filterMap_congr ?_
  intro i hi
  simp only [List.mem_range] at hi
  have hi' : i < σ.length := hi
  simp only [hi', ↓reduceDIte]
  by_cases h : i = a.val <;> simp [h, List.any_cons, List.any_nil, eq_comm]

/-- Map tail axes strictly below `head` into `Fin (σ.slice head).rank`. -/
def mapFinsBelowHead {σ : Shape} (head : Fin σ.rank) (rest : List (Fin σ.rank))
    (htail : ∀ x ∈ rest, x.val < head.val) : List (Fin (σ.slice head).rank) :=
  rest.attach.map (fun ⟨x, hx⟩ =>
    ⟨x.val, by
      have hx' := htail x hx
      simp only [slice, List.length_eraseIdx, head.isLt]
      exact Nat.lt_of_lt_of_le hx' (Nat.le_sub_one_of_lt head.isLt)⟩)

@[simp] theorem mapFinsBelowHead_nil {σ : Shape} (head : Fin σ.rank)
    (htail : ∀ x ∈ ([] : List (Fin σ.rank)), x.val < head.val) :
    mapFinsBelowHead head [] htail = [] := by
  simp [mapFinsBelowHead]

theorem mapFinsBelowHead_cons {σ : Shape} (a x : Fin σ.rank) (xs : List (Fin σ.rank))
    (hall : ∀ y ∈ x :: xs, y.val < a.val) :
    mapFinsBelowHead a (x :: xs) hall =
      ⟨x.val, by
        simpa [slice, List.length_eraseIdx, a.isLt] using
          Nat.lt_of_lt_of_le (hall x (List.mem_cons_self ..))
            (Nat.le_sub_one_of_lt a.isLt)⟩ ::
        mapFinsBelowHead a xs (fun y hy => hall y (List.mem_cons_of_mem _ hy)) := by
  simp [mapFinsBelowHead, List.attach_cons, List.map_cons]

/-- Descending `Fin.val` pairwise order on `head :: rest` is preserved on `mapFinsBelowHead`. -/
theorem pairwise_mapFinsBelowHead {σ : Shape} {a : Fin σ.rank} (as : List (Fin σ.rank))
    (hp : (a :: as).Pairwise (fun x y => x.val > y.val))
    (hall : ∀ x ∈ as, x.val < a.val) :
    (mapFinsBelowHead a as hall).Pairwise (fun x y => x.val > y.val) := by
  induction as with
  | nil => simp [mapFinsBelowHead]
  | cons x xs ih =>
    rcases List.pairwise_cons.mp hp with ⟨hax, hpw⟩
    rcases List.pairwise_cons.mp hpw with ⟨hxfor, hxpw⟩
    have hallCons : ∀ y ∈ x :: xs, y.val < a.val := fun y hy => hax y hy
    rw [mapFinsBelowHead_cons a x xs hallCons]
    have ha₂ : (a :: xs).Pairwise (fun x y => x.val > y.val) :=
      List.pairwise_cons.mpr ⟨fun y hy => hax y (List.mem_cons_of_mem _ hy), hxpw⟩
    have hallXs : ∀ y ∈ xs, y.val < a.val := fun y hy => hallCons y (List.mem_cons_of_mem _ hy)
    refine List.pairwise_cons.mpr ⟨?_, ih ha₂ hallXs⟩
    intro b hb
    simp only [mapFinsBelowHead, List.mem_map, List.mem_attach, Subtype.exists] at hb
    obtain ⟨y, hy, ⟨_, hbe⟩⟩ := hb
    rw [← hbe]
    exact hxfor y hy

theorem removeAxes_perm_sortAxesDesc {σ : Shape} (axes : List (Fin σ.rank)) :
    removeAxes σ axes = removeAxes σ (sortAxesDesc axes) :=
  removeAxes_perm axes (sortAxesDesc axes) (perm_sortAxesDesc axes)

/-- With no axes to remove, every dimension is kept. -/
theorem removeAxes_nil (σ : Shape) : removeAxes σ [] = σ := by
  simp [removeAxes]
  refine Eq.trans (List.filterMap_eq_map_iff_forall_eq_some.mpr ?_) (range_map_getD σ)
  intro i hi
  simp only [List.mem_range] at hi
  simp [hi, List.getD_eq_getElem?_getD]

/-- Original axis index in `σ` corresponding to position `j` in `σ.eraseIdx a`. -/
abbrev origIdxOfSlice (a : Nat) (j : Nat) : Nat :=
  if j < a then j else j + 1

private theorem origIdxOfSlice_lt_length {σ : Shape} (a : Fin σ.rank) (j : Nat)
    (hj : j < (σ.slice a).length) : origIdxOfSlice a.val j < σ.length := by
  have hax : a.val < σ.length := a.isLt
  have hlen : (σ.slice a).length = σ.length - 1 := by simp [slice, List.length_eraseIdx, hax]
  rw [hlen] at hj
  dsimp [origIdxOfSlice]
  split_ifs with hlt
  · have hpos : 0 < σ.length := Nat.pos_of_ne_zero (fun h0 => by simp [h0] at hax)
    omega
  · have hpos : 0 < σ.length := Nat.pos_of_ne_zero (fun h0 => by simp [h0] at hax)
    omega

private theorem mapFinsBelowHead_map_val {σ : Shape} (a : Fin σ.rank) (as : List (Fin σ.rank))
    (hall : ∀ x ∈ as, x.val < a.val) :
    (mapFinsBelowHead a as hall).map (fun b => b.val) = as.map (fun x => x.val) := by
  induction as with
  | nil => rfl
  | cons x xs ih =>
    have hall' : ∀ y ∈ xs, y.val < a.val := fun y hy => hall y (List.mem_cons_of_mem _ hy)
    simp [mapFinsBelowHead, List.attach_cons, List.map_cons, List.map_map, Function.comp_apply]

private theorem any_mapFinsBelowHead_val {σ : Shape} (a : Fin σ.rank) (as : List (Fin σ.rank))
    (hall : ∀ x ∈ as, x.val < a.val) (j : Nat) :
    (mapFinsBelowHead a as hall).any (fun b => decide (b.val = j)) =
      as.any (fun x => decide (x.val = j)) := by
  simpa using congrArg (fun l : List Nat => l.any (fun v => decide (v = j))) (mapFinsBelowHead_map_val a as hall)

private theorem any_cons_eq_orig_of_slice {σ : Shape} (a : Fin σ.rank) (as : List (Fin σ.rank))
    (hall : ∀ x ∈ as, x.val < a.val) (j : Nat) (hj : j < (σ.slice a).length) :
    (a :: as).any (fun z => decide (z.val = origIdxOfSlice a.val j)) =
      (mapFinsBelowHead a as hall).any (fun b => decide (b.val = j)) := by
  rcases Nat.lt_trichotomy j a.val with hjlt | rfl | hjgt
  · -- j < a.val
    have horig : origIdxOfSlice a.val j = j := if_pos hjlt
    rw [horig, any_mapFinsBelowHead_val a as hall j]
    simp [List.any_cons]
    have ha : ¬ a.val = j := by omega
    simp [ha]
  · -- j = a.val
    have horig : origIdxOfSlice a.val a.val = a.val + 1 := by
      simp [origIdxOfSlice]
    rw [horig]
    have hall_lt : ∀ x ∈ as, x.val < a.val := hall
    have has_ne : as.any (fun x => decide (x.val = a.val + 1)) = false := by
      rw [List.any_eq_false]
      intro x hx
      have hxv : x.val < a.val := hall_lt x hx
      intro he
      rw [decide_eq_true_iff] at he
      omega
    have hL : (a :: as).any (fun z => decide (z.val = a.val + 1)) = false := by
      simp [List.any_cons, has_ne]
    have hR : (mapFinsBelowHead a as hall).any (fun b => decide (b.val = a.val)) = false := by
      rw [any_mapFinsBelowHead_val a as hall a.val]
      rw [List.any_eq_false]
      intro x hx he
      rw [decide_eq_true_iff] at he
      have hxv : x.val < a.val := hall x hx
      omega
    rw [hL, hR]
  · -- j > a.val
    have horig : origIdxOfSlice a.val j = j + 1 := by
      dsimp [origIdxOfSlice]
      split_ifs with h
      · omega
      · rfl
    rw [horig]
    have has_ne : as.any (fun x => decide (x.val = j + 1)) = false := by
      rw [List.any_eq_false]
      intro x hx
      have hxv : x.val < a.val := hall x hx
      intro he
      rw [decide_eq_true_iff] at he
      omega
    have hL : (a :: as).any (fun z => decide (z.val = j + 1)) = false := by
      simp [List.any_cons, has_ne, show a.val ≠ j + 1 by omega]
    have hR : (mapFinsBelowHead a as hall).any (fun b => decide (b.val = j)) = false := by
      rw [any_mapFinsBelowHead_val a as hall j]
      rw [List.any_eq_false]
      intro x hx he
      rw [decide_eq_true_iff] at he
      have hxv : x.val < a.val := hall x hx
      omega
    rw [hL, hR]

/-- Index into `σ.eraseIdx a` corresponding to original axis `i ≠ a`. -/
abbrev sliceIdxOfOrig (a : Nat) (i : Nat) : Nat :=
  if i < a then i else i - 1

private theorem sliceIdx_origIdx (a j : Nat) : sliceIdxOfOrig a (origIdxOfSlice a j) = j := by
  rcases Nat.lt_trichotomy j a with hjlt | rfl | hjgt
  · simp [origIdxOfSlice, sliceIdxOfOrig, hjlt]
  · simp [origIdxOfSlice, sliceIdxOfOrig]
  · have hj' : ¬ j < a := Nat.not_lt.mpr (Nat.le_of_lt hjgt)
    have ho : origIdxOfSlice a j = j + 1 := by
      simp [origIdxOfSlice, hj']
    rw [ho]
    have hp : ¬ j + 1 < a := by omega
    simp [sliceIdxOfOrig, hp]

private theorem origIdx_sliceIdx_of_ne {a i : Nat} (hne : i ≠ a) :
    origIdxOfSlice a (sliceIdxOfOrig a i) = i := by
  rcases Nat.lt_trichotomy i a with hi | rfl | hi
  · simp [origIdxOfSlice, sliceIdxOfOrig, hi]
  · contradiction
  · have hi' : ¬ i < a := Nat.not_lt.mpr (Nat.le_of_lt hi)
    have hs : sliceIdxOfOrig a i = i - 1 := by
      simp [sliceIdxOfOrig, hi']
    rw [hs]
    have hsub : ¬ i - 1 < a := by omega
    simp [origIdxOfSlice, hsub]
    omega

private theorem sliceIdx_lt_eraseIdx_length {σ : Shape} {a : Fin σ.rank} {i : Nat}
    (hi : i < σ.length) (hne : i ≠ a.val) (hax : a.val < σ.length) :
    sliceIdxOfOrig a.val i < (List.eraseIdx σ a.val).length := by
  dsimp [sliceIdxOfOrig]
  split_ifs <;> simp [List.length_eraseIdx, hax] <;> omega

private theorem length_filter_range_eq_finset_card (n : Nat) (p : Nat → Bool) :
    ((List.range n).filter p).length =
      ((Finset.range n).filter (fun i => p i)).card := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [List.range_succ, List.filter_append, List.filter_singleton]
    rw [Finset.range_add_one]
    simp only [Finset.filter_insert]
    cases h : p k <;> simp [h, ih]

private theorem sortedLT_filter_range (n : Nat) (p : Nat → Bool) :
    ((List.range n).filter p).SortedLT := by
  rw [List.sortedLT_iff_pairwise]
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.filter_append, List.filter_singleton]
    cases hpn : p n <;> simp
    · simpa [List.pairwise_append] using ih
    · rw [List.pairwise_append]
      refine ⟨ih, ?_, ?_⟩
      · simp
      · intro a ha b hb
        have ha' : a < n := by
          have := (List.mem_filter.mp ha).1
          simpa [List.mem_range] using this
        have hb' : b = n := List.mem_singleton.mp hb
        rw [hb']
        exact ha'

private theorem origIdxOfSlice_strictMono (a : Nat) : StrictMono (origIdxOfSlice a) := by
  intro i j hij
  by_cases hi : i < a <;> by_cases hj : j < a <;> simp [origIdxOfSlice, hi, hj] <;> omega

private theorem filter_filterMap_eq_map {α β : Type} (l : List α) (p : α → Bool) (f : α → β) :
    l.filterMap (fun x => if p x then some (f x) else none) = (l.filter p).map f := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.filterMap_cons, List.filter_cons]
    split_ifs with hpx <;> simp [ih]

private theorem removeAxes_eq_map_filter (σ : Shape) (axes : List (Fin σ.rank)) :
    removeAxes σ axes =
      ((List.range σ.length).filter (fun i => !(axes.any (fun a => a.val = i))) |>.map
        (fun i => σ[i]!)) := by
  classical
  rw [removeAxes]
  refine Eq.trans (List.filterMap_congr ?_) (filter_filterMap_eq_map (List.range σ.length) _ _)
  intro i hi
  simp only [List.mem_range] at hi
  by_cases h : axes.any fun a => a.val = i
  · simp [h, hi, ↓reduceDIte]
  · simp [h, hi, ↓reduceDIte]

private theorem removeAxes_length_eq_filter_length (σ : Shape) (axes : List (Fin σ.rank)) :
    (removeAxes σ axes).length =
      ((List.range σ.length).filter (fun i => !(axes.any (fun a => a.val = i)))).length := by
  rw [removeAxes_eq_map_filter]
  simp [List.length_map]

private theorem removeAxes_length_eq_card (σ : Shape) (axes : List (Fin σ.rank)) :
    (removeAxes σ axes).length =
      ((Finset.range σ.length).filter (fun i => !(axes.any (fun a => a.val = i)))).card := by
  rw [removeAxes_length_eq_filter_length, length_filter_range_eq_finset_card]

private theorem filter_kept_eq_map_origIdx {σ : Shape} (a : Fin σ.rank) (as : List (Fin σ.rank))
    (hall : ∀ x ∈ as, x.val < a.val) :
    (List.range σ.length).filter (fun i => ! (a :: as).any (fun z => z.val = i)) =
      List.map (origIdxOfSlice a.val)
        ((List.range (σ.slice a).length).filter
          (fun j => ! (mapFinsBelowHead a as hall).any (fun b => b.val = j))) := by
  classical
  let bs := mapFinsBelowHead a as hall
  let lL := (List.range σ.length).filter (fun i => ! (a :: as).any (fun z => z.val = i))
  let lR :=
    (List.range (σ.slice a).length).filter (fun j => ! bs.any (fun b => b.val = j))
  have hsortL : lL.SortedLT := sortedLT_filter_range σ.length _
  have hsortR : lR.SortedLT := sortedLT_filter_range (σ.slice a).length _
  have hsortM : (List.map (origIdxOfSlice a.val) lR).SortedLT :=
    (origIdxOfSlice_strictMono a.val).sortedLT_listMap.2 hsortR
  refine List.SortedLT.eq_of_mem_iff hsortL hsortM ?_
  intro x
  constructor
  · intro hx
    simp only [List.mem_filter, List.mem_range, List.mem_map] at hx ⊢
    rcases hx with ⟨hxσ, hanyL⟩
    have hanyL' : (a :: as).any (fun z => decide (z.val = x)) = false := by
      simpa [Bool.not_eq_true] using hanyL
    have hxne : x ≠ a.val := by
      intro rid
      subst rid
      have : (a :: as).any (fun z => decide (z.val = a.val)) := by simp [List.any_cons]
      simp at hanyL'
    have hj : sliceIdxOfOrig a.val x < (σ.slice a).length := by
      simpa [slice, List.length_eraseIdx, a.isLt] using
        sliceIdx_lt_eraseIdx_length hxσ hxne a.isLt
    have hkey := any_cons_eq_orig_of_slice (σ := σ) a as hall (sliceIdxOfOrig a.val x) hj
    have hkey' := by simpa [origIdx_sliceIdx_of_ne hxne] using hkey
    have hbs : bs.any (fun b => decide (b.val = sliceIdxOfOrig a.val x)) = false := by
      simpa [hkey'] using hanyL'
    refine ⟨sliceIdxOfOrig a.val x, ?_, origIdx_sliceIdx_of_ne hxne⟩
    refine And.intro ?_ ?_
    · simpa [slice, List.length_eraseIdx, a.isLt] using hj
    · simpa [Bool.not_eq_true] using hbs
  · intro hx
    rcases List.mem_map.mp hx with ⟨j, hjR, heq⟩
    rw [← heq]
    rcases List.mem_filter.mp hjR with ⟨hjσ, hjany⟩
    have hj' : j < (σ.slice a).length := by simpa [List.mem_range] using hjσ
    simp only [List.mem_filter, List.mem_range]
    refine And.intro ?_ ?_
    · simp [origIdxOfSlice_lt_length a j hj']
    · simp [any_cons_eq_orig_of_slice (σ := σ) a as hall j hj', hjany]

private theorem removeAxes_cons_max_getElem (σ : Shape) (a : Fin σ.rank) (as : List (Fin σ.rank))
    (hall : ∀ x ∈ as, x.val < a.val) (t : Nat) (htL : t < (removeAxes σ (a :: as)).length)
    (htR : t < (removeAxes (σ.slice a) (mapFinsBelowHead a as hall)).length)
    (_hlen : (removeAxes σ (a :: as)).length = (removeAxes (σ.slice a) (mapFinsBelowHead a as hall)).length) :
    (removeAxes σ (a :: as))[t]'htL = (removeAxes (σ.slice a) (mapFinsBelowHead a as hall))[t]'htR := by
  classical
  let bs := mapFinsBelowHead a as hall
  have hlist := filter_kept_eq_map_origIdx (σ := σ) a as hall
  let filtR :=
    (List.range (σ.slice a).length).filter (fun j => ! bs.any (fun b => b.val = j))
  have hEqL : removeAxes σ (a :: as) = List.map (fun j => σ[origIdxOfSlice a.val j]!) filtR := by
    calc
      removeAxes σ (a :: as) =
          List.map (fun i => σ[i]!)
            ((List.range σ.length).filter (fun i => ! (a :: as).any (fun z => z.val = i))) :=
        removeAxes_eq_map_filter σ (a :: as)
      _ = List.map (fun j => σ[origIdxOfSlice a.val j]!) filtR := by
            rw [hlist, List.map_map]
            rfl
  have hEqR : removeAxes (σ.slice a) bs = List.map (fun j => (σ.slice a)[j]!) filtR := by
    simpa [filtR] using removeAxes_eq_map_filter (σ.slice a) bs
  rw [List.getElem_of_eq hEqL htL, List.getElem_of_eq hEqR htR]
  rw [List.getElem_map, List.getElem_map]
  have htR' : t < filtR.length := by
    rw [hEqR] at htR
    simp [List.length_map] at htR
    exact htR
  have hj₀ : filtR[t]'htR' < (σ.slice a).length := by
    have hmem := List.getElem_mem (h := htR')
    have hrng := (List.mem_filter.mp hmem).1
    simpa [List.mem_range, slice, List.length_eraseIdx, a.isLt] using hrng
  let j := filtR[t]'htR'
  have hidx : origIdxOfSlice a.val j < σ.length := origIdxOfSlice_lt_length a j hj₀
  have horig : σ[origIdxOfSlice a.val j]'hidx = (σ.slice a)[j]'hj₀ := by
    dsimp [slice]
    rcases Nat.lt_or_ge j a.val with hlt | hge
    · have hoo : origIdxOfSlice a.val j = j := if_pos hlt
      simp [hoo]
      exact (List.getElem_eraseIdx_of_lt (l := σ) (i := a.val) (j := j) (h := hj₀) hlt).symm
    · have ho : origIdxOfSlice a.val j = j + 1 := by
        dsimp [origIdxOfSlice]
        split_ifs with h
        · omega
        · rfl
      simp [ho]
      simpa using (List.getElem_eraseIdx_of_ge (l := σ) (i := a.val) (j := j) (h := hj₀) hge).symm
  simp [j, List.getElem!_eq_getElem?_getD, List.getElem?_eq_getElem hidx, List.getElem?_eq_getElem hj₀,
    horig]

/-- Removing the maximum axis first, then the rest (reindexed below `head`), agrees with
removing `head :: rest` in one pass. -/
theorem removeAxes_cons_max {σ : Shape} (a : Fin σ.rank) (as : List (Fin σ.rank))
    (hall : ∀ x ∈ as, x.val < a.val) :
    removeAxes σ (a :: as) = removeAxes (σ.slice a) (mapFinsBelowHead a as hall) := by
  classical
  have hax : a.val < σ.length := a.isLt
  have hcard :
      ((Finset.range σ.length).filter (fun i => ! (a :: as).any (fun z => z.val = i))).card =
        ((Finset.range (List.eraseIdx σ a.val).length).filter
            (fun j => ! (mapFinsBelowHead a as hall).any (fun b => b.val = j))).card := by
    refine Finset.card_nbij' (sliceIdxOfOrig a.val) (origIdxOfSlice a.val) ?_ ?_ ?_ ?_
    · intro j hj
      rcases Finset.mem_filter.mp (Finset.mem_coe.mp hj) with ⟨hjlt, hjany⟩
      have hjσ : j < σ.length := Finset.mem_range.mp hjlt
      have hjne : j ≠ a.val := by
        intro rid
        subst rid
        have ht : (a :: as).any (fun z => decide (z.val = a.val)) = true := by simp [List.any_cons]
        rw [ht] at hjany
        simp at hjany
      have hj' : sliceIdxOfOrig a.val j < (σ.slice a).length := by
        simpa [slice, List.length_eraseIdx, hax] using
          sliceIdx_lt_eraseIdx_length hjσ hjne hax
      have hany := any_cons_eq_orig_of_slice (σ := σ) a as hall (sliceIdxOfOrig a.val j) hj'
      have hkey : origIdxOfSlice a.val (sliceIdxOfOrig a.val j) = j := origIdx_sliceIdx_of_ne hjne
      rw [hkey] at hany
      have hjL : (a :: as).any (fun z => decide (z.val = j)) = false := by
        simpa [Bool.not_eq_true] using hjany
      have hjR : (mapFinsBelowHead a as hall).any (fun b => decide (b.val = sliceIdxOfOrig a.val j)) = false := by
        rw [← hany, hjL]
      refine Finset.mem_coe.mpr (Finset.mem_filter.mpr ⟨Finset.mem_range.2 ?_, ?_⟩)
      · simpa [List.length_eraseIdx, hax] using sliceIdx_lt_eraseIdx_length hjσ hjne hax
      · simp [hjR]
    · intro i hi
      rcases Finset.mem_filter.mp (Finset.mem_coe.mp hi) with ⟨hiσ, hiany⟩
      have hi' : i < (σ.slice a).length := by
        simpa [slice, List.length_eraseIdx, hax] using Finset.mem_range.mp hiσ
      have hany := any_cons_eq_orig_of_slice (σ := σ) a as hall i hi'
      have horig : origIdxOfSlice a.val i < σ.length := origIdxOfSlice_lt_length a i hi'
      have hiR : (mapFinsBelowHead a as hall).any (fun b => decide (b.val = i)) = false := by
        simpa [Bool.not_eq_true] using hiany
      have hiL : (a :: as).any (fun z => decide (z.val = origIdxOfSlice a.val i)) = false := by
        rw [hany]
        exact hiR
      refine Finset.mem_coe.mpr (Finset.mem_filter.mpr ⟨Finset.mem_range.2 horig, ?_⟩)
      simp [hiL]
    · intro j hj
      rcases Finset.mem_filter.mp (Finset.mem_coe.mp hj) with ⟨hjlt, hjany⟩
      have hjσ : j < σ.length := Finset.mem_range.mp hjlt
      have hjne : j ≠ a.val := by
        intro rid
        subst rid
        have ht : (a :: as).any (fun z => decide (z.val = a.val)) = true := by simp [List.any_cons]
        rw [ht] at hjany
        simp at hjany
      simp [origIdx_sliceIdx_of_ne hjne]
    · intro i _hi
      exact sliceIdx_origIdx a i
  have hlen :
      (removeAxes σ (a :: as)).length =
        (removeAxes (σ.slice a) (mapFinsBelowHead a as hall)).length := by
    rw [removeAxes_length_eq_card, removeAxes_length_eq_card, hcard]
  refine List.ext_getElem hlen ?_
  intro t htL htR
  -- Elementwise equality: same multiset of kept indices (`hcard`) and monotone `origIdxOfSlice`
  -- identify the `t`-th kept axis; conclude with `List.getElem_eraseIdx`.
  -- (Detailed `filterMap`/`range` indexing lemmas are left to a small local induction.)
  simpa [removeAxes, slice, List.getElem_eraseIdx, any_cons_eq_orig_of_slice, origIdx_sliceIdx_of_ne,
    sliceIdx_origIdx, mapFinsBelowHead_map_val, any_mapFinsBelowHead_val] using
    (removeAxes_cons_max_getElem σ a as hall t htL htR hlen)

/-- Multiply dimensions to get element count. -/
def volume (σ : Shape) : Nat :=
  σ.foldl (fun acc d => acc * d) 1

/-- A shape is well-formed if every dimension is positive. -/
def WellFormed (σ : Shape) : Prop :=
  ∀ i (hi : i < σ.rank), 0 < σ[i]'hi

/-- Cons is well-formed iff the head is positive and the tail is well-formed. -/
theorem WellFormed.cons_iff (n : Nat) (σ : Shape) :
    WellFormed (n :: σ) ↔ 0 < n ∧ WellFormed σ :=
  ⟨fun h => ⟨h 0 (by simp), fun i hi => h (i + 1) (by simp; omega)⟩,
   fun ⟨hn, ht⟩ i hi => by
     match i with
     | 0 => simpa
     | i + 1 => exact ht i (by simp at hi; omega)⟩

/-- Left-pad a shape with ones to a target rank. -/
def pad (σ : Shape) (len : Nat) : Shape :=
  let missing := len - σ.rank
  (List.replicate missing 1) ++ σ

/-- Apply a valid permutation to a shape. -/
def permute (σ : Shape) (perm : Permutation) (h : Permutation.Valid perm σ.rank) : Shape :=
  perm.pmap (fun i (_ : i < σ.rank) => σ[i]) h.forall_lt

/-! ## Index-stride arithmetic -/

/-- Compute strides for each axis of a shape. -/
def computeStrides : Shape → List Nat
  | [] => []
  | _ :: ds => volume ds :: computeStrides ds

/-- Convert a flat index to a multi-index given strides. -/
def unravelIndex (strides : List Nat) (idx : Nat) : List Nat :=
  match strides with
  | [] => []
  | s :: rest =>
    if s == 0 then 0 :: unravelIndex rest 0
    else (idx / s) :: unravelIndex rest (idx % s)

/-- Convert a multi-index to a flat index given strides. -/
def ravelIndex (strides : List Nat) (indices : List Nat) : Nat :=
  (strides.zip indices).foldl (fun acc (s, i) => acc + s * i) 0

/-! ## Shape volume lemmas -/

/-- `volume (d :: ds) = d * volume ds`. -/
theorem volume_cons (d : Nat) (ds : Shape) : volume (d :: ds) = d * volume ds := by
  simp only [volume, List.foldl_cons]
  rw [show 1 * d = d * 1 from by omega]
  exact List.foldl_assoc (op := (· * ·))

/-- `volume [] = 1`. -/
theorem volume_nil : volume [] = 1 := rfl

/-- `volume (List.replicate n 1) = 1`. -/
theorem volume_replicate_one (n : Nat) : volume (List.replicate n 1) = 1 := by
  induction n with
  | zero => rfl
  | succ n ih => simp [List.replicate_succ, volume_cons, ih]

/-! ## Shape permutation lemmas -/

open List in
/-- Product of dimensions is invariant under permutation. -/
private theorem volume_perm (a b : Shape) (h : a.Perm b) : volume a = volume b := by
  unfold volume
  exact h.foldl_eq' (by intro x _ y _ z; rw [Nat.mul_assoc, Nat.mul_comm x y, ← Nat.mul_assoc]) 1

/-- A valid permutation of length `n` is a permutation of `List.range n`. -/
private theorem perm_range_of_valid {n : Nat} (perm : Permutation)
    (h : Permutation.Valid perm n) :
    perm.Perm (List.range perm.length) :=
  h.length_eq ▸ h.perm

/-- The `pmap`-based `permute` equals `map` with `getD` for valid permutations. -/
theorem permute_eq_map (σ : Shape) (perm : Permutation) (h : Permutation.Valid perm σ.rank) :
    permute σ perm h = perm.map (fun i => σ.getD i 0) := by
  simp only [permute]
  apply List.ext_getElem
  · simp [List.length_pmap]
  · intro i h1 h2
    simp only [List.getElem_pmap, List.getElem_map]
    have hi_perm : i < perm.length := by rwa [List.length_pmap] at h1
    have hlt := h.forall_lt _ (List.getElem_mem hi_perm)
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt]

open List in
/-- Applying a valid permutation to a shape produces a `Perm` of that shape. -/
private theorem permute_perm (σ : Shape) (perm : Permutation)
    (h : Permutation.Valid perm σ.rank) :
    (permute σ perm h).Perm σ := by
  rw [permute_eq_map]
  have hperm_range := perm_range_of_valid perm h
  have hmap_perm := hperm_range.map (fun i => σ.getD i 0)
  rw [h.length_eq] at hmap_perm
  rw [range_map_getD] at hmap_perm
  exact hmap_perm

/-- A valid permutation preserves element count (`volume`). -/
theorem volume_permute (σ : Shape) (perm : Permutation)
    (h : Permutation.Valid perm σ.rank) :
    volume (permute σ perm h) = volume σ :=
  volume_perm _ _ (permute_perm σ perm h)

/-! ## Stride / index arithmetic lemmas -/

/-- Strides have the same length as the shape. -/
theorem computeStrides_length (σ : Shape) :
    (computeStrides σ).length = σ.length := by
  induction σ with
  | nil => rfl
  | cons _ ds ih => simp [computeStrides, ih]

/-- `unravelIndex` has the same length as strides. -/
theorem unravelIndex_length (strides : List Nat) (idx : Nat) :
    (unravelIndex strides idx).length = strides.length := by
  induction strides generalizing idx with
  | nil => rfl
  | cons s rest ih =>
    simp only [unravelIndex]
    split <;> simp [ih]

/-- Accumulator shift for `ravelIndex` fold. -/
private theorem ravelIndex_foldl_add (pairs : List (Nat × Nat)) (init : Nat) :
    pairs.foldl (fun acc (s, i) => acc + s * i) init =
    init + pairs.foldl (fun acc (s, i) => acc + s * i) 0 := by
  induction pairs generalizing init with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons]
    rw [ih (init + p.1 * p.2), ih (0 + p.1 * p.2)]
    omega

/-- `ravelIndex` cons decomposition. -/
theorem ravelIndex_cons (s : Nat) (ss : List Nat) (i : Nat) (is_ : List Nat) :
    ravelIndex (s :: ss) (i :: is_) = s * i + ravelIndex ss is_ := by
  simp only [ravelIndex, List.zip_cons_cons, List.foldl_cons]
  rw [ravelIndex_foldl_add]
  simp

/-- Each component of `unravelIndex` is bounded by the corresponding shape dimension. -/
theorem unravelIndex_bounded (σ : Shape) (k : Nat) (hk : k < volume σ)
    (i : Nat) (hi : i < σ.rank) :
    (unravelIndex (computeStrides σ) k)[i]'(by
      rw [unravelIndex_length, computeStrides_length]; exact hi)
    < σ[i]'hi := by
  induction σ generalizing k i with
  | nil => exact absurd hi (Nat.not_lt_zero i)
  | cons d ds ih =>
    rw [volume_cons] at hk
    have hvolume_pos : 0 < volume ds := by
      rcases Nat.eq_zero_or_pos (volume ds) with h | h
      · rw [h, Nat.mul_zero] at hk; omega
      · exact h
    have hvolume_ne : (volume ds == 0) = false := by
      simp only [beq_eq_false_iff_ne]; omega
    simp only [computeStrides, unravelIndex, hvolume_ne]
    match i, hi with
    | 0, _ =>
      simp only [List.getElem_cons_zero]
      exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hk)
    | i + 1, hi =>
      simp only [List.getElem_cons_succ]
      have hi' : i < ds.length := by simp [List.length_cons] at hi; omega
      exact ih (k % volume ds) (Nat.mod_lt k hvolume_pos) i hi'

/-- `ravelIndex` of bounded indices is less than `volume`. -/
theorem ravelIndex_lt_volume (σ : Shape) (indices : List Nat)
    (hlen : indices.length = σ.rank)
    (hbnd : ∀ i (hi : i < σ.rank), indices[i]'(hlen ▸ hi) < σ[i]'hi) :
    ravelIndex (computeStrides σ) indices < volume σ := by
  induction σ generalizing indices with
  | nil =>
    cases indices with
    | nil => simp [ravelIndex, volume]
    | cons => simp at hlen
  | cons d ds ih =>
    cases indices with
    | nil => simp at hlen
    | cons idx rest =>
      simp only [computeStrides]
      rw [ravelIndex_cons, volume_cons]
      have hidx : idx < d := by
        have := hbnd 0 (by simp)
        simp only [List.getElem_cons_zero] at this
        exact this
      have hrest_len : rest.length = ds.length := by simp at hlen; omega
      have hrest_bnd : ∀ i (hi : i < ds.length), rest[i]'(hrest_len ▸ hi) < ds[i]'hi := by
        intro i hi
        have := hbnd (i + 1) (by simp; omega)
        simp only [List.getElem_cons_succ] at this
        exact this
      have ih_val := ih rest hrest_len hrest_bnd
      have hsucc : volume ds * idx + volume ds = volume ds * (idx + 1) :=
        (Nat.mul_succ (volume ds) idx).symm
      have hmul_le : volume ds * (idx + 1) ≤ volume ds * d :=
        Nat.mul_le_mul_left (volume ds) (by omega)
      have hcomm : volume ds * d = d * volume ds := Nat.mul_comm (volume ds) d
      omega

/-- Round-trip: `unravelIndex ∘ ravelIndex = id` for valid indices. -/
theorem unravelIndex_ravelIndex (σ : Shape) (indices : List Nat)
    (hlen : indices.length = σ.rank)
    (hbnd : ∀ i (hi : i < σ.rank), indices[i]'(hlen ▸ hi) < σ[i]) :
    unravelIndex (computeStrides σ) (ravelIndex (computeStrides σ) indices) = indices := by
  induction σ generalizing indices with
  | nil =>
    cases indices with
    | nil => rfl
    | cons => simp at hlen
  | cons d ds ih =>
    cases indices with
    | nil => simp at hlen
    | cons idx rest =>
      have hidx : idx < d := by
        have := hbnd 0 (by simp); simp at this; exact this
      have hrest_len : rest.length = ds.length := by simp at hlen; omega
      have hrest_bnd : ∀ i (hi : i < ds.length), rest[i]'(hrest_len ▸ hi) < ds[i] := by
        intro i hi
        have := hbnd (i + 1) (by simp; omega); simp at this; exact this
      -- r = ravelIndex strides_ds rest < volume ds
      have hr_lt : ravelIndex (computeStrides ds) rest < volume ds :=
        ravelIndex_lt_volume ds rest hrest_len hrest_bnd
      have hvolume_pos : 0 < volume ds := Nat.pos_of_ne_zero (by omega)
      have hvolume_ne : (volume ds == 0) = false := by simp; omega
      simp only [computeStrides, ravelIndex_cons]
      show unravelIndex (volume ds :: computeStrides ds)
        (volume ds * idx + ravelIndex (computeStrides ds) rest) = idx :: rest
      unfold unravelIndex
      simp only [hvolume_ne, Bool.false_eq_true, ↓reduceIte]
      congr 1
      · -- Head: (volume_ds * idx + r) / volume_ds = idx
        rw [Nat.add_comm, Nat.add_mul_div_left _ _ hvolume_pos, Nat.div_eq_of_lt hr_lt]
        simp
      · -- Tail: (volume_ds * idx + r) % volume_ds = r, then IH
        rw [Nat.mul_add_mod, Nat.mod_eq_of_lt hr_lt]
        exact ih rest hrest_len hrest_bnd

/-- Helper: element of the inverse permutation at `i < n` is less than `n`. -/
private theorem getElem_invert_lt {perm : Permutation} {n : Nat}
    (h : Permutation.Valid perm n) (i : Nat) (hi : i < n) :
    (perm.invert h)[i]'(by rw [h.invert_length]; exact hi) < n :=
  h.invert_forall_lt _ (List.getElem_mem ..)

/-- The permuted shape at the inverse index equals the original shape at the original index. -/
theorem permute_getElem_invert (σ : Shape) (perm : Permutation)
    (h : Permutation.Valid perm σ.rank) (i : Nat) (hi : i < σ.rank) :
    (permute σ perm h)[(perm.invert h)[i]'(by rw [h.invert_length]; exact hi)]'(by
      simp only [permute, List.length_pmap, h.length_eq]
      exact getElem_invert_lt h i hi) =
    σ[i]'hi := by
  simp only [permute, List.getElem_pmap]
  congr 1
  exact h.invert_spec i hi

/-- The reindexed flat index is in bounds (used by `transposeByFlattening`). -/
theorem ravelIndex_permuted_lt (σ : Shape) (perm : Permutation)
    (h : Permutation.Valid perm σ.rank) (k : Nat)
    (hk : k < volume (permute σ perm h)) :
    let outShape := permute σ perm h
    let inv := perm.invert h
    let outMulti := unravelIndex (computeStrides outShape) k
    let inMulti := inv.map (fun p => outMulti.getD p 0)
    ravelIndex (computeStrides σ) inMulti < volume σ := by
  simp only
  -- 1. inMulti.length = σ.rank
  have hinvLen := h.invert_length
  have hInMultiLen : ((perm.invert h).map (fun p =>
      (unravelIndex (computeStrides (permute σ perm h)) k).getD p 0)).length
      = σ.rank := by
    rw [List.length_map, hinvLen]
  -- 2. Each inMulti[i] < σ[i]
  have hInMultiBnd : ∀ i (hi : i < σ.rank),
      ((perm.invert h).map (fun p =>
        (unravelIndex (computeStrides (permute σ perm h)) k).getD p 0))[i]'(by
          rw [hInMultiLen]; exact hi)
      < σ[i]'hi := by
    intro i hi
    simp only [List.getElem_map]
    -- After simp, goal uses inv[i] (getElem notation)
    have hi_inv : i < (perm.invert h).length := by rw [hinvLen]; exact hi
    have hv_elem : (perm.invert h)[i]'hi_inv < σ.rank :=
      h.invert_forall_lt _ (List.getElem_mem hi_inv)
    have hv_out : (perm.invert h)[i]'hi_inv < (permute σ perm h).length := by
      simp only [permute, List.length_pmap, h.length_eq]; exact hv_elem
    have hv_multi : (perm.invert h)[i]'hi_inv <
        (unravelIndex (computeStrides (permute σ perm h)) k).length := by
      rw [unravelIndex_length, computeStrides_length]; exact hv_out
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hv_multi, Option.getD_some]
    -- Goal: outMulti[inv[i]] < σ[i]
    -- From unravelIndex_bounded: outMulti[inv[i]] < outShape[inv[i]]
    have hbnd := unravelIndex_bounded (permute σ perm h) k hk _ hv_out
    -- Prove outShape[inv[i]] = σ[i]
    suffices heq : (permute σ perm h)[(perm.invert h)[i]'hi_inv]'hv_out
        = σ[i]'hi by rw [heq] at hbnd; exact hbnd
    simp only [permute, List.getElem_pmap]
    congr 1
    exact h.invert_spec i hi
  -- 3. ravelIndex < volume
  exact ravelIndex_lt_volume σ _ hInMultiLen hInMultiBnd

/-! ## Shape broadcast lemmas -/

theorem pad_rank (σ : Shape) (len : Nat) :
    (pad σ len).rank = max σ.rank len := by
  simp [rank, pad, List.length_append, List.length_replicate]
  omega

theorem pad_of_rank_le (σ : Shape) (len : Nat) (h : σ.rank ≤ len) :
    (pad σ len).rank = len := by
  simp [pad_rank]; omega

theorem pad_eq_of_rank_eq (σ : Shape) (len : Nat) (h : σ.rank = len) :
    pad σ len = σ := by
  simp [pad, h, Nat.sub_self]

end Shape

/-- A well-formed shape: a `Shape` together with proof that all dimensions are positive. -/
structure WellFormedShape where
  /-- The underlying dimension list. -/
  shape : Shape
  /-- Every dimension is positive. -/
  wf : Shape.WellFormed shape


end Palladium.TensorLib
