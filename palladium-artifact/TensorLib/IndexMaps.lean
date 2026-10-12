import TensorLib.Indices
import TensorLib.Shape

/-! # Index maps

The index mappings used by the index-wise semantics of Jay (paper, §"Index Space and
Index Mappings"):

* `Indices.proj h : Indices σₒ → Indices σ` (*proj*) adapts an index of a broadcast
  result to the broadcast operand: it drops prepended broadcast dimensions and clamps
  singleton dimensions to `0`.
* `Indices.flatten : Indices σ → Fin σ.volume` (*flatten*) and
  `Indices.unflatten : Fin σ.volume → Indices σ` (*unflatten*) convert between
  multi-indices and row-major flat indices.
* `Indices.insertIdx` (*ins*) inserts an index at an axis; see `TensorLib.Indices`.
-/

namespace Palladium.TensorLib

open Shape

/-- Dimension-wise reading of aligned broadcastability. -/
theorem BroadcastableAligned.getElem_eq_or {σ τ : Shape} (h : BroadcastableAligned σ τ) :
    ∀ j (hj : j < σ.length) (hj' : j < τ.length), σ[j] = 1 ∨ σ[j] = τ[j] := by
  induction h with
  | nil => intro j hj; exact absurd hj (by simp)
  | eq _ ih =>
    intro j hj hj'
    cases j with
    | zero => exact .inr rfl
    | succ j => exact ih j (by simpa using hj) (by simpa using hj')
  | one _ ih =>
    intro j hj hj'
    cases j with
    | zero => exact .inl rfl
    | succ j => exact ih j (by simpa using hj) (by simpa using hj')

/-- Dimension-wise reading of broadcastability: each dimension of `σ` is `1` or equals the
    right-aligned dimension of `σₒ`. -/
theorem Broadcastable.getElem_eq_or {σ σₒ : Shape} (h : Broadcastable σ σₒ) (j : Nat) (hj : j < σ.rank) :
    σ[j] = 1 ∨
      σ[j] = σₒ[j + (σₒ.rank - σ.rank)]'(by have := Broadcastable.rank_le h; simp only [Shape.rank] at *; omega) := by
  have hle := Broadcastable.rank_le h
  have hpad : j + (σₒ.rank - σ.rank) < (pad σ σₒ.rank).length := by
    simp only [pad, List.length_append, List.length_replicate, Shape.rank] at hle hj ⊢; omega
  have hpadj : (pad σ σₒ.rank)[j + (σₒ.rank - σ.rank)]'hpad = σ[j] := by
    show (List.replicate (σₒ.rank - σ.rank) 1 ++ σ)[j + (σₒ.rank - σ.rank)] = σ[j]
    rw [List.getElem_append_right (by simp)]
    simp
  have := BroadcastableAligned.getElem_eq_or h (j + (σₒ.rank - σ.rank)) hpad
    (by simp only [Shape.rank] at hle hj ⊢; omega)
  rwa [hpadj] at this

/-- `ravelIndex ∘ unravelIndex = id` on flat indices below the volume. -/
theorem Shape.ravelIndex_unravelIndex (σ : Shape) (k : Nat) (hk : k < volume σ) :
    ravelIndex (computeStrides σ) (unravelIndex (computeStrides σ) k) = k := by
  induction σ generalizing k with
  | nil => simp [volume] at hk; subst hk; rfl
  | cons d ds ih =>
    rw [volume_cons] at hk
    have hpos : 0 < volume ds := by
      rcases Nat.eq_zero_or_pos (volume ds) with h | h
      · rw [h, Nat.mul_zero] at hk; omega
      · exact h
    have hne : (volume ds == 0) = false := by simp; omega
    show ravelIndex (volume ds :: computeStrides ds)
      (unravelIndex (volume ds :: computeStrides ds) k) = k
    unfold unravelIndex
    simp only [hne, Bool.false_eq_true, ↓reduceIte, ravelIndex_cons]
    rw [ih (k % volume ds) (Nat.mod_lt k hpos)]
    exact Nat.div_add_mod k (volume ds)

namespace Indices

/-- `proj^σ_{σₒ}(i)_j = 0` if `σ[j] = 1`, and `i_{j + rank σₒ − rank σ}` otherwise. -/
def proj {σ σₒ : Shape} (h : Broadcastable σ σₒ) (i : Indices σₒ) : Indices σ :=
  ⟨List.ofFn (fun j : Fin σ.rank =>
      if σ[j.val]'j.isLt = 1 then 0
      else i.values[j.val + (σₒ.rank - σ.rank)]'(by
        have := Broadcastable.rank_le h; have := j.isLt; rw [i.length_eq]; omega)),
   by simp,
   fun j hj => by
     simp only [List.getElem_ofFn]
     split
     · next h1 => omega
     · next h1 =>
       rcases Broadcastable.getElem_eq_or h j hj with h2 | h2
       · exact absurd h2 h1
       · rw [h2]
         exact i.valid _ _⟩

@[simp] theorem proj_values {σ σₒ : Shape} (h : Broadcastable σ σₒ) (i : Indices σₒ) :
    (i.proj h).values = List.ofFn (fun j : Fin σ.rank =>
      if σ[j.val]'j.isLt = 1 then 0
      else i.values[j.val + (σₒ.rank - σ.rank)]'(by
        have := Broadcastable.rank_le h; have := j.isLt; rw [i.length_eq]; omega)) := rfl

/-- `flatten_σ(i) = Σ_j i_j · Π_{m > j} σ_m`, the row-major flat index. -/
def flatten {σ : Shape} (i : Indices σ) : Fin σ.volume :=
  ⟨ravelIndex (computeStrides σ) i.values,
   ravelIndex_lt_volume σ i.values i.length_eq i.valid⟩

/-- `unflatten_σ(x)_j = ⌊x / Π_{m > j} σ_m⌋ mod σ_j`, the inverse of `flatten`. -/
def unflatten {σ : Shape} (x : Fin σ.volume) : Indices σ :=
  ⟨unravelIndex (computeStrides σ) x.val,
   by rw [unravelIndex_length, computeStrides_length],
   fun j hj => unravelIndex_bounded σ x.val x.isLt j hj⟩

/-- `unflatten ∘ flatten = id`. -/
@[simp] theorem unflatten_flatten {σ : Shape} (i : Indices σ) : unflatten (flatten i) = i :=
  Indices.ext (unravelIndex_ravelIndex σ i.values i.length_eq i.valid)

/-- `flatten ∘ unflatten = id`. -/
@[simp] theorem flatten_unflatten {σ : Shape} (x : Fin σ.volume) : flatten (unflatten x) = x :=
  Fin.ext (Shape.ravelIndex_unravelIndex σ x.val x.isLt)

/-- `ins_n(i, v)`: insert index `v` at axis `n`. -/
abbrev ins {σ : Shape} (axis : Fin σ.rank) (i : Indices (σ.slice axis)) (v : Nat)
    (hv : v < σ[axis.val]'axis.isLt) : Indices σ :=
  i.insertIdx axis v hv

end Indices

/-! ## Sums of indexing functions -/

namespace IdxFn

/-- `λi. Σ_{j < σ[axis]} f(ins_axis(i, j))`: the sum of an indexing function along `axis`,
    folding with `add` from `zero`. -/
def sum {α : Type} (add : α → α → α) (zero : α) {σ : Shape} (f : Indices σ → α)
    (axis : Fin σ.rank) : Indices (σ.slice axis) → α :=
  fun i => (List.ofFn (fun v : Fin (σ[axis.val]'axis.isLt) =>
    f (i.ins axis v.val v.isLt))).foldr add zero

/-- Sum along axes given in strictly decreasing order, largest axis first. -/
def sumSorted {α : Type} (add : α → α → α) (zero : α) {σ : Shape} (f : Indices σ → α)
    (sorted : List (Fin σ.rank)) (hp : sorted.Pairwise (fun x y => x.val > y.val)) :
    Indices (removeAxes σ sorted) → α :=
  match sorted with
  | [] => fun i => f (i.cast (removeAxes_nil σ))
  | a :: as =>
      have hall : ∀ x ∈ as, x.val < a.val := (List.pairwise_cons.mp hp).1
      fun i => sumSorted add zero (sum add zero f a) (mapFinsBelowHead a as hall)
        (pairwise_mapFinsBelowHead as hp hall) (i.cast (removeAxes_cons_max a as hall))
termination_by sorted.length
decreasing_by
  simp_wf
  simp [mapFinsBelowHead, List.length_map, List.length_attach]

/-- Sum along every axis in `axes`. -/
def sumAxes {α : Type} (add : α → α → α) (zero : α) {σ : Shape} (f : Indices σ → α)
    (axes : List (Fin σ.rank)) (h : axes.Nodup) : Indices (removeAxes σ axes) → α :=
  fun i => sumSorted add zero f (sortAxesDesc axes) (pairwise_sortAxesDesc axes h)
    (i.cast (removeAxes_perm_sortAxesDesc axes))

end IdxFn

/-! ## Lemmas on index maps -/

namespace Indices

/-- An entry of a dimension of size `1` is `0`. -/
theorem getElem_eq_zero_of_dim_one {σ : Shape} (k : Indices σ) (j : Nat) (hj : j < σ.rank)
    (h1 : σ[j] = 1) : k.values[j]'(by rw [k.length_eq]; exact hj) = 0 := by
  have := k.valid j hj
  omega

/-- The entries of `proj^σ_{σₒ}(i)`. -/
theorem proj_getElem {σ σₒ : Shape} (h : Broadcastable σ σₒ) (i : Indices σₒ) (j : Nat)
    (hj : j < (i.proj h).values.length) :
    (i.proj h).values[j] =
      if σ[j]'(by rw [(i.proj h).length_eq] at hj; exact hj) = 1 then 0
      else i.values[j + (σₒ.rank - σ.rank)]'(by
        have := Broadcastable.rank_le h; rw [(i.proj h).length_eq] at hj; rw [i.length_eq]
        omega) := by
  simp [proj_values, List.getElem_ofFn]

/-- `proj^σ_{σₒ}(i) = k` when `k` agrees with `i` at the right-aligned entries of the
    dimensions of `σ` that are not `1`. -/
theorem proj_eq {σ σₒ : Shape} {h : Broadcastable σ σₒ} {i : Indices σₒ} {k : Indices σ}
    (H : ∀ j (hj : j < σ.rank), σ[j] ≠ 1 →
      i.values[j + (σₒ.rank - σ.rank)]'(by
        have := Broadcastable.rank_le h; rw [i.length_eq]; omega) =
        k.values[j]'(by rw [k.length_eq]; exact hj)) :
    i.proj h = k := by
  apply Indices.ext
  apply List.ext_getElem (by rw [(i.proj h).length_eq, k.length_eq])
  intro j hj₁ hj₂
  have hj : j < σ.rank := by rw [(i.proj h).length_eq] at hj₁; exact hj₁
  rw [proj_getElem]
  split
  · next h1 => exact (getElem_eq_zero_of_dim_one k j hj h1).symm
  · next h1 => exact H j hj h1

/-- `proj^σ_σ = id`. -/
@[simp] theorem proj_refl {σ : Shape} (h : Broadcastable σ σ) (i : Indices σ) :
    i.proj h = i :=
  proj_eq fun j _ _ => by simp

/-- `proj` along a shape equality is a cast. -/
theorem proj_of_eq {σ σₒ : Shape} (h : Broadcastable σ σₒ) (hσ : σ = σₒ) (i : Indices σₒ) :
    i.proj h = i.cast hσ.symm := by
  subst hσ; exact proj_refl h i

/-- `proj` from a cast index is `proj` from the original index. -/
theorem proj_cast {σ σ₁ σ₂ : Shape} (i : Indices σ₁) (h : σ₁ = σ₂) (hb : Broadcastable σ σ₂)
    (hb' : Broadcastable σ σ₁) : (i.cast h).proj hb = i.proj hb' := by
  subst h; rfl

/-- `proj^{σ₁}_{σ₂} ∘ proj^{σ₂}_{σ₃} = proj^{σ₁}_{σ₃}`. -/
theorem proj_proj {σ₁ σ₂ σ₃ : Shape} (h₁ : Broadcastable σ₁ σ₂) (h₂ : Broadcastable σ₂ σ₃)
    (h : Broadcastable σ₁ σ₃) (i : Indices σ₃) :
    (i.proj h₂).proj h₁ = i.proj h := by
  apply proj_eq
  intro j hj hne
  have hr₁ := Broadcastable.rank_le h₁
  have hr₂ := Broadcastable.rank_le h₂
  rw [proj_getElem, proj_getElem]
  rcases Broadcastable.getElem_eq_or h₁ j hj with h1 | h1
  · exact absurd h1 hne
  · rw [if_neg hne, if_neg (h1 ▸ hne)]
    congr 1
    omega

/-- Every index of the scalar shape is `nil`. -/
theorem eq_nil (i : Indices []) : i = nil :=
  Indices.ext (List.eq_nil_of_length_eq_zero i.length_eq)

/-- `proj` from `d :: σₒ` drops the leading index when `σ` broadcasts to `σₒ`. -/
theorem proj_cons {σ σₒ : Shape} {d : Nat} (h : Broadcastable σ (d :: σₒ))
    (h' : Broadcastable σ σₒ) (i : Indices (d :: σₒ)) :
    i.proj h = i.tail.proj h' := by
  apply proj_eq
  intro j hj hne
  have hr := Broadcastable.rank_le h'
  rw [proj_getElem, if_neg hne]
  simp only [tail, List.getElem_tail]
  congr 1
  simp only [Shape.rank, List.length_cons] at hr ⊢
  omega

/-- `proj` from `pre ++ σ` to `σ` keeps the trailing indices. -/
theorem proj_append {pre σ : Shape} (h : Broadcastable σ (pre ++ σ)) (i : Indices pre)
    (k : Indices σ) : (i.append k).proj h = k := by
  apply proj_eq
  intro j hj _
  simp only [append_values]
  rw [List.getElem_append_right (by rw [i.length_eq]; simp [Shape.rank])]
  congr 1
  simp [Shape.rank, i.length_eq]

end Indices

namespace IdxFn

/-- Summing along a single axis. -/
theorem sumAxes_singleton {α : Type} (add : α → α → α) (zero : α) {σ : Shape}
    (f : Indices σ → α) (a : Fin σ.rank) (h : [a].Nodup) (i : Indices (removeAxes σ [a])) :
    sumAxes add zero f [a] h i = sum add zero f a (i.cast (removeAxes_singleton σ a)) := by
  rw [sumAxes, sumSorted.eq_def]
  dsimp only [sortAxesDesc, List.insertionSort, List.foldr, List.orderedInsert]
  rw [sumSorted.eq_def]
  rfl

end IdxFn

/-- `ravelIndex` of zeros is `0`. -/
theorem Shape.ravelIndex_replicate_zero (ss : List Nat) (k : Nat) :
    ravelIndex ss (List.replicate k 0) = 0 := by
  induction k generalizing ss with
  | zero => cases ss <;> rfl
  | succ k ih =>
    cases ss with
    | nil => rfl
    | cons s ss => rw [List.replicate_succ, ravelIndex_cons, ih]; simp

/-- The flat index of `v :: 0 … 0` in `d :: 1 … 1` is `v`. -/
theorem Indices.flatten_enum {d k : Nat} (i : Indices (d :: List.replicate k 1)) :
    i.flatten.val = (i.values.head?.getD 0) := by
  obtain ⟨vs, hlen, hvalid⟩ := i
  match vs, hlen with
  | v :: rest, hlen =>
    have hrest : rest = List.replicate k 0 := by
      apply List.ext_getElem (by simp at hlen; simp [hlen])
      intro j h₁ h₂
      have := hvalid (j + 1) (by simp [Shape.rank] at hlen ⊢; omega)
      simp at this ⊢
      omega
    subst hrest
    simp [Indices.flatten, computeStrides, volume_replicate_one, ravelIndex_cons,
      Shape.ravelIndex_replicate_zero]

end Palladium.TensorLib
