import TensorLib.TensorLemmas
import TensorLib.IndexMaps

/-! # Exact indexing lemmas

For every lawful layout, indexing the result of a tensor operation reads its operands at
the indices given by the index maps of `TensorLib.IndexMaps`:

* `index_broadcast_proj`: `(t ▷ σₒ)[i] = t[proj(i)]`;
* `index_reshape`: `(reshape t σₒ)[i] = t[unflatten(flatten(i))]`;
* `index_squeeze`: `(squeeze t n)[i] = t[ins_n(i, 0)]`;
* `index_sumAxes`: indexing `sumAxes` sums over the removed axes (`IdxFn.sumAxes`);
* `index_ofFn`: the tensor `Tensor.ofFn f` built from an indexing function `f` reads `f`.
-/

namespace Palladium.TensorLib

open Shape

/-! ### Shape facts -/

/-- A shape of rank at most `rank τ` that broadcasts to `d :: τ` broadcasts to `τ`. -/
theorem Broadcastable.of_cons_target {σ τ : Shape} {d : Nat}
    (h : Broadcastable σ (d :: τ)) (hr : σ.rank ≤ τ.rank) : Broadcastable σ τ := by
  unfold Broadcastable at *
  have hpad : pad σ (Shape.rank (d :: τ)) = 1 :: pad σ τ.rank := by
    simp only [pad, Shape.rank, List.length_cons] at hr ⊢
    rw [show τ.length + 1 - σ.length = (τ.length - σ.length) + 1 by omega]
    rfl
  rw [hpad] at h
  exact h.tail

/-- Equal-rank broadcastability of cons shapes splits into head and tail facts. -/
theorem Broadcastable.cons_cons {σ τ : Shape} {d' d : Nat}
    (h : Broadcastable (d' :: σ) (d :: τ)) (hr : σ.rank = τ.rank) :
    (d' = d ∨ d' = 1) ∧ Broadcastable σ τ := by
  unfold Broadcastable at *
  rw [pad_eq_of_rank_eq _ _ (by simp [Shape.rank] at hr ⊢; omega)] at h
  rw [pad_eq_of_rank_eq _ _ hr]
  cases h with
  | eq h => exact ⟨.inl rfl, h⟩
  | one h => exact ⟨.inr rfl, h⟩

section

variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]

/-! ### Broadcast -/

/-- Indexing a broadcast reads the operand at the projected index. -/
theorem Tensor.index_broadcast_proj {α : Type} :
    (σₒ : Shape) → {σ : Shape} → (t : L σ α) → (h : Broadcastable σ σₒ) → (i : Indices σₒ) →
    Tensor.Operations.index σₒ (Tensor.Operations.broadcast t σₒ h) i =
      Tensor.Operations.index σ t (i.proj h)
  | [], σ, t, h, i => by
      cases σ with
      | cons d ds => exact absurd (Broadcastable.rank_le h) (by simp [Shape.rank])
      | nil =>
        rw [Tensor.LawfulOperations.broadcast_broadcastable_refl]
        exact congrArg _ (Indices.ext (by
          simp [List.eq_nil_of_length_eq_zero i.length_eq]))
  | d :: τ, σ, t, h, i => by
      by_cases hr : Shape.rank σ ≤ Shape.rank τ
      · have h' := Broadcastable.of_cons_target h hr
        rw [Tensor.Operations.broadcast_index_cons_left' d t h h' i,
          Tensor.index_broadcast_proj τ t h' i.tail]
        congr 1
        apply Indices.ext
        apply List.ext_getElem (by simp)
        intro k hk1 hk2
        simp only [Indices.proj_values, List.getElem_ofFn]
        split
        · rfl
        · simp only [Indices.tail, List.getElem_tail]
          congr 1
          simp only [Shape.rank, List.length_cons] at hr ⊢
          omega
      · cases σ with
        | nil => exact absurd (Nat.zero_le _) hr
        | cons d' σ' =>
          have hrank : Shape.rank σ' = Shape.rank τ := by
            have := Broadcastable.rank_le h
            simp only [Shape.rank, List.length_cons] at this hr ⊢; omega
          obtain ⟨hd, h'⟩ := Broadcastable.cons_cons h hrank
          have hproj_tail : (i.proj h).tail = i.tail.proj h' := by
            apply Indices.ext
            apply List.ext_getElem (by simp [Indices.tail])
            intro k hk1 hk2
            simp only [Indices.tail, List.getElem_tail, Indices.proj_values, List.getElem_ofFn]
            simp only [Shape.rank, List.length_cons, List.getElem_cons_succ] at hrank ⊢
            split
            · rfl
            · congr 1; omega
          show Tensor.Operations.index τ
              (Tensor.Operations.get (Tensor.Operations.broadcast t (d :: τ) h) i.head) i.tail =
            Tensor.Operations.index σ'
              (Tensor.Operations.get t (i.proj h).head) (i.proj h).tail
          rw [hproj_tail]
          by_cases h1 : d' = 1
          · subst h1
            rw [Tensor.LawfulOperations.get_broadcast_one t h h' hrank i.head,
              Tensor.index_broadcast_proj τ _ h' i.tail]
            congr 2
            apply Fin.ext
            simp [Indices.head, Indices.proj]
          · have hdd : d' = d := hd.resolve_right h1
            subst hdd
            rw [Tensor.LawfulOperations.get_broadcast_eq t h h' hrank i.head,
              Tensor.index_broadcast_proj τ _ h' i.tail]
            congr 2
            apply Fin.ext
            simp [Indices.head, Indices.proj, h1, List.head_eq_getElem]
            try (congr 1; simp only [Shape.rank] at hrank; omega)

/-- `Tensor.broadcast` form of `index_broadcast_proj`. -/
theorem Tensor.index_broadcast {α : Type} {σ σₒ : Shape} (t : Tensor (L := L) σ α)
    (h : Broadcastable σ σₒ) (i : Indices σₒ) :
    Tensor.index (Tensor.broadcast t σₒ h) i = Tensor.index t (i.proj h) :=
  Tensor.index_broadcast_proj σₒ t h i

/-! ### Flatten, unflatten, reshape -/

/-- Indexing an unflattened list reads the list at the flat index. -/
theorem Tensor.index_unflatten {α : Type} (σ : Shape) (xs : List α)
    (h : xs.length = σ.volume) (i : Indices σ) :
    Tensor.Operations.index σ (Tensor.Operations.unflatten (L := L) σ xs h) i =
      xs[i.flatten.val]'(by rw [h]; exact i.flatten.isLt) := by
  rw [Tensor.LawfulOperations.index_eq_getElem_flatten]
  exact List.getElem_of_eq (Tensor.LawfulOperations.flatten_unflatten σ xs h) _

/-- Reading the flattening of a tensor at a flat index reads the tensor at the unflattened
    index. -/
theorem Tensor.getElem_flatten {α : Type} {σ : Shape} (t : L σ α) (x : Fin σ.volume) :
    (Tensor.Operations.flatten t)[x.val]'(by rw [Tensor.Operations.flatten_length]; exact x.isLt) =
      Tensor.Operations.index σ t (Indices.unflatten x) := by
  rw [Tensor.LawfulOperations.index_eq_getElem_flatten]
  congr 1
  exact (congrArg Fin.val (Indices.flatten_unflatten x)).symm

/-- Indexing a reshape reads the operand at `unflatten_σ(flatten_σₒ(i))`. -/
theorem Tensor.index_reshape {α : Type} {σ : Shape} (t : L σ α) (σₒ : Shape)
    (h : Reshapeable σ σₒ) (i : Indices σₒ) :
    Tensor.Operations.index σₒ (Tensor.reshape t σₒ h) i =
      Tensor.Operations.index σ t (Indices.unflatten (i.flatten.cast h.symm)) := by
  unfold Tensor.reshape
  rw [Tensor.index_unflatten]
  exact Tensor.getElem_flatten t (i.flatten.cast h.symm)

/-- The tensor whose entry at every index `i` is `f i`. -/
def Tensor.ofFn {α : Type} {σ : Shape} (f : Indices σ → α) : L σ α :=
  Tensor.Operations.unflatten σ (List.ofFn (fun x : Fin σ.volume => f (Indices.unflatten x)))
    (List.length_ofFn)

/-- Indexing `Tensor.ofFn f` reads `f`. -/
theorem Tensor.index_ofFn {α : Type} {σ : Shape} (f : Indices σ → α) (i : Indices σ) :
    Tensor.Operations.index σ (Tensor.ofFn (L := L) f) i = f i := by
  unfold Tensor.ofFn
  rw [Tensor.index_unflatten, List.getElem_ofFn, Indices.unflatten_flatten]

/-! ### Squeeze -/

/-- Indexing a squeeze reads the operand with index `0` inserted at the squeezed axis. -/
theorem Tensor.index_squeeze {α : Type} :
    (σ : Shape) → (t : L σ α) → (axis : Fin σ.rank) → (h : Squeezable σ axis) →
    (i : Indices (σ.slice axis)) →
    Tensor.Operations.index (σ.slice axis) (Tensor.Operations.squeeze t axis h) i =
      Tensor.Operations.index σ t (i.ins axis 0 (by unfold Squeezable at h; omega))
  | [], _, axis, _, _ => Fin.elim0 axis
  | d :: σ, t, ⟨0, h0⟩, h, i => by
      have hd : d = 1 := h
      subst hd
      rw [Tensor.LawfulOperations.squeeze_zero]
      show _ = Tensor.Operations.index σ (Tensor.Operations.get t _) _
      congr 1
      all_goals first
        | exact congrArg _ (Fin.ext (by simp [Indices.ins, Indices.insertIdx, Indices.head]))
        | (apply Indices.ext; simp [Indices.ins, Indices.insertIdx, Indices.tail])
  | d :: σ, t, ⟨a + 1, ha⟩, h, i => by
      have ha' : a < Shape.rank σ := by simp [Shape.rank] at ha ⊢; omega
      have hone : σ[a]'ha' = 1 := h
      show Tensor.Operations.index (Shape.slice σ ⟨a, ha'⟩)
          (Tensor.Operations.get (Tensor.Operations.squeeze t ⟨a + 1, ha⟩ h) i.head) i.tail = _
      rw [Tensor.LawfulOperations.get_squeeze t ⟨a, ha'⟩ hone i.head,
        Tensor.index_squeeze σ _ ⟨a, ha'⟩ hone i.tail]
      show _ = Tensor.Operations.index σ (Tensor.Operations.get t _) _
      congr 1
      · exact congrArg _ (Fin.ext (by
          obtain ⟨vs, hl, hv⟩ := i
          cases vs with
          | nil => simp [Shape.slice, Shape.rank] at hl
          | cons v vs => simp [Indices.ins, Indices.insertIdx, Indices.head]))
      · apply Indices.ext
        obtain ⟨vs, hl, hv⟩ := i
        cases vs with
        | nil => simp [Shape.slice, Shape.rank] at hl
        | cons v vs => simp [Indices.ins, Indices.insertIdx, Indices.tail]

/-! ### Sums -/

section

variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]

/-- Indexing `sumSortedAxesAux` sums the indexing function along the sorted axes. -/
theorem Tensor.index_sumSortedAxesAux {α : Type} (add : α → α → α) (zero : α) {σ : Shape}
    (t : L σ α) (sorted : List (Fin σ.rank))
    (hp : sorted.Pairwise (fun x y => x.val > y.val)) (i : Indices (removeAxes σ sorted)) :
    Tensor.index (Tensor.sumSortedAxesAux add zero t sorted hp) i =
      IdxFn.sumSorted add zero (Tensor.index t) sorted hp i := by
  match sorted with
  | [] =>
    rw [Tensor.sumSortedAxesAux, IdxFn.sumSorted, Tensor.index_tensorCastShape]
  | a :: as =>
    rw [Tensor.sumSortedAxesAux, IdxFn.sumSorted, Tensor.index_tensorCastShape,
      Tensor.index_sumSortedAxesAux]
    congr 1
    funext j
    exact Tensor.Operations.index_sum t add zero a j
termination_by sorted.length
decreasing_by
  simp_wf
  simp [mapFinsBelowHead, List.length_map, List.length_attach]

/-- Indexing `sumAxes` sums the indexing function along the removed axes. -/
theorem Tensor.index_sumAxes {α : Type} (add : α → α → α) (zero : α) {σ : Shape}
    (t : L σ α) (axes : List (Fin σ.rank)) (h : axes.Nodup) (i : Indices (removeAxes σ axes)) :
    Tensor.index (Tensor.sumAxes add zero t axes h) i =
      IdxFn.sumAxes add zero (Tensor.index t) axes h i := by
  rw [Tensor.sumAxes, Tensor.index_tensorCastShape, Tensor.index_sumSortedAxesAux]
  rfl

end

end

end Palladium.TensorLib
