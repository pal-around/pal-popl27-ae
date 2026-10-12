import TensorLib.Shape

namespace Palladium.TensorLib

/-!
## Proof-carrying tensor indices

`Indices shape` bundles a list of natural numbers with proofs that the list
has the right length and every entry is within the corresponding dimension.
This enables total indexing via `index` and bracket notation `tensor[indices]`.
-/

/-- Proof-carrying index list for a shape: length matches and every entry is in bounds. -/
structure Indices (σ : Shape) where
  /-- The raw index values. -/
  values : List Nat
  /-- The index list has the same length as the shape. -/
  length_eq : values.length = σ.rank
  /-- Every index is within the corresponding dimension. -/
  valid : ∀ i (hi : i < σ.rank), values[i]'(length_eq ▸ hi) < σ[i]'hi

namespace Indices

/-- The empty index list for the scalar shape `[]`. -/
def nil : Indices [] :=
  ⟨[], rfl, fun _ hi => absurd hi (by simp)⟩

/-- Prepend an index for the outermost dimension. -/
def cons (i : Nat) (hi : i < d) (rest : Indices ds) : Indices (d :: ds) :=
  ⟨i :: rest.values,
   by simp [rest.length_eq],
   fun j hj => by
     cases j with
     | zero => simpa using hi
     | succ k =>
       simp only [List.getElem_cons_succ, List.length_cons] at hj ⊢
       exact rest.valid k (by simp only [Shape.rank]; omega)⟩

/-- Extract the head index as a `Fin d`. -/
def head (indices : Indices (d :: ds)) : Fin d :=
  have hne : indices.values ≠ [] := by
    have hlen := indices.length_eq; intro h; simp [h] at hlen
  ⟨indices.values.head hne, by
    have h := indices.valid 0 (by simp)
    simp only [List.getElem_cons_zero] at h
    rwa [List.head_eq_getElem]⟩

/-- Drop the head index, keeping the tail. -/
def tail (indices : Indices (d :: ds)) : Indices ds := by
  have hlen : indices.values.tail.length = ds.length := by
    have := indices.length_eq; simp at this ⊢; omega
  refine ⟨indices.values.tail, hlen, fun i hi => ?_⟩
  have h := indices.valid (i + 1) (by simp; omega)
  simp only [List.getElem_cons_succ] at h
  rw [List.getElem_tail]
  exact h

/-- Transport indices across a shape equality. -/
def cast (indices : Indices σ₁) (h : σ₁ = σ₂) : Indices σ₂ :=
  ⟨indices.values, h ▸ indices.length_eq, h ▸ indices.valid⟩

/-- Insert an index at position `axis`, extending an erased-shape index list
    to a full-shape index list. Needed for `index_sum`. -/
def insertIdx {σ : Shape} (axis : Fin σ.rank)
    (indices : Indices (σ.slice axis))
    (val : Nat) (hval : val < σ[axis.val]'axis.isLt) :
    Indices σ := by
  have hEraseLen : indices.values.length = σ.rank - 1 := by
    rw [indices.length_eq]; simp [List.length_eraseIdx, axis.isLt]
  have hle : axis.val ≤ indices.values.length := by omega
  have hInsertLen : (indices.values.insertIdx axis.val val).length = σ.rank := by
    rw [List.length_insertIdx, if_pos hle]; omega
  refine ⟨indices.values.insertIdx axis.val val, hInsertLen, fun i hi => ?_⟩
  simp only [List.getElem_insertIdx]
  split
  · -- i < axis.val
    rename_i h_lt
    have hi' : i < indices.values.length := by omega
    have hv := indices.valid i (Nat.lt_of_lt_of_eq hi' indices.length_eq)
    simp only [List.getElem_eraseIdx, dif_pos h_lt] at hv
    exact hv
  · split
    · -- i = axis.val
      rename_i _ h_eq; subst h_eq; exact hval
    · -- i > axis.val
      rename_i h_nlt h_ne
      have h_gt : axis.val < i := by omega
      have hi' : i - 1 < indices.values.length := by omega
      have hv := indices.valid (i - 1) (Nat.lt_of_lt_of_eq hi' indices.length_eq)
      have h_nlt' : ¬(i - 1 < axis.val) := by omega
      simp only [List.getElem_eraseIdx, dif_neg h_nlt'] at hv
      -- hv : indices.values[i-1] < σ[i-1+1], goal: indices.values[i-1] < σ[i]
      -- Since i - 1 + 1 = i (h_gt : axis.val < i so i ≥ 1)
      exact Nat.lt_of_lt_of_eq hv (by congr 1; omega)

/-- All-zeros index for a shape where every dimension is positive. -/
def zeros (σ : Shape) (hpos : ∀ i (hi : i < σ.rank), 0 < σ[i]'hi) :
    Indices σ :=
  ⟨List.replicate σ.rank 0,
   by simp,
   fun i hi => by
     simp only [List.getElem_replicate]
     exact hpos i hi⟩

/-- Apply the inverse permutation to indices of a permuted shape,
    mapping them back to indices of the original shape. -/
def invPermute {σ : Shape} {perm : Permutation}
    {hv : Permutation.Valid perm σ.rank}
    (indices : Indices (σ.permute perm hv)) : Indices σ :=
  let inv := perm.invert hv
  ⟨inv.map (fun p => indices.values.getD p 0),
   by simp only [List.length_map]; exact hv.invert_length,
   fun j hj => by
     simp only [List.getElem_map]
     -- inv[j] is in bounds for indices.values
     have hinv_j_lt : inv[j]'(by rw [hv.invert_length]; exact hj) <
         indices.values.length := by
       simp only [indices.length_eq, Shape.rank, Shape.permute, List.length_pmap, hv.length_eq]
       exact hv.invert_forall_lt _ (List.getElem_mem ..)
     rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hinv_j_lt, Option.getD_some]
     have hval := indices.valid
       (inv[j]'(by rw [hv.invert_length]; exact hj))
       (by simp [Shape.permute, List.length_pmap, hv.length_eq];
           exact hv.invert_forall_lt _ (List.getElem_mem ..))
     -- (σ.permute perm hv)[inv[j]] = σ[j]
     rw [Shape.permute_getElem_invert σ perm hv j hj] at hval
     exact hval⟩

/-- Head of a `cons` index is the prepended `Fin`. -/
@[simp]
theorem cons_head {i : Nat} {hi : i < d} {rest : Indices ds} :
    (cons i hi rest).head = ⟨i, hi⟩ := by
  simp only [head, cons]
  exact Fin.ext rfl

/-- Tail of a `cons` index is the original tail. -/
@[simp]
theorem cons_tail {i : Nat} {hi : i < d} {rest : Indices ds} :
    (cons i hi rest).tail = rest := by
  cases rest
  simp only [tail, cons, List.tail_cons]

/-- Two `Indices` are equal when their raw value lists are equal. -/
@[ext]
theorem ext {indices₁ indices₂ : Indices σ} (h : indices₁.values = indices₂.values) : indices₁ = indices₂ := by
  cases indices₁; cases indices₂; simp at h; subst h; rfl

/-- Concatenate indices for `σ₁` and `σ₂` into indices for `σ₁ ++ σ₂`. -/
def append {σ₁ σ₂ : Shape} (i : Indices σ₁) (j : Indices σ₂) : Indices (σ₁ ++ σ₂) :=
  ⟨i.values ++ j.values,
   by simp [i.length_eq, j.length_eq, Shape.rank],
   fun k hk => by
     by_cases hlt : k < σ₁.rank
     · have hi : k < i.values.length := by rw [i.length_eq]; exact hlt
       rw [List.getElem_append_left hi, List.getElem_append_left hlt]
       exact i.valid k hlt
     · have hi : i.values.length ≤ k := by rw [i.length_eq]; omega
       have hσ : σ₁.length ≤ k := by simp only [Shape.rank] at hlt; omega
       rw [List.getElem_append_right hi, List.getElem_append_right hσ]
       have hk' : k - σ₁.length < σ₂.rank := by
         simp only [Shape.rank, List.length_append] at hk ⊢; omega
       have := j.valid (k - σ₁.length) hk'
       simpa [i.length_eq, Shape.rank] using this⟩

@[simp] theorem append_values {σ₁ σ₂ : Shape} (i : Indices σ₁) (j : Indices σ₂) :
    (i.append j).values = i.values ++ j.values := rfl

end Indices

end Palladium.TensorLib
