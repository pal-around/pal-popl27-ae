import TensorLib.Notations
import TensorLib.Operations.LawfulOperations

namespace Palladium.TensorLib

open Shape

/-!
## Derived lemmas

These theorems are derived from the core laws in `LawfulOperations`.
All lemmas use the total `index` API (indexing via `Indices`).
-/

section

variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]

/-- Broadcasting distributes over `fill`: broadcasting a filled tensor
    to any compatible target gives a filled tensor of the target shape. -/
theorem Tensor.Operations.broadcast_fill {α : Type} {σₛ σₜ : Shape}
    (v : α) (bc : σₛ ≤ σₜ) :
    Tensor.Operations.broadcast (Tensor.Operations.fill (L := L) v σₛ) σₜ bc =
    Tensor.Operations.fill v σₜ := by
  have bc_nil : [] ≤ σₛ := Shape.Broadcastable.nil_left σₛ
  have h1 : Tensor.Operations.fill (L := L) v σₛ =
      Tensor.Operations.broadcast (Tensor.Operations.fill v []) σₛ bc_nil := by
    rw [Tensor.LawfulOperations.broadcast_scalar, Tensor.LawfulOperations.scalar_fill]
  rw [h1, Tensor.LawfulOperations.broadcast_broadcast _ bc_nil bc (bc_nil.trans bc),
      Tensor.LawfulOperations.broadcast_scalar, Tensor.LawfulOperations.scalar_fill]

/-- Broadcasting commutes with the foldr in sumOuterAxis: broadcasting the
    accumulated sum equals summing the broadcasted slices. -/
private theorem broadcast_sumOuterAxis_foldr {d : Nat} {σₛ σₜ : Shape} {α : Type}
    (xs : L (d :: σₛ) α) (add : α → α → α)
    (bc : σₛ ≤ σₜ)
    (bc_full : (d :: σₛ) ≤ (d :: σₜ))
    (hlen : σₛ.rank = σₜ.rank)
    (items : List Nat) (hitems : ∀ j ∈ items, j < d)
    (acc : L σₛ α) :
    Tensor.Operations.broadcast
      (items.foldr (fun j a =>
        if h : j < d then
          Id.run (Tensor.Operations.zipWithM
            (Tensor.Operations.get xs ⟨j, h⟩)
            a
            (fun x y => pure (add x y)))
        else a) acc) σₜ bc =
    items.foldr (fun j a =>
      if h : j < d then
        Id.run (Tensor.Operations.zipWithM
          (Tensor.Operations.get
            (Tensor.Operations.broadcast xs (d :: σₜ) bc_full) ⟨j, h⟩)
          a
          (fun x y => pure (add x y)))
      else a) (Tensor.Operations.broadcast acc σₜ bc) := by
  induction items generalizing acc with
  | nil => simp
  | cons j js ih =>
    simp only [List.foldr_cons]
    have hj : j < d := hitems j (.head _)
    simp only [hj, ↓reduceDIte]
    rw [Tensor.LawfulOperations.broadcast_zipWith,
        (Tensor.LawfulOperations.get_broadcast_eq xs bc_full bc hlen ⟨j, hj⟩).symm,
        ih (hitems := fun i hi => hitems i (.tail _ hi)) acc]

/-- Broadcasting commutes with sum at axis 0: broadcasting the sum
    equals summing the broadcasted tensor. -/
theorem Tensor.Operations.broadcast_sum_axis0 {d : Nat} {σₛ σₜ : Shape} {α : Type}
    (t : L (d :: σₛ) α) (add : α → α → α) (zero : α)
    (bc : σₛ ≤ σₜ)
    (bc_full : (d :: σₛ) ≤ (d :: σₜ))
    (hlen : σₛ.rank = σₜ.rank) :
    Tensor.Operations.broadcast
      (Tensor.sum (L := L) add zero (d :: σₛ) t ⟨0, Nat.succ_pos _⟩) σₜ bc =
    Tensor.sum (L := L) add zero (d :: σₜ)
      (Tensor.Operations.broadcast t (d :: σₜ) bc_full) ⟨0, Nat.succ_pos _⟩ := by
  show Tensor.Operations.broadcast
    ((List.range d).foldr (fun j a =>
      if h : j < d then
        Id.run (Tensor.Operations.zipWithM
          (Tensor.Operations.get t ⟨j, h⟩)
          a
          (fun x y => pure (add x y)))
      else a) (Tensor.Operations.fill zero σₛ)) σₜ bc =
    (List.range d).foldr (fun j a =>
      if h : j < d then
        Id.run (Tensor.Operations.zipWithM
          (Tensor.Operations.get
            (Tensor.Operations.broadcast t (d :: σₜ) bc_full) ⟨j, h⟩)
          a
          (fun x y => pure (add x y)))
      else a) (Tensor.Operations.fill zero σₜ)
  rw [broadcast_sumOuterAxis_foldr t add bc bc_full hlen
    (List.range d) (fun j hj => List.mem_range.mp hj),
    Tensor.Operations.broadcast_fill]

end

/-!
## Total `index`-level lemmas

These theorems relate `Tensor.Operations.index` to tensor operations.
They are proved directly by induction on shape using the core `LawfulOperations` laws.
-/

section

variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]

/-- `index` of a filled tensor returns the fill value. -/
theorem Tensor.Operations.index_fill {α : Type} (v : α)
    (σ : Shape) (indices : Indices σ) :
    Tensor.Operations.index σ (Tensor.Operations.fill (L := L) v σ) indices = v := by
  induction σ with
  | nil => simp [Tensor.Operations.index, Tensor.LawfulOperations.scalar_fill]
  | cons d ds ih =>
    simp only [Tensor.Operations.index]
    rw [Tensor.LawfulOperations.get_fill]
    exact ih indices.tail

/-- `index` of a broadcasted scalar tensor returns the scalar value. -/
theorem Tensor.Operations.index_broadcast_scalar {α : Type}
    (t : L [] α) (σ : Shape)
    (h : [] ≤ σ)
    (indices : Indices σ) :
    Tensor.Operations.index σ
      (Tensor.Operations.broadcast t σ h) indices =
    Tensor.Operations.scalar t := by
  rw [Tensor.LawfulOperations.broadcast_scalar]
  exact Tensor.Operations.index_fill _ _ _

/-- `index` distributes over `map`. -/
theorem Tensor.Operations.index_map {α β : Type} {σ : Shape}
    (t : L σ α) (f : α → β) (indices : Indices σ) :
    Tensor.Operations.index σ (Tensor.Operations.map t f) indices =
    f (Tensor.Operations.index σ t indices) := by
  induction σ with
  | nil => simp [Tensor.Operations.index, Tensor.LawfulOperations.scalar_map]
  | cons d ds ih =>
    simp only [Tensor.Operations.index]
    rw [Tensor.LawfulOperations.get_map]
    exact ih _ _

/-- `index` distributes over pure `zipWith`. -/
theorem Tensor.Operations.index_zipWith {α β γ : Type} {σ : Shape}
    (t1 : L σ α) (t2 : L σ β)
    (f : α → β → γ) (indices : Indices σ) :
    Tensor.Operations.index σ
      (Id.run (Tensor.Operations.zipWithM t1 t2
        (fun a b => pure (f a b)))) indices =
    f (Tensor.Operations.index σ t1 indices)
      (Tensor.Operations.index σ t2 indices) := by
  induction σ with
  | nil => simp [Tensor.Operations.index, Tensor.LawfulOperations.scalar_zipWith]
  | cons d ds ih =>
    simp only [Tensor.Operations.index]
    rw [Tensor.LawfulOperations.get_zipWith]
    exact ih _ _ _

/-- `index` distributes over pure `zipWith3`. -/
theorem Tensor.Operations.index_zipWith3 {α β γ δ : Type} {σ : Shape}
    (t1 : L σ α) (t2 : L σ β) (t3 : L σ γ)
    (f : α → β → γ → δ) (indices : Indices σ) :
    Tensor.Operations.index σ
      (Id.run (Tensor.Operations.zipWith3M t1 t2 t3
        (fun a b c => pure (f a b c)))) indices =
    f (Tensor.Operations.index σ t1 indices)
      (Tensor.Operations.index σ t2 indices)
      (Tensor.Operations.index σ t3 indices) := by
  induction σ with
  | nil => simp [Tensor.Operations.index, Tensor.LawfulOperations.scalar_zipWith3]
  | cons d ds ih =>
    simp only [Tensor.Operations.index]
    rw [Tensor.LawfulOperations.get_zipWith3]
    exact ih _ _ _ _

/-- Broadcasting from `shape` to `d :: σₜ` and indexing at `cons i rest` gives
    the same result as broadcasting to `σₜ` and indexing at `rest`. -/
theorem Tensor.Operations.broadcast_index_cons_left {α : Type}
    {σₛ σₜ : Shape} (d : Nat)
    (t : L σₛ α)
    (h : σₛ ≤ (d :: σₜ))
    (h' : σₛ ≤ σₜ)
    (i : Nat) (hi : i < d) (rest : Indices σₜ) :
    Tensor.Operations.index (d :: σₜ)
      (Tensor.Operations.broadcast t (d :: σₜ) h) (Indices.cons i hi rest) =
    Tensor.Operations.index σₜ
      (Tensor.Operations.broadcast t σₜ h') rest := by
  -- index for cons σₛ unfolds to: index σₜ (get (...) ⟨i, hi⟩) rest
  simp only [Tensor.Operations.index, Indices.cons_head, Indices.cons_tail]
  -- Decompose broadcast to d :: σₜ as broadcast(broadcast(t, σₜ), d :: σₜ)
  have h_td : σₜ ≤ (d :: σₜ) :=
    Shape.Broadcastable.cons_left d (Shape.Broadcastable.refl σₜ)
  rw [show Tensor.Operations.broadcast t (d :: σₜ) h =
    Tensor.Operations.broadcast (Tensor.Operations.broadcast t σₜ h') (d :: σₜ) h_td from
    (Tensor.LawfulOperations.broadcast_broadcast t h' h_td h).symm]
  -- Apply get_broadcast_cons_left: get (broadcast _ (d :: σₜ) _) i = _
  rw [Tensor.LawfulOperations.get_broadcast_cons_left d
    (Tensor.Operations.broadcast t σₜ h') h_td ⟨i, hi⟩]

/-- Like `broadcast_index_cons_left` but takes an arbitrary `Indices (d :: σₜ)` instead
    of requiring `Indices.cons i hi rest` form. Useful when indices come from `Environment.toIndices`. -/
theorem Tensor.Operations.broadcast_index_cons_left' {α : Type}
    {σₛ σₜ : Shape} (d : Nat)
    (t : L σₛ α)
    (h : σₛ ≤ (d :: σₜ))
    (h' : σₛ ≤ σₜ)
    (indices : Indices (d :: σₜ)) :
    Tensor.Operations.index (d :: σₜ)
      (Tensor.Operations.broadcast t (d :: σₜ) h) indices =
    Tensor.Operations.index σₜ
      (Tensor.Operations.broadcast t σₜ h') indices.tail := by
  exact broadcast_index_cons_left d t h h' indices.head.val indices.head.isLt indices.tail

/-- `(List.finRange d).map g = List.ofFn g` — connects finRange-based iteration with ofFn. -/
theorem finRange_map_eq_ofFn {α : Type} {d : Nat} (g : Fin d → α) :
    (List.finRange d).map g = List.ofFn g := by
  apply List.ext_getElem
  · simp [List.length_ofFn]
  · intro i hi1 hi2
    simp [List.getElem_ofFn, List.getElem_map, List.getElem_finRange]

/-- A foldr over `range d` with a dite guard equals an ofFn-based foldr.
    For elements in `range d`, the guard `j < d` always holds. -/
theorem range_foldr_dite_eq_ofFn_foldr {α : Type} {d : Nat}
    (add : α → α → α) (zero : α) (g : Fin d → α) :
    (List.range d).foldr (fun j (a : α) =>
      if h : j < d then add (g ⟨j, h⟩) a else a) zero =
    (List.ofFn g).foldr add zero := by
  rw [show List.range d = (List.finRange d).map Fin.val
    from List.map_coe_finRange_eq_range.symm]
  rw [List.foldr_map]
  rw [← finRange_map_eq_ofFn g, List.foldr_map]
  congr 1; ext v a
  simp only [v.isLt, ↓reduceDIte]

/-- Helper: `index` distributes over the foldr in `sumOuterAxis`.
    Maintains the invariant that `index` of the accumulator tracks the partial sum. -/
private theorem index_sumOuterAxis_foldr {α : Type} {d : Nat} {σ : Shape}
    (xs : L (d :: σ) α) (add : α → α → α)
    (indices : Indices σ)
    (items : List Nat) (hitems : ∀ j ∈ items, j < d)
    (acc : L σ α) (accVal : α)
    (hacc : Tensor.Operations.index σ acc indices = accVal) :
    Tensor.Operations.index σ
      (items.foldr (fun j a =>
        if h : j < d then
          Id.run (Tensor.Operations.zipWithM
            (Tensor.Operations.get xs ⟨j, h⟩)
            a
            (fun x y => pure (add x y)))
        else a) acc) indices =
    items.foldr (fun j a =>
      if h : j < d then
        add (Tensor.Operations.index σ (Tensor.Operations.get xs ⟨j, h⟩) indices) a
      else a) accVal := by
  induction items generalizing acc accVal with
  | nil => simpa
  | cons j js ih_js =>
    simp only [List.foldr_cons]
    have hj : j < d := hitems j (.head _)
    simp only [hj, ↓reduceDIte]
    rw [Tensor.Operations.index_zipWith]
    congr 1
    exact ih_js (hitems := fun i hi => hitems i (.tail _ hi)) acc accVal hacc

/-- Indexing a `sum` result with total indices equals summing across the axis
    using `Fin`-indexed summation. -/
theorem Tensor.Operations.index_sum {α : Type}
    {σ : Shape} (tensor : L σ α) (add : α → α → α) (zero : α)
    (axis : Fin σ.rank)
    (indices : Indices (σ.slice axis)) :
    Tensor.Operations.index (σ.slice axis)
      (Tensor.sum add zero σ tensor axis) indices =
    (List.ofFn (fun v : Fin (σ[axis.val]'axis.isLt) =>
      Tensor.Operations.index σ tensor
        (indices.insertIdx axis v.val v.isLt))).foldr add zero := by
  induction σ with
  | nil => exact absurd axis.isLt (by simp)
  | cons d ds ih =>
    match axis with
    | ⟨0, _⟩ =>
      -- axis = 0: sum calls sumOuterAxis (foldr over range d)
      show Tensor.Operations.index ds
        ((List.range d).foldr (fun j a =>
          if h : j < d then
            Id.run (Tensor.Operations.zipWithM
              (Tensor.Operations.get tensor ⟨j, h⟩)
              a
              (fun x y => pure (add x y)))
          else a) (Tensor.Operations.fill zero ds)) indices =
        (List.ofFn _).foldr add zero
      -- Apply index_sumOuterAxis_foldr to track index through the foldr
      have h_fill := Tensor.Operations.index_fill (L := L) zero ds indices
      rw [index_sumOuterAxis_foldr tensor add indices
        (List.range d) (fun j hj => List.mem_range.mp hj)
        (Tensor.Operations.fill zero ds) zero h_fill]
      -- Convert range-foldr with dite to ofFn-foldr
      rw [range_foldr_dite_eq_ofFn_foldr add zero
        (fun v => Tensor.Operations.index ds
          (Tensor.Operations.get tensor v) indices)]
      -- The ofFn functions agree definitionally (insertIdx at 0 = cons)
      congr 1
    | ⟨axis + 1, haxis⟩ =>
      -- axis > 0: sum calls mapSlicesM + recursive sum
      have haxis' : axis < ds.length := by simp at haxis; omega
      -- Expose the index unfolding at d :: ds.slice axis
      show Tensor.Operations.index (ds.eraseIdx axis)
        (Tensor.Operations.get
          (Tensor.sum add zero (d :: ds) tensor ⟨axis + 1, haxis⟩) indices.head) indices.tail =
        (List.ofFn _).foldr add zero
      -- get (sum (d :: ds) tensor ⟨axis+1, haxis⟩) indices.head
      -- = get (mapSlicesM tensor (fun slice => sum ds slice ⟨axis, haxis'⟩)) indices.head
      -- = sum ds (get tensor indices.head) ⟨axis, haxis'⟩  [by get_mapSlices]
      rw [show Tensor.Operations.get
        (Tensor.sum add zero (d :: ds) tensor ⟨axis + 1, haxis⟩) indices.head =
        Tensor.Operations.get (Id.run (Tensor.Operations.mapSlicesM tensor (m := Id)
          (fun slice => pure (Tensor.sum add zero ds slice ⟨axis, haxis'⟩)))) indices.head from rfl]
      rw [Tensor.LawfulOperations.get_mapSlices]
      -- Apply IH
      have h_ih := ih (Tensor.Operations.get tensor indices.head) ⟨axis, haxis'⟩ indices.tail
      rw [h_ih]
      -- Show the ofFn functions agree
      congr 1
      apply List.ext_getElem
      · simp [List.length_ofFn]
      · intro i hi1 hi2
        simp only [List.getElem_ofFn]
        -- Need: index (d :: ds) tensor (indices.insertIdx ⟨axis+1,haxis⟩ v.val v.isLt)
        --      = index ds (get tensor indices.head) (indices.tail.insertIdx ⟨axis,haxis'⟩ v.val v.isLt)
        -- Unfold index at (d :: ds) level
        show Tensor.Operations.index ds
          (Tensor.Operations.get tensor indices.head)
          (indices.tail.insertIdx ⟨axis, haxis'⟩ (⟨i, by simpa using hi1⟩ : Fin _).val (⟨i, by simpa using hi1⟩ : Fin _).isLt) =
          Tensor.Operations.index ds
            (Tensor.Operations.get tensor
              (indices.insertIdx ⟨axis + 1, haxis⟩ (⟨i, by simpa using hi2⟩ : Fin _).val (⟨i, by simpa using hi2⟩ : Fin _).isLt).head)
            (indices.insertIdx ⟨axis + 1, haxis⟩ (⟨i, by simpa using hi2⟩ : Fin _).val (⟨i, by simpa using hi2⟩ : Fin _).isLt).tail
        -- insertIdx at axis+1 on indices (for d :: ...) preserves head, inserts in tail
        have hne : indices.values ≠ [] := by
          have hlen := indices.length_eq; intro h
          simp [h] at hlen
        obtain ⟨a, rest, ha⟩ : ∃ a rest, indices.values = a :: rest :=
          ⟨indices.values.head hne, indices.values.tail, (List.cons_head_tail hne).symm⟩
        have h_head : (indices.insertIdx ⟨axis + 1, haxis⟩ (⟨i, by simpa using hi2⟩ : Fin _).val (⟨i, by simpa using hi2⟩ : Fin _).isLt).head = indices.head :=
          Fin.ext (by simp [Indices.head, Indices.insertIdx, ha, List.insertIdx_succ_cons])
        have h_tail : (indices.insertIdx ⟨axis + 1, haxis⟩ (⟨i, by simpa using hi2⟩ : Fin _).val (⟨i, by simpa using hi2⟩ : Fin _).isLt).tail =
            indices.tail.insertIdx ⟨axis, haxis'⟩ (⟨i, by simpa using hi1⟩ : Fin _).val (⟨i, by simpa using hi1⟩ : Fin _).isLt :=
          Indices.ext (by simp [Indices.tail, Indices.insertIdx, ha, List.insertIdx_succ_cons])
        rw [h_head, h_tail]

/-- Broadcasting from a `[1^k]` shape to any target shape returns the
    same value regardless of indices — equivalent to indexing the source
    at all-zeros. -/
theorem Tensor.Operations.index_broadcast_ones_eq {α : Type} {k : Nat}
    (t : L (List.replicate k 1) α) (σₜ : Shape)
    (_hlen : σₜ.rank = k)
    (bc : List.replicate k 1 ≤ σₜ)
    (indices : Indices σₜ) :
    Tensor.Operations.index σₜ
      (Tensor.Operations.broadcast t σₜ bc) indices =
    Tensor.Operations.index (List.replicate k 1) t
      (Indices.zeros (List.replicate k 1)
        (fun i hi => by rw [List.getElem_replicate]; omega)) := by
  -- broadcast_index gives ∃ indices' : Indices (replicate k 1), index σₜ ... = index ... t indices'
  obtain ⟨indices', h⟩ := Tensor.LawfulOperations.broadcast_index t bc indices
  rw [h]
  -- Any Indices (replicate k 1) equals zeros: each value < 1 forces value = 0
  congr 1
  apply Indices.ext
  have hlen_idx : indices'.values.length = k := by
    have := indices'.length_eq; simp [List.length_replicate] at this; exact this
  have hlen_z : (Indices.zeros (List.replicate k 1)
      (fun i hi => by rw [List.getElem_replicate]; omega)).values.length = k := by
    simp [Indices.zeros, List.length_replicate]
  apply List.ext_getElem
  · omega
  · intro i hi1 hi2
    have hi' : i < (List.replicate k 1).length := by
      rw [List.length_replicate]; omega
    have hv := indices'.valid i hi'
    rw [List.getElem_replicate] at hv
    simp only [Indices.zeros, List.length_replicate, List.getElem_replicate]
    omega

end

end Palladium.TensorLib
