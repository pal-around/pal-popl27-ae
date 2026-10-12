import TensorLib.Notations
import TensorLib.Operations.Lemmas
import TensorLib.Operations.TabulatedInterop

/-!
# Tensor namespace aliases

Re-exports `Tensor.Operations.*` and `Tensor.LawfulOperations.*` lemmas under
the shorter `Tensor.*` namespace for use in compiler correctness proofs.
-/

namespace Palladium.TensorLib

open Shape Tensor

/-! ## Typeclass aliases -/

/-- Alias for `Tensor.Operations.OfTabulated`. -/
abbrev Tensor.OfTabulated := Tensor.Operations.OfTabulated

/-- Alias for `Tensor.Operations.LawfulOfTabulated`. -/
abbrev Tensor.LawfulOfTabulated := Tensor.Operations.LawfulOfTabulated

/-! ## Index recursion lemmas -/

section
variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Recursion for `Tensor.index` at a cons shape: peel the leading dimension. -/
theorem Tensor.index_cons {d : Nat} {σ : Shape} {α : Type}
    (t : Tensor (L := L) (d :: σ) α) (indices : Indices (d :: σ)) :
    t[indices] = (Tensor.get t indices.head)[indices.tail] :=
  rfl

/-- Base case for `Tensor.index` at the empty shape: extract the scalar. -/
theorem Tensor.index_nil {α : Type}
    (t : Tensor (L := L) [] α) (indices : Indices []) :
    t[indices] = Tensor.scalar t :=
  rfl

/-- Bracket indexing on a layout reduces to `Tensor.index`. -/
theorem Tensor.getElem_eq_index {σ : Shape} {α : Type}
    (t : Tensor (L := L) σ α) (indices : Indices σ) :
    t[indices] = Tensor.index t indices :=
  rfl

end

/-! ## Lemma aliases from `Tensor.Operations.*` (Lemmas.lean) -/

section
variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]

/-- Alias for `Tensor.Operations.broadcast_fill`. -/
theorem Tensor.broadcast_fill {α : Type} {σₛ σₜ : Shape}
    (v : α) (bc : σₛ ≤ σₜ) :
    (Tensor.fill (L := L) v σₛ) ▷ σₜ ‖ bc =
    Tensor.fill v σₜ :=
  Tensor.Operations.broadcast_fill v bc

/-- Alias for `Tensor.Operations.broadcast_sum_axis0`. -/
theorem Tensor.broadcast_sum_axis0 {d : Nat} {σₛ σₜ : Shape} {α : Type}
    (t : L (d :: σₛ) α) (add : α → α → α) (zero : α)
    (bc : σₛ ≤ σₜ)
    (bc_full : (d :: σₛ) ≤ (d :: σₜ))
    (hlen : σₛ.rank = σₜ.rank) :
    (Tensor.sum (L := L) add zero (d :: σₛ) t ⟨0, Nat.succ_pos _⟩) ▷ σₜ ‖ bc =
    Tensor.sum (L := L) add zero (d :: σₜ)
      (t ▷ (d :: σₜ) ‖ bc_full) ⟨0, Nat.succ_pos _⟩ :=
  Tensor.Operations.broadcast_sum_axis0 t add zero bc bc_full hlen

/-- Alias for `Tensor.Operations.index_fill`. -/
theorem Tensor.index_fill {α : Type} (v : α)
    (σ : Shape) (indices : Indices σ) :
    (Tensor.fill (L := L) v σ)[indices] = v :=
  Tensor.Operations.index_fill v σ indices

/-- Alias for `Tensor.Operations.index_broadcast_scalar`. -/
theorem Tensor.index_broadcast_scalar {α : Type}
    (t : L [] α) (σ : Shape)
    (h : [] ≤ σ)
    (indices : Indices σ) :
    (t ▷ σ ‖ h)[indices] =
    Tensor.scalar t :=
  Tensor.Operations.index_broadcast_scalar t σ h indices

/-- Alias for `Tensor.Operations.index_map`. -/
theorem Tensor.index_map {α β : Type} {σ : Shape}
    (t : L σ α) (f : α → β) (indices : Indices σ) :
    (Tensor.map t f)[indices] =
    f t[indices] :=
  Tensor.Operations.index_map t f indices

/-- Alias for `Tensor.Operations.index_zipWith`. -/
theorem Tensor.index_zipWith {α β γ : Type} {σ : Shape}
    (t1 : L σ α) (t2 : L σ β)
    (f : α → β → γ) (indices : Indices σ) :
    (Id.run (Tensor.zipWithM t1 t2
        (fun a b => pure (f a b))))[indices] =
    f t1[indices]
      t2[indices] :=
  Tensor.Operations.index_zipWith t1 t2 f indices

/-- Alias for `Tensor.Operations.index_zipWith3`. -/
theorem Tensor.index_zipWith3 {α β γ δ : Type} {σ : Shape}
    (t1 : L σ α) (t2 : L σ β) (t3 : L σ γ)
    (f : α → β → γ → δ) (indices : Indices σ) :
    (Id.run (Tensor.zipWith3M t1 t2 t3
        (fun a b c => pure (f a b c))))[indices] =
    f t1[indices]
      t2[indices]
      t3[indices] :=
  Tensor.Operations.index_zipWith3 t1 t2 t3 f indices

/-- Alias for `Tensor.Operations.broadcast_index_cons_left'`. -/
theorem Tensor.broadcast_index_cons_left' {α : Type}
    {σₛ σₜ : Shape} (d : Nat)
    (t : L σₛ α)
    (h : σₛ ≤ (d :: σₜ))
    (h' : σₛ ≤ σₜ)
    (indices : Indices (d :: σₜ)) :
    (t ▷ (d :: σₜ) ‖ h)[indices] =
    (t ▷ σₜ ‖ h')[indices.tail] :=
  Tensor.Operations.broadcast_index_cons_left' d t h h' indices

/-- Alias for `Tensor.Operations.index_sum`. -/
theorem Tensor.index_sum {α : Type}
    {σ : Shape} (tensor : L σ α) (add : α → α → α) (zero : α)
    (axis : Fin σ.rank)
    (indices : Indices (σ.slice axis)) :
    (Tensor.sum add zero σ tensor axis)[indices] =
    (List.ofFn (fun v : Fin (σ[axis.val]'axis.isLt) =>
      tensor[indices.insertIdx axis v.val v.isLt])).foldr add zero :=
  Tensor.Operations.index_sum tensor add zero axis indices

/-- Alias for `Tensor.Operations.index_broadcast_ones_eq`. -/
theorem Tensor.index_broadcast_ones_eq {α : Type} {k : Nat}
    (t : L (List.replicate k 1) α) (σₜ : Shape)
    (hlen : σₜ.rank = k)
    (bc : List.replicate k 1 ≤ σₜ)
    (indices : Indices σₜ) :
    (t ▷ σₜ ‖ bc)[indices] =
    t[Indices.zeros (List.replicate k 1)
        (fun i hi => by rw [List.getElem_replicate]; omega)] :=
  Tensor.Operations.index_broadcast_ones_eq t σₜ hlen bc indices

end

/-! ## Lemma aliases from `Tensor.LawfulOperations.*` (LawfulOperations.lean) -/

section
variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]

/-- Alias for `Tensor.LawfulOperations.broadcast_broadcast`. -/
theorem Tensor.broadcast_broadcast {α : Type} {s1 s2 s3 : Shape}
    (t : L s1 α) (h12 : s1 ≤ s2) (h23 : s2 ≤ s3)
    (h13 : s1 ≤ s3) :
    (t ▷ s2 ‖ h12) ▷ s3 ‖ h23 =
    t ▷ s3 ‖ h13 :=
  Tensor.LawfulOperations.broadcast_broadcast t h12 h23 h13

/-- Alias for `Tensor.LawfulOperations.broadcast_zipWith`. -/
theorem Tensor.broadcast_zipWith {α β γ : Type} {σₛ σₜ : Shape}
    (t1 : L σₛ α) (t2 : L σₛ β) (f : α → β → γ)
    (h : σₛ ≤ σₜ) :
    (Id.run (Tensor.zipWithM t1 t2 (fun a b => pure (f a b)))) ▷ σₜ ‖ h =
    Id.run (Tensor.zipWithM
      (t1 ▷ σₜ ‖ h)
      (t2 ▷ σₜ ‖ h)
      (fun a b => pure (f a b))) :=
  Tensor.LawfulOperations.broadcast_zipWith t1 t2 f h

/-- Alias for `Tensor.LawfulOperations.broadcast_zipWith3`. -/
theorem Tensor.broadcast_zipWith3 {α β γ δ : Type} {σₛ σₜ : Shape}
    (t1 : L σₛ α) (t2 : L σₛ β) (t3 : L σₛ γ) (f : α → β → γ → δ)
    (h : σₛ ≤ σₜ) :
    (Id.run (Tensor.zipWith3M t1 t2 t3 (fun a b c => pure (f a b c)))) ▷ σₜ ‖ h =
    Id.run (Tensor.zipWith3M
      (t1 ▷ σₜ ‖ h)
      (t2 ▷ σₜ ‖ h)
      (t3 ▷ σₜ ‖ h)
      (fun a b c => pure (f a b c))) :=
  Tensor.LawfulOperations.broadcast_zipWith3 t1 t2 t3 f h

/-- Alias for `Tensor.LawfulOperations.broadcast_map`. -/
theorem Tensor.broadcast_map {α β : Type} {σₛ σₜ : Shape}
    (t : L σₛ α) (f : α → β) (h : σₛ ≤ σₜ) :
    (Tensor.map t f) ▷ σₜ ‖ h =
    Tensor.map (t ▷ σₜ ‖ h) f :=
  Tensor.LawfulOperations.broadcast_map t f h

/-- Alias for `Tensor.LawfulOperations.broadcast_broadcastable_refl`. -/
theorem Tensor.broadcast_broadcastable_refl {α : Type} {σ : Shape}
    (t : L σ α) :
    t ▷ σ ‖ Broadcastable.refl σ = t :=
  Tensor.LawfulOperations.broadcast_broadcastable_refl t

/-- Alias for `Tensor.LawfulOperations.get_broadcast_cons_left`. -/
theorem Tensor.get_broadcast_cons_left {α : Type} {σ : Shape} (d : Nat)
    (t : L σ α) (h : σ ≤ (d :: σ))
    (i : Fin d) :
    Tensor.get (t ▷ (d :: σ) ‖ h) i = t :=
  Tensor.LawfulOperations.get_broadcast_cons_left d t h i

/-- Alias for `Tensor.LawfulOperations.get_broadcast_eq`. -/
theorem Tensor.get_broadcast_eq {α : Type} {d : Nat} {σₛ σₜ : Shape}
    (t : L (d :: σₛ) α)
    (bc : (d :: σₛ) ≤ (d :: σₜ))
    (bc' : σₛ ≤ σₜ)
    (h_len : σₛ.rank = σₜ.rank)
    (i : Fin d) :
    Tensor.get (t ▷ (d :: σₜ) ‖ bc) i =
    (Tensor.get t i) ▷ σₜ ‖ bc' :=
  Tensor.LawfulOperations.get_broadcast_eq t bc bc' h_len i

/-- Alias for `Tensor.LawfulOperations.index_transpose`. -/
theorem Tensor.index_transpose {α : Type} {σ : Shape}
    (t : L σ α) (perm : Permutation) (hv : Permutation.Valid perm σ.rank)
    (indices : Indices (σ.permute perm hv)) :
    (Tensor.transpose t perm hv)[indices] =
    t[indices.invPermute] :=
  Tensor.LawfulOperations.index_transpose t perm hv indices

/-- Alias for `Tensor.LawfulOperations.index_reshape_replicate_ones`. -/
theorem Tensor.index_reshape_replicate_ones {α : Type} {n k : Nat}
    (t : L [n] α)
    (h : [n] ≅ₛ (n :: List.replicate k 1))
    (i : Nat) (hi : i < n) :
    (Tensor.get (Tensor.reshape t (n :: List.replicate k 1) h) ⟨i, hi⟩)[Indices.zeros (List.replicate k 1)
        (fun j hj => by rw [List.getElem_replicate]; omega)] =
    t[Indices.cons i hi Indices.nil] :=
  Tensor.LawfulOperations.index_reshape_replicate_ones t h i hi

end

/-! ## Lemma aliases from `Tensor.Operations.LawfulOfTabulated.*` -/

section
variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.Operations.OfTabulated L] [Tensor.Operations.LawfulOfTabulated L]

/-- Alias for `Tensor.Operations.LawfulOfTabulated.scalar_ofTabulated`. -/
theorem Tensor.scalar_ofTabulated {α : Type} (data : Tabulated [] α) :
    Tensor.scalar (Tensor.ofTabulated (L := L) data) = data :=
  Tensor.Operations.LawfulOfTabulated.scalar_ofTabulated data

/-- Alias for `Tensor.Operations.LawfulOfTabulated.get_ofTabulated`. -/
theorem Tensor.get_ofTabulated {d : Nat} {σ : Shape} {α : Type}
    (data : Tabulated (d :: σ) α) (i : Fin d) :
    Tensor.get (Tensor.ofTabulated (L := L) data) i =
    Tensor.ofTabulated (data[i]) :=
  Tensor.Operations.LawfulOfTabulated.get_ofTabulated data i

end

end Palladium.TensorLib
