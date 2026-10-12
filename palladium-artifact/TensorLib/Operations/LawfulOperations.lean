import TensorLib.Notations
import TensorLib.Operations.Basics
import TensorLib.Tensor

namespace Palladium.TensorLib

open Shape

/-!
# Lawful Operations

Laws relating `scalar`, `get`, and `index` to other tensor operations.
These lemmas are the workhorses for compiler correctness proofs.

## Structure

- `Tensor.LawfulOperations`: typeclass adding equational laws to `Operations`
- Derived `index`-level lemmas
-/

/-!
## LawfulOperations typeclass

Core laws about how `scalar`, `get`, and `index` interact with `map`,
`fill`, `broadcast`, `zipWithM`, `transpose`, and `reshape`. These are
the primitive building blocks from which derived lemmas are obtained.
-/

/-- A `LawfulOperations` instance certifies that a Layout's methods
    interact with `scalar` and `get` in the expected ways. -/
class Tensor.LawfulOperations (L : Tensor.Layout)
    [Tensor.Operations L] : Prop where
  /-- `scalar` distributes over `map`. -/
  scalar_map : ∀ {α β : Type} (t : L [] α) (f : α → β),
    Tensor.Operations.scalar (Tensor.Operations.map t f) =
      f (Tensor.Operations.scalar t)
  /-- `get` distributes over `map`. -/
  get_map : ∀ {d : Nat} {σ : Shape} {α β : Type}
    (t : L (d :: σ) α) (f : α → β) (i : Fin d),
    Tensor.Operations.get (Tensor.Operations.map t f) i =
      Tensor.Operations.map (Tensor.Operations.get t i) f
  /-- `scalar` of a rank-0 filled tensor is the fill value. -/
  scalar_fill : ∀ {α : Type} (v : α),
    Tensor.Operations.scalar (Tensor.Operations.fill (L := L) v []) = v
  /-- `get` of a filled tensor yields a filled sub-tensor. -/
  get_fill : ∀ {α : Type} (v : α) (d : Nat) (σ : Shape) (i : Fin d),
    Tensor.Operations.get (Tensor.Operations.fill (L := L) v (d :: σ)) i =
      Tensor.Operations.fill v σ
  /-- `scalar` of a pure zipWith on rank-0 tensors applies the function
      to the two scalars. -/
  scalar_zipWith : ∀ {α β γ : Type} (t1 : L [] α) (t2 : L [] β)
    (f : α → β → γ),
    Tensor.Operations.scalar
      (Id.run (Tensor.Operations.zipWithM t1 t2
        (fun a b => pure (f a b)))) =
      f (Tensor.Operations.scalar t1) (Tensor.Operations.scalar t2)
  /-- `get` distributes over pure zipWith. -/
  get_zipWith : ∀ {d : Nat} {σ : Shape} {α β γ : Type}
    (t1 : L (d :: σ) α) (t2 : L (d :: σ) β) (f : α → β → γ) (i : Fin d),
    Tensor.Operations.get
      (Id.run (Tensor.Operations.zipWithM t1 t2
        (fun a b => pure (f a b)))) i =
      Id.run (Tensor.Operations.zipWithM
        (Tensor.Operations.get t1 i) (Tensor.Operations.get t2 i)
        (fun a b => pure (f a b)))
  /-- `scalar` of a pure zipWith3 on rank-0 tensors applies the function
      to the three scalars. -/
  scalar_zipWith3 : ∀ {α β γ δ : Type} (t1 : L [] α) (t2 : L [] β)
    (t3 : L [] γ) (f : α → β → γ → δ),
    Tensor.Operations.scalar
      (Id.run (Tensor.Operations.zipWith3M t1 t2 t3
        (fun a b c => pure (f a b c)))) =
      f (Tensor.Operations.scalar t1) (Tensor.Operations.scalar t2)
        (Tensor.Operations.scalar t3)
  /-- `get` distributes over pure zipWith3. -/
  get_zipWith3 : ∀ {d : Nat} {σ : Shape} {α β γ δ : Type}
    (t1 : L (d :: σ) α) (t2 : L (d :: σ) β) (t3 : L (d :: σ) γ)
    (f : α → β → γ → δ) (i : Fin d),
    Tensor.Operations.get
      (Id.run (Tensor.Operations.zipWith3M t1 t2 t3
        (fun a b c => pure (f a b c)))) i =
      Id.run (Tensor.Operations.zipWith3M
        (Tensor.Operations.get t1 i) (Tensor.Operations.get t2 i)
        (Tensor.Operations.get t3 i)
        (fun a b c => pure (f a b c)))
  /-- Broadcasting commutes with pointwise mapping. -/
  broadcast_map : ∀ {α β : Type} {σₛ σₜ : Shape}
    (t : L σₛ α) (f : α → β) (h : σₛ ≤ σₜ),
    Tensor.Operations.broadcast (Tensor.Operations.map t f) σₜ h =
      Tensor.Operations.map (Tensor.Operations.broadcast t σₜ h) f
  /-- Broadcasting a rank-0 tensor to a shape is the same as filling. -/
  broadcast_scalar : ∀ {α : Type} (t : L [] α) (σ : Shape)
    (h : [] ≤ σ),
    Tensor.Operations.broadcast t σ h =
      Tensor.Operations.fill (Tensor.Operations.scalar t) σ
  /-- Indexing a broadcast result at valid target indices yields the same value
      as indexing the source at some (projected) indices. This captures how
      broadcasting preserves values while potentially duplicating them. -/
  broadcast_index : ∀ {α : Type} {σₛ σₜ : Shape}
    (t : L σₛ α) (h : σₛ ≤ σₜ)
    (indices : Indices σₜ),
    ∃ indices' : Indices σₛ,
      Tensor.Operations.index σₜ
        (Tensor.Operations.broadcast t σₜ h) indices =
      Tensor.Operations.index σₛ t indices'
  /-- Broadcasting to the same shape with `Broadcastable.refl` is the identity. -/
  broadcast_broadcastable_refl : ∀ {α : Type} {σ : Shape} (t : L σ α),
    Tensor.Operations.broadcast t σ (Shape.Broadcastable.refl σ) = t
  /-- Broadcasting is transitive: broadcasting twice gives the same result
      as broadcasting directly to the final shape. -/
  broadcast_broadcast : ∀ {α : Type} {σ₁ σ₂ σ₃ : Shape}
    (t : L σ₁ α) (h12 : σ₁ ≤ σ₂) (h23 : σ₂ ≤ σ₃)
    (h13 : σ₁ ≤ σ₃),
    Tensor.Operations.broadcast (Tensor.Operations.broadcast t σ₂ h12) σ₃ h23 =
      Tensor.Operations.broadcast t σ₃ h13
  /-- Broadcasting distributes over pure zipWith. -/
  broadcast_zipWith : ∀ {α β γ : Type} {σₛ σₜ : Shape}
    (t1 : L σₛ α) (t2 : L σₛ β) (f : α → β → γ)
    (h : σₛ ≤ σₜ),
    Tensor.Operations.broadcast
      (Id.run (Tensor.Operations.zipWithM t1 t2 (fun a b => pure (f a b))))
      σₜ h =
    Id.run (Tensor.Operations.zipWithM
      (Tensor.Operations.broadcast t1 σₜ h)
      (Tensor.Operations.broadcast t2 σₜ h)
      (fun a b => pure (f a b)))
  /-- Broadcasting distributes over pure zipWith3. -/
  broadcast_zipWith3 : ∀ {α β γ δ : Type} {σₛ σₜ : Shape}
    (t1 : L σₛ α) (t2 : L σₛ β) (t3 : L σₛ γ) (f : α → β → γ → δ)
    (h : σₛ ≤ σₜ),
    Tensor.Operations.broadcast
      (Id.run (Tensor.Operations.zipWith3M t1 t2 t3 (fun a b c => pure (f a b c))))
      σₜ h =
    Id.run (Tensor.Operations.zipWith3M
      (Tensor.Operations.broadcast t1 σₜ h)
      (Tensor.Operations.broadcast t2 σₜ h)
      (Tensor.Operations.broadcast t3 σₜ h)
      (fun a b c => pure (f a b c)))
  /-- `get` distributes over pure `mapSlicesM`. -/
  get_mapSlices : ∀ {d : Nat} {σ σ' : Shape} {α β : Type}
    (xs : L (d :: σ) α) (f : L σ α → L σ' β) (i : Fin d),
    Tensor.Operations.get
      (Id.run (Tensor.Operations.mapSlicesM xs (m := Id) (fun s => pure (f s)))) i =
      f (Tensor.Operations.get xs i)
  /-- Broadcasting from `shape` to `d :: shape` (prepending a dimension) and
      then indexing at any slice gives back the original tensor.
      The prepended dimension is replicated, so the slice index is irrelevant. -/
  get_broadcast_cons_left : ∀ {α : Type} {σ : Shape} (d : Nat)
    (t : L σ α) (h : σ ≤ (d :: σ))
    (i : Fin d),
    Tensor.Operations.get (Tensor.Operations.broadcast t (d :: σ) h) i = t
  /-- `squeeze` at axis 0 produces the same result as `get` at index 0. -/
  squeeze_zero : ∀ {α : Type} {σ : Shape}
    (t : L (1 :: σ) α) (h : Shape.Squeezable (1 :: σ) ⟨0, by simp [Shape.rank]⟩),
    Tensor.Operations.squeeze t ⟨0, by simp [Shape.rank]⟩ h =
      Tensor.Operations.get t ⟨0, by simp⟩
  /-- `get` distributes over `squeeze` at axis > 0. -/
  get_squeeze : ∀ {d : Nat} {σ : Shape} {α : Type}
    (t : L (d :: σ) α) (axis : Fin σ.rank)
    (hone : σ[axis.val]'axis.isLt = 1)
    (i : Fin d),
    Tensor.Operations.get
      (Tensor.Operations.squeeze t
        ⟨axis.val + 1, Nat.succ_lt_succ axis.isLt⟩
        (by simpa [List.getElem_cons_succ] using hone)) i =
    Tensor.Operations.squeeze (Tensor.Operations.get t i) axis hone
  /-- When source and target have equal leading dimensions and equal lengths,
      `get` distributes over `broadcast`. -/
  get_broadcast_eq : ∀ {α : Type} {d : Nat} {σₛ σₜ : Shape}
    (t : L (d :: σₛ) α)
    (bc : (d :: σₛ) ≤ (d :: σₜ))
    (bc' : σₛ ≤ σₜ)
    (_h_len : σₛ.rank = σₜ.rank)
    (i : Fin d),
    Tensor.Operations.get (Tensor.Operations.broadcast t (d :: σₜ) bc) i =
      Tensor.Operations.broadcast (Tensor.Operations.get t i) σₜ bc'
  /-- When source has leading dimension 1 and source/target have equal lengths,
      `get` of a broadcast result at any index gives the broadcast of the single
      source slice at index 0. -/
  get_broadcast_one : ∀ {α : Type} {σₛ σₜ : Shape} {d : Nat}
    (t : L (1 :: σₛ) α)
    (bc : (1 :: σₛ) ≤ (d :: σₜ))
    (bc' : σₛ ≤ σₜ)
    (_h_len : σₛ.rank = σₜ.rank)
    (i : Fin d),
    Tensor.Operations.get (Tensor.Operations.broadcast t (d :: σₜ) bc) i =
      Tensor.Operations.broadcast (Tensor.Operations.get t ⟨0, by omega⟩) σₜ bc'
  /-- Reshape from `[n]` to `[n, 1^k]` preserves element access: getting the i-th
      slice at all-zeros indices gives the same as indexing the original at `[i]`. -/
  index_reshape_replicate_ones : ∀ {α : Type} {n k : Nat}
    (t : L [n] α)
    (h : [n] ≅ₛ (n :: List.replicate k 1))
    (i : Nat) (hi : i < n),
    Tensor.Operations.index (List.replicate k 1)
      (Tensor.Operations.get
        (Tensor.reshape t (n :: List.replicate k 1) h)
        ⟨i, hi⟩)
      (Indices.zeros (List.replicate k 1)
        (fun j hj => by rw [List.getElem_replicate]; omega)) =
    Tensor.Operations.index [n] t (Indices.cons i hi Indices.nil)
  /-- Indexing equals reading the row-major flattening at the flat index. -/
  index_eq_getElem_flatten : ∀ {α : Type} {σ : Shape} (t : L σ α) (indices : Indices σ),
    Tensor.Operations.index σ t indices =
      (Tensor.Operations.flatten t)[Shape.ravelIndex (Shape.computeStrides σ) indices.values]'(by
        rw [Tensor.Operations.flatten_length]
        exact Shape.ravelIndex_lt_volume σ indices.values indices.length_eq indices.valid)
  /-- Flattening an unflattened list gives the list back. -/
  flatten_unflatten : ∀ {α : Type} (σ : Shape) (xs : List α) (h : xs.length = σ.volume),
    Tensor.Operations.flatten (Tensor.Operations.unflatten (L := L) σ xs h) = xs
  /-- Indexing a transposed tensor at `indices` equals indexing the original
      tensor at the inverse-permuted indices. -/
  index_transpose : ∀ {α : Type} {σ : Shape}
    (t : L σ α) (perm : Permutation) (hv : Permutation.Valid perm σ.rank)
    (indices : Indices (σ.permute perm hv)),
    Tensor.Operations.index (σ.permute perm hv)
      (Tensor.Operations.transpose t perm hv) indices =
    Tensor.Operations.index σ t indices.invPermute

end Palladium.TensorLib
