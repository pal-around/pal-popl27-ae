import TensorLib.Indices
import TensorLib.Permutation
import Std

namespace Palladium.TensorLib

open Shape

namespace Tensor

/-!
## Operations typeclass (Layout)

A Layout is any `L : Tensor.Layout` with a `Tensor.Operations L` instance.
This typeclass abstracts over shape-indexed tensor data representations; the
`Tabulated` layout (nested vectors) is an instance.
-/

/-- A Layout is a shape-indexed type constructor `Shape → Type → Type`. -/
abbrev Layout := Shape → Type → Type

/-- Typeclass defining a Layout: a shape-indexed tensor data representation. -/
class Operations (L : Tensor.Layout) where
  /-- Build a BEq instance for a given shape. -/
  instBEq : [BEq α] → (σ : Shape) → BEq (L σ α)
  /-- Build an Inhabited instance for a given shape. -/
  instInhabited : [Inhabited α] → (σ : Shape) → Inhabited (L σ α)
  /-- Extract the scalar from a rank-0 tensor in this layout. -/
  scalar : {α : Type} → L [] α → α
  /-- Fill a shape with a scalar value. -/
  fill : α → (σ : Shape) → L σ α
  /-- Index into the outermost axis. -/
  get : {d : Nat} → {σ : Shape} → L (d :: σ) α → Fin d → L σ α
  /-- Total broadcast given a structural broadcastability proof. -/
  broadcast : {α : Type} → {σₛ : Shape} → L σₛ α →
    (σₜ : Shape) → Shape.Broadcastable σₛ σₜ → L σₜ α
  /-- Flatten to a list of scalar elements (row-major). -/
  flatten : {σ : Shape} → L σ α → List α
  /-- Build from a flat list, given a proof that the list has exactly the right length. -/
  unflatten : (σ : Shape) → (xs : List α) →
    xs.length = σ.volume → L σ α
  /-- Total squeeze given a structural squeezability proof. -/
  squeeze : {α : Type} → {σ : Shape} → L σ α → (axis : Fin σ.rank) →
    Shape.Squeezable σ axis →
    L (σ.slice axis) α
  /-- Total transpose given a structural permutation validity proof. -/
  transpose : {α : Type} → {σ : Shape} → L σ α → (perm : Permutation) →
    (h : Permutation.Valid perm σ.rank) →
    L (σ.permute perm h) α
  /-- Map a pure function over every scalar element. -/
  map : {σ : Shape} → L σ α → (α → β) → L σ β
  /-- Apply an effectful function to each slice along the outermost axis. -/
  mapSlicesM : {m : Type → Type} → [Monad m] → {d : Nat} → {σ σ' : Shape} →
    L (d :: σ) α → (L σ α → m (L σ' β)) → m (L (d :: σ') β)
  /-- Effectful pointwise binary operation. -/
  zipWithM : {m : Type → Type} → [Monad m] → {σ : Shape} →
    L σ α → L σ β → (α → β → m γ) → m (L σ γ)
  /-- Effectful pointwise ternary operation. -/
  zipWith3M : {m : Type → Type} → [Monad m] → {σ : Shape} →
    L σ α → L σ β → L σ γ → (α → β → γ → m δ) → m (L σ δ)

  /-- flatten preserves element count. -/
  flatten_length : {α : Type} → {σ : Shape} → (t : L σ α) →
    (flatten t).length = σ.volume

end Tensor

namespace Tensor.Operations

/-- Total scalar lookup by proof-carrying indices. -/
def index {L : Tensor.Layout} [Tensor.Operations L] {α : Type} :
    (σ : Shape) → L σ α → Indices σ → α
  | [], a, _ => Tensor.Operations.scalar a
  | _d :: ds, data, indices =>
    index ds (Tensor.Operations.get data indices.head) indices.tail

/-- Generic transpose via flatten → reindex → unflatten.
    Parameterized by the layout-specific `flatten`/`unflatten` methods so it can be
    used in `Tensor.Operations` instance definitions without circular typeclass lookup.
    Individual layouts may override with a specialized implementation. -/
def transposeByFlattening {L : Tensor.Layout}
    (flatten : {σ : Shape} → L σ α → List α)
    (flatten_length : {σ : Shape} → (t : L σ α) → (flatten t).length = σ.volume)
    (unflatten : (σ : Shape) → (xs : List α) → xs.length = σ.volume → L σ α)
    {σ : Shape}
    (data : L σ α) (perm : Permutation)
    (h : Permutation.Valid perm σ.rank) :
    L (σ.permute perm h) α :=
  let outShape := σ.permute perm h
  let flat := flatten data
  have hflatLen : flat.length = σ.volume := flatten_length data
  have hSizeEq : outShape.volume = σ.volume :=
    Shape.volume_permute σ perm h
  let inv := h.invert
  let inStrides := Shape.computeStrides σ
  let outStrides := Shape.computeStrides outShape
  if hpos : 0 < flat.length then
    let reindexed := List.ofFn (n := outShape.volume) (fun ⟨k, hk⟩ =>
      let outMulti := Shape.unravelIndex outStrides k
      let inMulti := inv.map (fun p => outMulti.getD p 0)
      let inFlat := Shape.ravelIndex inStrides inMulti
      have hInFlat : inFlat < flat.length := by
        rw [hflatLen]; exact Shape.ravelIndex_permuted_lt σ perm h k (hSizeEq ▸ hk)
      flat[inFlat])
    have hReindexLen : reindexed.length = outShape.volume := List.length_ofFn ..
    unflatten outShape reindexed hReindexLen
  else
    unflatten outShape [] (by simp; omega)

end Tensor.Operations

/-- Index into the outermost axis of a layout using a `Nat`. -/
instance (priority := 2000) {L : Tensor.Layout} [Tensor.Operations L]
    {d : Nat} {σ : Shape} {α : Type} :
    GetElem (L (d :: σ) α) Nat (L σ α) (fun _ i => i < d) where
  getElem data i h := Tensor.Operations.get data ⟨i, h⟩

end Palladium.TensorLib
