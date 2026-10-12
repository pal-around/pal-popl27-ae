import TensorLib.Shape
import TensorLib.Permutation
import TensorLib.Operations.Basics
import TensorLib.Indices

namespace Palladium.TensorLib

open Shape

/-- Shape-indexed tensor data parameterized by a layout `L`. -/
abbrev Tensor {L : Tensor.Layout} (σ : Shape) (α : Type) :=
  L σ α

namespace Tensor

section

variable {L : Tensor.Layout} [Tensor.Operations L]

instance [BEq α] {σ : Shape} : BEq (Tensor (L := L) σ α) :=
  Tensor.Operations.instBEq (L := L) (α := α) σ

instance [Inhabited α] {σ : Shape} : Inhabited (Tensor (L := L) σ α) :=
  Tensor.Operations.instInhabited (L := L) (α := α) σ

-- Tensor operations

/-- Total scalar lookup by proof-carrying indices. -/
abbrev index {σ : Shape} {α : Type} (t : Tensor (L := L) σ α) (indices : Indices σ) : α :=
  Tensor.Operations.index σ t indices

/-- Total indexing by proof-carrying `Indices σ`. -/
instance (priority := 1500) {σ : Shape} {α : Type} :
    GetElem (Tensor (L := L) σ α) (Indices σ) α (fun _ _ => True) where
  getElem t indices _ := t.index indices

/-! ## Proof-carrying total operations -/

/-- Total broadcast given a structural broadcastability proof. -/
abbrev broadcast {σₛ : Shape} {α : Type}
    (t : Tensor (L := L) σₛ α) (σₜ : Shape)
    (h : Broadcastable σₛ σₜ) : Tensor (L := L) σₜ α :=
  Tensor.Operations.broadcast t σₜ h

/-- Total squeeze given a structural squeezability proof. -/
abbrev squeeze {σ : Shape} {α : Type}
    (t : Tensor (L := L) σ α) (axis : Fin σ.rank)
    (h : Shape.Squeezable σ axis) : Tensor (L := L) (σ.slice axis) α :=
  Tensor.Operations.squeeze t axis h

/-- Total transpose given a structural permutation validity proof. -/
abbrev transpose {σ : Shape} {α : Type}
    (t : Tensor (L := L) σ α) (perm : Permutation)
    (h : Permutation.Valid perm σ.rank) : Tensor (L := L) (σ.permute perm h) α :=
  Tensor.Operations.transpose t perm h

/-- Total reshape given a structural reshapeability proof. -/
def reshape {σₛ : Shape} {α : Type}
    (t : Tensor (L := L) σₛ α) (σₜ : Shape)
    (h : Reshapeable σₛ σₜ) : Tensor (L := L) σₜ α :=
  Tensor.Operations.unflatten σₜ (Tensor.Operations.flatten t)
    ((Tensor.Operations.flatten_length t).trans h)

/-! ## Pointwise operations -/

/-- Pure pointwise binary operation. -/
def zipWith {σ : Shape} {α β γ : Type}
    (t1 : Tensor (L := L) σ α) (t2 : Tensor (L := L) σ β)
    (f : α → β → γ) : Tensor (L := L) σ γ :=
  Id.run <| Tensor.Operations.zipWithM t1 t2 (fun x y => pure (f x y))

/-- Pure pointwise ternary operation. -/
def zipWith3 {σ : Shape} {α β γ δ : Type}
    (t1 : Tensor (L := L) σ α) (t2 : Tensor (L := L) σ β)
    (t3 : Tensor (L := L) σ γ)
    (f : α → β → γ → δ) : Tensor (L := L) σ δ :=
  Id.run <| Tensor.Operations.zipWith3M t1 t2 t3 (fun x y z => pure (f x y z))

/-- Broadcast two tensors to a common shape, then apply a binary function pointwise. -/
def zipWithBroadcast {α β γ : Type} {σ₁ σ₂ : Shape}
    (a : Tensor (L := L) σ₁ α) (b : Tensor (L := L) σ₂ β)
    (σₜ : Shape) (f : α → β → γ)
    (ha : Broadcastable σ₁ σₜ)
    (hb : Broadcastable σ₂ σₜ) :
    Tensor (L := L) σₜ γ :=
  (a.broadcast σₜ ha).zipWith (b.broadcast σₜ hb) f

/-- Broadcast three tensors to a common shape, then apply a ternary function pointwise. -/
def zipWith3Broadcast {α β γ δ : Type} {σ₁ σ₂ σ₃ : Shape}
    (a : Tensor (L := L) σ₁ α) (b : Tensor (L := L) σ₂ β)
    (c : Tensor (L := L) σ₃ γ)
    (σₜ : Shape) (f : α → β → γ → δ)
    (ha : Broadcastable σ₁ σₜ)
    (hb : Broadcastable σ₂ σₜ)
    (hc : Broadcastable σ₃ σₜ) :
    Tensor (L := L) σₜ δ :=
  (a.broadcast σₜ ha).zipWith3 (b.broadcast σₜ hb) (c.broadcast σₜ hc) f

/-- Transport a tensor along an equality of shapes (used in `sumAxes` and compiler proofs). -/
abbrev tensorCastShape {σ₁ σ₂ : Shape} {α : Type} (h : σ₁ = σ₂)
    (t : Tensor (L := L) σ₁ α) : Tensor (L := L) σ₂ α :=
  cast (congrArg (fun s => Tensor (L := L) s α) h) t

/-- Sum along the outermost (axis-0) dimension using a provided binary operation and zero. -/
private def sumOuterAxis {α : Type} (add : α → α → α) (zero : α) {d : Nat} {σ : Shape}
    (xs : Tensor (L := L) (d :: σ) α) :
    Tensor (L := L) σ α :=
  let zeros : Tensor (L := L) σ α := Tensor.Operations.fill zero σ
  (List.range d).foldr (fun i acc =>
    if h : i < d then
      let slice := Tensor.Operations.get xs ⟨i, h⟩
      slice.zipWith acc add
    else acc
  ) zeros

/-- Sum along an arbitrary axis using a provided binary operation and zero. -/
def sum {α : Type} (add : α → α → α) (zero : α) :
    (σ : Shape) → Tensor (L := L) σ α → (axis : Fin σ.rank) →
    Tensor (L := L) (σ.slice axis) α
  | [], _, axis => Fin.elim0 axis
  | _d :: _ds, xs, ⟨0, _⟩ => sumOuterAxis add zero xs
  | _d :: ds, xs, ⟨axis + 1, haxis⟩ =>
      Id.run <| Tensor.Operations.mapSlicesM xs (m := Id) (fun slice =>
        sum add zero ds slice ⟨axis, by simp at haxis; omega⟩)

/-- Sum along axes given in strictly decreasing order (largest axis first). -/
def sumSortedAxesAux {α : Type} (add : α → α → α) (zero : α) {σ : Shape}
    (t : Tensor (L := L) σ α) (sorted : List (Fin σ.rank))
    (hp : sorted.Pairwise (fun x y => x.val > y.val)) :
    Tensor (L := L) (removeAxes σ sorted) α :=
  match sorted with
  | [] => tensorCastShape (removeAxes_nil σ).symm t
  | a :: as =>
      have hp' := List.pairwise_cons.mp hp
      have hax := hp'.1
      have hall : ∀ x ∈ as, x.val < a.val := hax
      let t1 := sum add zero σ t a
      tensorCastShape (removeAxes_cons_max a as hall).symm
        (sumSortedAxesAux add zero t1 (mapFinsBelowHead a as hall)
          (pairwise_mapFinsBelowHead as hp hall))
termination_by sorted.length
decreasing_by
  simp_wf
  simp [mapFinsBelowHead, List.length_map, List.length_attach]

/-- Sum along every axis in `axes` (order-independent; uses `sortAxesDesc` internally). -/
def sumAxes {α : Type} (add : α → α → α) (zero : α) {σ : Shape}
    (t : Tensor (L := L) σ α) (axes : List (Fin σ.rank)) (h : axes.Nodup) :
    Tensor (L := L) (removeAxes σ axes) α :=
  tensorCastShape (removeAxes_perm_sortAxesDesc axes).symm
    (sumSortedAxesAux add zero t (sortAxesDesc axes) (pairwise_sortAxesDesc axes h))

end

section

variable {L : Tensor.Layout}

private theorem tensorCastShape_trans {σ₁ σ₂ σ₃ : Shape} {α : Type}
    (h₁₂ : σ₁ = σ₂) (h₂₃ : σ₂ = σ₃) (t : Tensor (L := L) σ₁ α) :
    tensorCastShape h₂₃ (tensorCastShape h₁₂ t) = tensorCastShape (h₁₂.trans h₂₃) t := by
  subst h₁₂; subst h₂₃; rfl

end

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Indexing commutes with `tensorCastShape` along a shape equality. -/
theorem index_tensorCastShape {σ₁ σ₂ : Shape} {α : Type} (h : σ₁ = σ₂)
    (t : Tensor (L := L) σ₁ α) (indices : Indices σ₂) :
    Tensor.index (tensorCastShape h t) indices =
      Tensor.index t (indices.cast h.symm) := by
  subst h
  rfl

/-- `sumSortedAxesAux` on a one-element sorted list is nested `tensorCastShape` around `sum`. -/
private theorem sumSortedAxesAux_singleton {α : Type} (add : α → α → α) (zero : α) {σ : Shape}
    (t : Tensor (L := L) σ α) (a : Fin σ.rank) (h : [a].Nodup) :
    sumSortedAxesAux add zero t [a] (pairwise_sortAxesDesc [a] h) =
      tensorCastShape (removeAxes_cons_max a [] (List.pairwise_cons.mp (pairwise_sortAxesDesc [a] h)).1).symm
        (tensorCastShape (removeAxes_nil (σ.slice a)).symm (Tensor.sum add zero σ t a)) := by
  simp [sumSortedAxesAux, mapFinsBelowHead]

/-- Summing along a singleton axis list matches a single `Tensor.sum` up to `tensorCastShape`. -/
theorem sumAxes_singleton {α : Type} (add : α → α → α) (zero : α) {σ : Shape}
    (t : Tensor (L := L) σ α) (a : Fin σ.rank) (h : [a].Nodup) :
    Tensor.sumAxes add zero t [a] h =
      tensorCastShape (removeAxes_singleton σ a).symm (Tensor.sum add zero σ t a) := by
  have hpaxes := pairwise_sortAxesDesc [a] h
  have hall : ∀ x ∈ ([] : List (Fin σ.rank)), x.val < a.val := (List.pairwise_cons.mp hpaxes).1
  have hpTrans :
      (removeAxes_nil (σ.slice a)).symm.trans (removeAxes_cons_max a [] hall).symm =
        (removeAxes_singleton σ a).symm := by
    simp [removeAxes_cons_max, mapFinsBelowHead_nil, removeAxes_nil]
  have hmerge :
      tensorCastShape (removeAxes_cons_max a [] hall).symm
          (tensorCastShape (removeAxes_nil (σ.slice a)).symm (Tensor.sum add zero σ t a)) =
        tensorCastShape (removeAxes_singleton σ a).symm (Tensor.sum add zero σ t a) := by
    rw [tensorCastShape_trans (removeAxes_nil (σ.slice a)).symm (removeAxes_cons_max a [] hall).symm,
      hpTrans]
  have inner_eq :
      sumSortedAxesAux add zero t (sortAxesDesc [a]) hpaxes =
        sumSortedAxesAux add zero t [a] hpaxes :=
    rfl
  calc
    Tensor.sumAxes add zero t [a] h
        = sumSortedAxesAux add zero t [a] hpaxes := by
          simp [sumAxes, inner_eq]
    _ = tensorCastShape (removeAxes_singleton σ a).symm (Tensor.sum add zero σ t a) := by
          rw [sumSortedAxesAux_singleton add zero t a h, hmerge]

/-- Extract the scalar from a rank-0 tensor. -/
abbrev scalar {α : Type} (t : Tensor (L := L) [] α) : α :=
  Tensor.Operations.scalar t

/-- Fill a shape with a scalar value. -/
abbrev fill {α : Type} (val : α) (σ : Shape) : Tensor (L := L) σ α :=
  Tensor.Operations.fill val σ

/-- Index into the outermost axis using a `Fin`. -/
abbrev get {d : Nat} {σ : Shape} {α : Type}
    (t : Tensor (L := L) (d :: σ) α) (i : Fin d) : Tensor (L := L) σ α :=
  Tensor.Operations.get t i

/-- Map a pure function over every scalar element. -/
abbrev map {σ : Shape} {α β : Type} (t : Tensor (L := L) σ α) (f : α → β) : Tensor (L := L) σ β :=
  Tensor.Operations.map t f

/-- Effectful pointwise binary operation. -/
abbrev zipWithM {m : Type → Type} [Monad m] {σ : Shape} {α β γ : Type}
    (t1 : Tensor (L := L) σ α) (t2 : Tensor (L := L) σ β) (f : α → β → m γ) : m (Tensor (L := L) σ γ) :=
  Tensor.Operations.zipWithM t1 t2 f

/-- Effectful pointwise ternary operation. -/
abbrev zipWith3M {m : Type → Type} [Monad m] {σ : Shape} {α β γ δ : Type}
    (t1 : Tensor (L := L) σ α) (t2 : Tensor (L := L) σ β) (t3 : Tensor (L := L) σ γ)
    (f : α → β → γ → m δ) : m (Tensor (L := L) σ δ) :=
  Tensor.Operations.zipWith3M t1 t2 t3 f

end

end Tensor

end Palladium.TensorLib
