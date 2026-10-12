import TensorLib.Operations.Basics
import Std

namespace Palladium.TensorLib

open Shape

/-- Nested Vector type matching the rank of a shape with length enforced at the type level. -/
@[reducible] def Tabulated : Tensor.Layout
  | [], α => α
  | d :: ds, α => Vector (Tabulated ds α) d

namespace Tabulated

/-- BEq instance for Tabulated (scalar case). -/
instance [BEq α] : BEq (Tabulated [] α) where
  beq a b := a == b

/-- BEq instance for Tabulated (Vector case). -/
instance [BEq (Tabulated ds α)] : BEq (Tabulated (d :: ds) α) where
  beq xs ys := xs.toArray == ys.toArray

/-- Inhabited instance for Tabulated (scalar case). -/
instance [Inhabited α] : Inhabited (Tabulated [] α) where
  default := default

/-- Inhabited instance for Tabulated (Vector case). -/
instance [Inhabited (Tabulated ds α)] : Inhabited (Tabulated (d :: ds) α) where
  default := Vector.replicate d default

/-- Map a function over nested tensor data. -/
def map {α β : Type} : {σ : Shape} → Tabulated σ α → (α → β) → Tabulated σ β
  | [], a, f => f a
  | _ :: ds, xs, f => Vector.map (fun x => map (σ := ds) x f) xs

/-- Build a BEq instance for nested tensor data at a given shape. -/
def instBEq {α : Type} [BEq α] : (σ : Shape) -> BEq (Tabulated σ α)
  | [] => inferInstance
  | _ :: ds =>
      let _ : BEq (Tabulated ds α) := instBEq ds
      inferInstance

/-- Build an Inhabited instance for nested tensor data at a given shape. -/
def instInhabited {α : Type} [Inhabited α] : (σ : Shape) → Inhabited (Tabulated σ α)
  | [] => inferInstance
  | _ :: ds =>
      let _ : Inhabited (Tabulated ds α) := instInhabited ds
      inferInstance

/-!
### Tabulated layout helpers

Standalone helpers that the layout instance delegates to.
-/

def zipWithM {m : Type → Type} [Monad m] :
    {σ : Shape} → Tabulated σ α → Tabulated σ β →
    (α → β → m γ) → m (Tabulated σ γ)
  | [], a, b, f => f a b
  | _ :: ds, xs, ys, f =>
      (Vector.zipWith (fun a b => zipWithM (σ := ds) a b f) xs ys).mapM id

def zipWith3M {m : Type → Type} [Monad m] :
    {σ : Shape} → Tabulated σ α → Tabulated σ β → Tabulated σ γ →
    (α → β → γ → m δ) → m (Tabulated σ δ)
  | [], a, b, c, f => f a b c
  | _ :: ds, xs, ys, zs, f =>
      let pairs := Vector.zip ys zs
      (Vector.zipWith (fun a bc => zipWith3M (σ := ds) a bc.1 bc.2 f) xs pairs).mapM id

def mapSlicesM {m : Type → Type} [Monad m]
    {d : Nat} {ds ds' : Shape}
    (xs : Tabulated (d :: ds) α)
    (f : Tabulated ds α → m (Tabulated ds' β)) : m (Tabulated (d :: ds') β) :=
  Vector.mapM f xs

/-- Index into the outermost axis. -/
def get {d : Nat} {ds : Shape} {α : Type} :
    Tabulated (d :: ds) α → Fin d → Tabulated ds α :=
  fun xs i => xs[i]

/-- Extract the scalar from a rank-0 Tabulated tensor. -/
def scalar {α : Type} : Tabulated [] α → α := fun a => a

def fill (v : α) : (σ : Shape) → Tabulated σ α
  | [] => v
  | d :: ds => Vector.replicate d (fill v ds)

def wrapSingletons {α : Type} :
    (σ : Shape) → Tabulated σ α → (n : Nat) →
    Tabulated (List.replicate n 1 ++ σ) α
  | σ, data, 0 => by
      simpa using data
  | σ, data, n + 1 => by
      simpa using
        (Vector.singleton (wrapSingletons σ data n) :
          Vector (Tabulated (List.replicate n 1 ++ σ) α) 1)

/-- Total aligned broadcast, given a structural aligned-broadcastability proof. -/
def broadcastAligned {α : Type} :
    (σₛ σₜ : Shape) → Tabulated σₛ α →
    Shape.BroadcastableAligned σₛ σₜ → Tabulated σₜ α
  | [], [], v, _ => v
  | df :: dfs, dt :: dts, xs, h =>
    let htail := h.tail
    if heq : df = dt then
      heq ▸ Vector.map (fun elem => broadcastAligned dfs dts elem htail) xs
    else
      have : df = 1 := h.one_of_ne heq
      have h0 : 0 < df := by omega
      Vector.replicate dt (broadcastAligned dfs dts (xs[0]'h0) htail)
  | [], _ :: _, _, h => nomatch h
  | _ :: _, [], _, h => nomatch h

/-- Total broadcast given a structural broadcastability proof. -/
def broadcast {α : Type} {σₛ : Shape}
    (data : Tabulated σₛ α) (σₜ : Shape)
    (h : Shape.Broadcastable σₛ σₜ) :
    Tabulated σₜ α :=
  broadcastAligned (Shape.pad σₛ σₜ.length) σₜ
    (wrapSingletons σₛ data (σₜ.length - σₛ.length))
    h

def flatten {α : Type} : {σ : Shape} → Tabulated σ α → List α
  | [], a => [a]
  | _ :: ds, xs => xs.toList.foldl (fun acc x => acc ++ flatten (σ := ds) x) []

/-! ### flatten_length -/

private theorem foldl_append_map_length {α β : Type} (init : List α) (xs : List β) (f : β → List α)
    (n : Nat) (hlen : ∀ x ∈ xs, (f x).length = n) :
    (xs.foldl (fun acc x => acc ++ f x) init).length = init.length + n * xs.length := by
  induction xs generalizing init with
  | nil => simp
  | cons y ys ih =>
    simp only [List.foldl_cons, List.length_cons]
    rw [ih (init ++ f y) (fun z hz => hlen z (List.mem_cons.mpr (Or.inr hz)))]
    have hy := hlen y (List.mem_cons.mpr (Or.inl rfl))
    rw [List.length_append, hy, Nat.mul_succ]
    omega

/-- `flatten` preserves element count: the flat list has exactly `shape.volume` elements. -/
theorem flatten_length {α : Type} : {σ : Shape} → (t : Tabulated σ α) →
    (flatten t).length = σ.volume
  | [], _ => by simp [flatten, Shape.volume]
  | d :: (ds : Shape), xs => by
    simp only [flatten, Shape.volume_cons]
    have hmap : ∀ x ∈ xs.toList, (flatten (σ := ds) x).length = ds.volume := by
      intro x hx
      exact flatten_length x
    rw [foldl_append_map_length [] xs.toList (fun x => flatten (σ := ds) x) ds.volume hmap]
    simp [Vector.length_toList, Nat.mul_comm]

/-- Total unflatten: build a Tabulated tensor from a flat list with a length proof.
    Recursion on `(shape.length, shape.headD 0)`: the `(d+1) :: ds` case calls `ds`
    (length decreases) then `d :: ds` (head decreases). -/
def unflatten {α : Type} : (σ : Shape) → (xs : List α) →
    xs.length = σ.volume → Tabulated σ α
  | [], xs, h => xs[0]'(by rw [Shape.volume_nil] at h; omega)
  | 0 :: _, _, _ => ⟨#[], rfl⟩
  | (d + 1) :: (ds : Shape), xs, h =>
    have hsz : (d + 1) * ds.volume = d * ds.volume + ds.volume := by
      rw [Nat.add_mul, Nat.one_mul]
    have hge : ds.volume ≤ xs.length := by rw [h, Shape.volume_cons]; omega
    let first := unflatten ds (xs.take ds.volume)
      (by rw [List.length_take, Nat.min_eq_left hge])
    let rest := xs.drop ds.volume
    have hrest : rest.length = d * ds.volume := by
      simp only [rest, List.length_drop, h, Shape.volume_cons]; omega
    let more := unflatten (d :: ds) rest (by rw [hrest, Shape.volume_cons])
    ((Vector.singleton first).append more).cast (by omega)
termination_by σ => (σ.length, σ.headD 0)

/-- Total squeeze: remove a singleton axis, given a structural squeezability proof. -/
def squeeze {α : Type} : {σ : Shape} → Tabulated σ α → (axis : Fin σ.rank) →
    Shape.Squeezable σ axis →
    Tabulated (σ.slice axis) α
  | d :: _, xs, ⟨0, _⟩, hunit =>
      have : d = 1 := by simp at hunit; exact hunit
      xs[0]'(by omega)
  | _ :: _, xs, ⟨axis + 1, hlt⟩, hunit =>
      Vector.map (fun elem => squeeze elem
        ⟨axis, by simp [Shape.rank, List.length_cons] at hlt; omega⟩
        (by simp at hunit; exact hunit)) xs

/-- Total transpose: reorders axes according to a valid permutation.
    Delegates to the shared `transposeByFlattening` algorithm. -/
def transpose {α : Type} {σ : Shape}
    (data : Tabulated σ α) (perm : Permutation)
    (h : Permutation.Valid perm σ.length) :
    Tabulated (σ.permute perm h) α :=
  Tensor.Operations.transposeByFlattening flatten flatten_length unflatten data perm h

/-! ### Flatten/unflatten helpers for index_transpose -/

/-- Accumulator shift for foldl with append. -/
private theorem foldl_append_shift {α β : Type} (xs : List α) (f : α → List β) (init : List β) :
    xs.foldl (fun acc x => acc ++ f x) init = init ++ xs.foldl (fun acc x => acc ++ f x) [] := by
  induction xs generalizing init with
  | nil => simp
  | cons h t ih =>
    simp only [List.foldl_cons]
    rw [ih (init ++ f h), ih ([] ++ f h)]
    simp [List.append_assoc]

/-- Indexing into a foldl-append at position `n*i + j` in the i-th segment (option version). -/
private theorem foldl_append_getElem? {α β : Type} (xs : List α) (f : α → List β)
    (n : Nat) (hsize : ∀ x ∈ xs, (f x).length = n)
    (i : Nat) (hi : i < xs.length) (j : Nat) (hj : j < n) :
    (xs.foldl (fun acc x => acc ++ f x) [])[n * i + j]? =
    (f xs[i])[j]? := by
  induction xs generalizing i with
  | nil => exact absurd hi (Nat.not_lt_of_le (Nat.zero_le _))
  | cons y ys ih =>
    have hy := hsize y (List.mem_cons.mpr (.inl rfl))
    have hsize' : ∀ x ∈ ys, (f x).length = n :=
      fun x hx => hsize x (List.mem_cons.mpr (.inr hx))
    simp only [List.foldl_cons, List.nil_append]
    rw [foldl_append_shift]
    match i with
    | 0 =>
      simp only [Nat.mul_zero, Nat.zero_add, List.getElem_cons_zero]
      rw [List.getElem?_append_left (by rw [hy]; exact hj)]
    | i + 1 =>
      simp only [List.getElem_cons_succ]
      have hi' : i < ys.length := by simp at hi; omega
      have hrw : n * (i + 1) + j = (f y).length + (n * i + j) := by
        rw [hy, Nat.mul_succ]; omega
      rw [hrw, List.getElem?_append_right (by omega)]
      have : (f y).length + (n * i + j) - (f y).length = n * i + j := by omega
      rw [this]
      exact ih hsize' i hi'

/-- Indexing into the flat list at `size ds * i + j` gives `(flatten xs[i])[j]` (option version). -/
theorem flatten_getElem_slice? {α : Type} {d : Nat} {ds : Shape}
    (xs : Tabulated (d :: ds) α) (i : Nat) (hi : i < d)
    (j : Nat) (hj : j < ds.volume) :
    (flatten xs)[ds.volume * i + j]? =
    (flatten (get xs ⟨i, hi⟩))[j]? := by
  show (xs.toList.foldl (fun acc x => acc ++ flatten (σ := ds) x) [])[ds.volume * i + j]? =
    (flatten (xs[i]))[j]?
  have hsize : ∀ x ∈ xs.toList, (flatten (σ := ds) x).length = ds.volume :=
    fun x _ => flatten_length x
  have hi' : i < xs.toList.length := by simp [Vector.length_toList]; exact hi
  have h := foldl_append_getElem? xs.toList (flatten (σ := ds)) ds.volume hsize i hi' j hj
  have hconv : xs.toList[i] = xs[i] := (Vector.getElem_toList ..).symm
  rw [hconv] at h
  exact h

/-- The flat list of a vector is the concatenation of the flats of its elements. -/
private theorem flatten_cons_eq {α : Type} {d : Nat} {ds : Shape}
    (first : Tabulated ds α) (more : Vector (Tabulated ds α) d) :
    flatten (σ := (d + 1) :: ds)
      (((Vector.singleton first).append more).cast (by omega)) =
    flatten first ++ flatten (σ := d :: ds) more := by
  -- Reduce to foldl over the toList
  suffices h : ∀ (v : Vector (Tabulated ds α) (d + 1)),
      v.toList = first :: more.toList →
      v.toList.foldl (fun acc x => acc ++ flatten (σ := ds) x) [] =
      flatten (σ := ds) first ++
        more.toList.foldl (fun acc x => acc ++ flatten (σ := ds) x) [] by
    exact h _ (by simp [Vector.append, Vector.singleton,
      Vector.toList, Array.toList_append])
  intro v hv
  rw [hv, List.foldl_cons, List.nil_append, foldl_append_shift]

/-- Unflattening and re-flattening recovers the original list. -/
theorem flatten_unflatten_eq {α : Type} : (σ : Shape) → (xs : List α) →
    (h : xs.length = σ.volume) →
    flatten (unflatten σ xs h) = xs
  | [], xs, h => by
    simp only [unflatten, flatten]
    have hlen : xs.length = 1 := by rw [h, Shape.volume_nil]
    match xs, hlen with
    | [v], _ => rfl
  | 0 :: _, xs, h => by
    simp only [unflatten, flatten]
    have : xs = [] := by
      cases xs with
      | nil => rfl
      | cons _ _ => simp [Shape.volume_cons] at h
    subst this; rfl
  | (d + 1) :: (ds : Shape), xs, h => by
    simp only [unflatten]
    rw [flatten_cons_eq]
    rw [flatten_unflatten_eq ds (xs.take ds.volume) _,
        flatten_unflatten_eq (d :: ds) (xs.drop ds.volume) _]
    exact List.take_append_drop ds.volume xs
termination_by σ => (σ.length, σ.headD 0)

end Tabulated

end Palladium.TensorLib
