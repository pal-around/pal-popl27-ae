import Palladium.PAL.Syntax
import Palladium.PAL.Denote
import TensorLib.Notations

namespace Palladium.PAL

open Palladium.Util
open Palladium.TensorLib

/-! ### Random context shape -/

/-- Compute the shape corresponding to a random context. Each discrete
    random variable contributes its support size as a dimension.
    Dimensions are ordered newest-first (matching the compiler's axis convention
    where the most recently sampled variable is at axis 0):
    `⟨x, .fin n⟩ :: rest` maps to `n :: RandomContext.toShape rest`. -/
def RandomContext.toShape : RandomContext → Shape
  | [] => []
  | ⟨_, .fin n _⟩ :: rest => n :: RandomContext.toShape rest

instance : Coe RandomContext Shape where
  coe := RandomContext.toShape

end Palladium.PAL

/-!
## From WellFormedShape to RandomContext

The inverse of `RandomContext.toShape`: given a well-formed shape and variable names
of matching length, construct the corresponding random context. Because every dimension
is positive by the `WellFormed` proof, no runtime guard is needed. The length constraint
`names.length = shape.length` eliminates dead branches and enables a strong
`keys = names` theorem.

Key equations:
- `WellFormedShape.toRandomContext ⟨[], _⟩ [] hlen = []`
- `WellFormedShape.toRandomContext ⟨n :: rest, wf⟩ (name :: names) hlen = ⟨name, .fin n h⟩ :: ...`
-/

namespace Palladium.TensorLib

open Palladium.Util
open Palladium.PAL

/-- Convert a well-formed shape to a random context given explicit variable names.
    Each dimension `n` paired with name `name` becomes `⟨name, .fin n h⟩`.
    Positivity of `n` is guaranteed by the `WellFormed` proof.
    Requires `names.length = wfs.shape.length` to ensure exact pairing. -/
def WellFormedShape.toRandomContext : (wfs : WellFormedShape) → (names : List String) →
    names.length = wfs.shape.length → RandomContext
  | ⟨[], _⟩, [], _ => []
  | ⟨n :: rest, wf⟩, name :: names, hlen =>
    let h := (Shape.WellFormed.cons_iff n rest).mp wf
    ⟨name, .fin n h.1⟩ :: toRandomContext ⟨rest, h.2⟩ names (by simp at hlen; omega)

/-- Coerce a well-formed shape to a random context using generated names `"_0"`, `"_1"`, etc.
    All dimensions are positive by `wf`, so `toRandomContext` keeps every dimension. -/
instance : Coe WellFormedShape RandomContext :=
  ⟨fun wfs => wfs.toRandomContext
      ((List.range wfs.shape.length).map (fun i => s!"_{i}"))
      (by simp)⟩

/-- Keys of `wfs.toRandomContext names hlen` equal `names`. -/
theorem WellFormedShape.toRandomContext_keys (wfs : WellFormedShape) (names : List String)
    (hlen : names.length = wfs.shape.length) :
    (wfs.toRandomContext names hlen).keys = names := by
  obtain ⟨σ, hwf⟩ := wfs
  induction σ generalizing names with
  | nil =>
    have : names = [] := match names, hlen with | [], _ => rfl
    subst this; rfl
  | cons n rest ih =>
    match names, hlen with
    | name :: names', hlen =>
      have hcons := (Shape.WellFormed.cons_iff n rest).mp hwf
      simp only [WellFormedShape.toRandomContext, List.keys_cons]
      exact congrArg (name :: ·) (ih names' hcons.2 (by simp at hlen; omega))

/-- Bundle `toRandomContext` with a `NodupKeys` proof to form a `WellFormedContext`. -/
def WellFormedShape.toWellFormedRandomContext (wfs : WellFormedShape)
    (names : List String) (hlen : names.length = wfs.shape.length) (hnd : names.Nodup) :
    WellFormedContext Discrete :=
  { ctx := wfs.toRandomContext names hlen
    wf  := by show (wfs.toRandomContext names hlen).keys.Nodup
              rw [wfs.toRandomContext_keys names hlen]; exact hnd }

end Palladium.TensorLib

namespace Palladium.PAL

open Palladium.Util
open Palladium.TensorLib

/-!
## Shape of a random context

The shape of a tensor representing a PMF over a random context is determined
by the support sizes of each discrete random variable.

Dimensions are ordered newest-first: the most recently sampled variable
corresponds to axis 0. Sampling `x⃗` prepends `𝒞⟦x⃗⟧` to the shape of the probability
tensor (`RandomContext.toShape_append`).

Example: after `x ← sample (x : fin 3); y ← sample (y : fin 5)`:
- Context: `[⟨y, fin 5⟩, ⟨x, fin 3⟩]`
- `keys = [y, x]`
- `ctx.toShape = [5, 3]` (y at axis 0, x at axis 1)

Key equations:
- `toShape(∅) = []`
- `toShape(x:fin(n) · Σ) = n :: toShape(Σ)`
- `toShape(Σ)[i] = n  ⟺  keys(Σ)[i] = x ∧ Σ(x) = fin(n)`
-/

/-- The length of `Δ.toShape` equals the number of keys in the context. -/
theorem RandomContext.toShape_length :
    (Δ : RandomContext) → Δ.toShape.rank = Δ.keys.length
  | [] => rfl
  | ⟨_x, .fin _n _⟩ :: rest => by
    simp only [RandomContext.toShape, List.keys, List.map, List.length_cons]
    exact congrArg (· + 1) (RandomContext.toShape_length rest)

/-- toShape(pre ++ Σ) = toShape(pre) ++ toShape(Σ). -/
theorem RandomContext.toShape_append :
    (pre ctx : RandomContext) → (pre ++ ctx).toShape = pre.toShape ++ ctx.toShape
  | [], _ => rfl
  | ⟨_x, .fin _n _⟩ :: rest, ctx => congrArg (_ :: ·) (RandomContext.toShape_append rest ctx)

/-- keys(Σ)[i] = x ∧ Σ(x) = fin(n) ∧ Nodup → `Δ.toShape[idx]? = some n`. -/
theorem RandomContext.toShape_getElem?
    (Δ : RandomContext) (name : String)
    (idx : Nat) (hIdxLt : idx < Δ.keys.length)
    (hKeysIdx : Δ.keys[idx]'hIdxLt = name)
    (n : Nat) (hn : 0 < n) (hmem : ⟨name, .fin n hn⟩ ∈ Δ)
    (hnd : Δ.keys.Nodup) :
    Δ.toShape[idx]? = some n := by
  induction Δ generalizing idx name n with
  | nil => exact absurd hIdxLt (by simp [List.keys])
  | cons s rest ih =>
    obtain ⟨y, D⟩ := s
    cases D with
    | fin m hm =>
      by_cases hname : y = name
      · -- Head case: y = name, idx = 0 by Nodup
        subst hname
        have hmn : m = n := by
          cases hmem with
          | head => rfl
          | tail _ h' =>
            exfalso
            exact (List.nodup_cons.mp hnd).1 (List.mem_keys.mpr ⟨_, h'⟩)
        subst hmn
        have hidx0 : idx = 0 := by
          match idx, hIdxLt, hKeysIdx with
          | 0, _, _ => rfl
          | k + 1, hlt, hkidx =>
            exfalso
            have hk : k < rest.keys.length := by
              simp [List.keys_cons] at hlt; omega
            have : rest.keys[k]'hk = y := by
              simp [List.keys_cons] at hkidx
              exact hkidx
            exact (List.nodup_cons.mp hnd).1 (this ▸ List.getElem_mem hk)
        subst hidx0
        simp [RandomContext.toShape]
      · -- Tail case: y ≠ name, idx > 0
        have hidx_pos : idx ≠ 0 := by
          intro heq; subst heq; simp [List.keys_cons] at hKeysIdx; exact hname hKeysIdx
        obtain ⟨k, rfl⟩ : ∃ k, idx = k + 1 := ⟨idx - 1, by omega⟩
        have hk : k < rest.keys.length := by
          simp [List.keys_cons] at hIdxLt; omega
        have hKeysIdx' : rest.keys[k]'hk = name := by
          have := hKeysIdx; simp [List.keys_cons] at this
          exact this
        have hnd_rest : rest.keys.Nodup := (List.nodup_cons.mp hnd).2
        have hmem_rest : ⟨name, .fin n hn⟩ ∈ rest := by
          cases hmem with
          | head => exact absurd rfl hname
          | tail _ h' => exact h'
        simp only [RandomContext.toShape, List.getElem?_cons_succ]
        exact ih (name := name) (idx := k) (n := n) (hn := hn) hk hKeysIdx' hmem_rest hnd_rest

/-- keys(Σ)[i] = x ∧ Σ(x) = fin(n) ∧ Nodup → `Δ.toShape[idx] = n`. -/
theorem RandomContext.toShape_getElem
    (Δ : RandomContext) (name : String)
    (idx : Nat) (hIdxLt : idx < Δ.keys.length)
    (hKeysIdx : Δ.keys[idx]'hIdxLt = name)
    (n : Nat) (hn : 0 < n) (hmem : ⟨name, .fin n hn⟩ ∈ Δ)
    (hnd : Δ.keys.Nodup) :
    have hAxLt : idx < Δ.toShape.rank := by
      rw [Δ.toShape_length]; exact hIdxLt
    Δ.toShape[idx]'hAxLt = n := by
  have h := Δ.toShape_getElem? name idx hIdxLt hKeysIdx n hn hmem hnd
  exact (List.getElem?_eq_some_iff.mp h).2

/-- All dimensions of `Δ.toShape` are positive. -/
theorem RandomContext.toShape_wellFormed (Δ : RandomContext) :
    Shape.WellFormed (RandomContext.toShape Δ) := by
  intro i hi
  induction Δ generalizing i with
  | nil => exact absurd hi (by simp [RandomContext.toShape])
  | cons s rest ih =>
    obtain ⟨xname, D⟩ := s
    cases D with
    | fin n hn =>
      simp only [RandomContext.toShape, List.length_cons] at hi
      match i with
      | 0 => simpa [RandomContext.toShape]
      | i + 1 =>
        simp only [RandomContext.toShape, List.getElem_cons_succ]
        exact ih i (by simp at hi; omega)

/-- Coerce a random context to a well-formed shape via `toShape` and `toShape_wellFormed`. -/
instance : Coe RandomContext WellFormedShape :=
  ⟨fun Δ => ⟨Δ.toShape, Δ.toShape_wellFormed⟩⟩

/-- The name list `["#", "##", ..., "#"^n]` has no duplicates, because
    each string has a distinct length `i + 1`. -/
private theorem generatedNames_nodup (n : Nat) :
    ((List.range n).map (fun i : Nat => String.ofList (List.replicate (i + 1) '#'))).Nodup := by
  apply List.Nodup.map _ List.nodup_range
  intro i j h
  have h1 : List.replicate (i + 1) '#' = List.replicate (j + 1) '#' :=
    String.ofList_injective h
  have h2 := congrArg List.length h1
  simp [List.length_replicate] at h2
  omega

/-- Coerce a well-formed random context to a well-formed shape via `toShape`. -/
instance : Coe (WellFormedContext Discrete) WellFormedShape :=
  ⟨fun wfc => ⟨RandomContext.toShape wfc, RandomContext.toShape_wellFormed wfc⟩⟩

/-- Coerce a well-formed shape to a well-formed random context using generated names.
    The `NodupKeys` proof follows from `generatedNames_nodup`. -/
instance : Coe WellFormedShape (WellFormedContext Discrete) :=
  ⟨fun wfs =>
    let names := (List.range wfs.shape.length).map
      (fun i : Nat => String.ofList (List.replicate (i + 1) '#'))
    wfs.toWellFormedRandomContext names
      (by show ((List.range wfs.shape.length).map _).length = wfs.shape.length; simp)
      (generatedNames_nodup _)⟩

/-- Erasing the axis for variable `name` from the shape equals the shape of the erased context. -/
theorem RandomContext.toShape_delete
    (Δ : RandomContext) (name : String)
    (idx : Nat) (hIdxLt : idx < Δ.keys.length)
    (hKeysIdx : Δ.keys[idx]'hIdxLt = name)
    (hnd : Δ.keys.Nodup) :
    Δ.toShape.slice ⟨idx, Δ.toShape_length ▸ hIdxLt⟩ =
      RandomContext.toShape (List.kerase name Δ) := by
  induction Δ generalizing idx name with
  | nil => exact absurd hIdxLt (by simp [List.keys])
  | cons s rest ih =>
    obtain ⟨y, D⟩ := s
    cases D with
    | fin m hm =>
      by_cases hname : y = name
      · subst hname
        have hidx0 : idx = 0 := by
          match idx, hIdxLt, hKeysIdx with
          | 0, _, _ => rfl
          | k + 1, hlt, hkidx =>
            exfalso
            have hk : k < rest.keys.length := by
              simp only [List.keys, List.length_map, List.length_cons] at hlt ⊢
              omega
            have : rest.keys[k]'hk = y := by
              simp only [List.keys, List.getElem_map] at hkidx ⊢
              simpa [List.getElem_cons_succ] using hkidx
            exact (List.nodup_cons.mp hnd).1 (this ▸ List.getElem_mem hk)
        subst hidx0
        have hdel : List.kerase y (⟨y, .fin m hm⟩ :: rest) = rest := by
          simp [List.kerase_cons_eq]
        simp only [RandomContext.toShape, hdel]; rfl
      · have hidx_pos : idx ≠ 0 := by
          intro heq; subst heq; simp [List.keys] at hKeysIdx; exact hname hKeysIdx
        obtain ⟨k, rfl⟩ : ∃ k, idx = k + 1 := ⟨idx - 1, by omega⟩
        have hk : k < rest.keys.length := by
          simp only [List.keys, List.length_map, List.length_cons] at hIdxLt ⊢; omega
        have hKeysIdx' : rest.keys[k]'hk = name := by
          have := hKeysIdx; simp [List.keys, List.getElem_map] at this ⊢
          simpa [List.getElem_cons_succ] using this
        have hnd_rest : rest.keys.Nodup := (List.nodup_cons.mp hnd).2
        simp only [RandomContext.toShape, List.eraseIdx_cons_succ]
        have hdel : List.kerase name (⟨y, .fin m hm⟩ :: rest) =
            ⟨y, .fin m hm⟩ :: (List.kerase name rest) := by
          rw [List.kerase_cons_ne (h := fun h => hname h.symm)]
        rw [hdel, show RandomContext.toShape (⟨y, .fin m hm⟩ :: (List.kerase name rest)) =
            m :: RandomContext.toShape (List.kerase name rest) from rfl]
        exact congrArg (m :: ·) (ih name k hk hKeysIdx' hnd_rest)

/-- If two random contexts have the same types at each position, they have the same shape. -/
theorem RandomContext.toShape_eq_of_map_snd_eq {ctx₁ ctx₂ : RandomContext}
    (h : ctx₁.map Sigma.snd = ctx₂.map Sigma.snd) :
    ctx₁.toShape = ctx₂.toShape := by
  induction ctx₁ generalizing ctx₂ with
  | nil =>
    have : ctx₂ = [] := by
      cases ctx₂ with
      | nil => rfl
      | cons _ _ => simp at h
    subst this; rfl
  | cons hd tl ih =>
    obtain ⟨x₁, D₁⟩ := hd
    cases D₁ with
    | fin n₁ hn₁ =>
      cases ctx₂ with
      | nil => simp at h
      | cons hd₂ rest₂ =>
        obtain ⟨x₂, D₂⟩ := hd₂
        cases D₂ with
        | fin n₂ hn₂ =>
          have h_tl : tl.map Sigma.snd = rest₂.map Sigma.snd := by
            have := congrArg List.tail h; simp at this; exact this
          simp only [RandomContext.toShape]
          have heqn : n₁ = n₂ := by
            have := congrArg (fun l => l.head?) h; simp at this; exact this
          exact congrArg₂ (· :: ·) heqn (ih h_tl)

/-! ### WellFormedRandomContext forwarding -/

/-- Compute the shape corresponding to a well-formed random context. -/
abbrev WellFormedRandomContext.toShape (Δ : WellFormedRandomContext) : Shape :=
  RandomContext.toShape ↑Δ

/-- The length of `Δ.toShape` equals the number of keys in the context. -/
theorem WellFormedRandomContext.toShape_length (Δ : WellFormedRandomContext) :
    Δ.toShape.rank = Δ.keys.length :=
  RandomContext.toShape_length ↑Δ

/-- Erasing the axis for variable `name` from the shape equals the shape of the erased context. -/
theorem WellFormedRandomContext.toShape_delete (Δ : WellFormedRandomContext)
    (name : String) (idx : Nat)
    (hIdxLt : idx < Δ.keys.length) (hKeysIdx : Δ.keys[idx]'hIdxLt = name) :
    Δ.toShape.slice ⟨idx, Δ.toShape_length ▸ hIdxLt⟩ =
      RandomContext.toShape (List.kerase name ↑Δ) :=
  RandomContext.toShape_delete ↑Δ name idx hIdxLt hKeysIdx Δ.wf

/-- keys(Σ)[i] = x ∧ Σ(x) = fin(n) ∧ WellFormed → `Δ.toShape[idx]? = some n`. -/
theorem WellFormedRandomContext.toShape_getElem? (Δ : WellFormedRandomContext)
    (name : String) (idx : Nat)
    (hIdxLt : idx < Δ.keys.length) (hKeysIdx : Δ.keys[idx]'hIdxLt = name)
    (n : Nat) (hn : 0 < n) (hmem : ⟨name, .fin n hn⟩ ∈ Δ) :
    Δ.toShape[idx]? = some n :=
  RandomContext.toShape_getElem? ↑Δ name idx hIdxLt hKeysIdx n hn hmem Δ.wf

end Palladium.PAL
