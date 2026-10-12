import Util.Denote

namespace Palladium.Util

/-!
# Discrete distributions

Shared type descriptor for discrete (finite-support) distributions.
Used by both PAL types (`PAL.Ty`) and tensor element types (`Tensor.DType`).
-/

/-- Descriptor for a discrete distribution with finite support.
    Requires `0 < n` to ensure the support type `Fin n` is inhabited. -/
inductive Discrete where
  /-- Finite support of size `n`, with proof that `n` is positive. -/
  | fin (n : Nat) (pos : 0 < n := by omega) : Discrete

instance : Repr Discrete where
  reprPrec d p := match d with
    | .fin n _ => reprPrec n p

instance : BEq Discrete where
  beq a b := match a, b with
    | .fin m _, .fin n _ => m == n

instance : DecidableEq Discrete
  | .fin m _, .fin n _ =>
    if h : m = n then
      .isTrue (by subst h; rfl)
    else
      .isFalse (by intro heq; cases heq; exact h rfl)

instance : Inhabited Discrete where
  default := .fin 1 (by omega)

/-- Denotation of a discrete descriptor as a Lean type. -/
@[reducible] def Discrete.denote : Discrete → Type
  | .fin n _ => Fin n

/-- Discrete distribution descriptors denote their support type. -/
@[reducible] instance : Denote Discrete where
  Sem := Type; denote := Discrete.denote

/-- Inhabited instance for discrete denotations.
    Since `0 < n` is guaranteed, `⟨0, pos⟩` is always a valid `Fin n`. -/
instance : DenotedInhabited Discrete Discrete.denote where
  instInhabited d :=
    match d with
    | .fin _ pos => ⟨⟨0, pos⟩⟩

/-- Direct `Inhabited` instance for `Discrete.denote` (used by `Environment`). -/
instance (d : Discrete) : Inhabited (Discrete.denote d) :=
  DenotedInhabited.instInhabited d

/-- The size (cardinality) of the support of a discrete distribution. -/
def Discrete.size : Discrete → Nat
  | .fin n _ => n

/-- Enumerate all elements in the support of a discrete distribution. -/
def Discrete.support : (d : Discrete) → List (Discrete.denote d)
  | .fin n _ => List.ofFn (fun i : Fin n => i)

instance : ToString Discrete where
  toString
    | .fin n _ => s!"Fin({n})"

/-- Coerce a Bool to Fin 2: false ↦ 0, true ↦ 1. -/
instance : Coe Bool (Fin 2) where
  coe b := ⟨b.toNat % 2, Nat.mod_lt _ (by omega)⟩

/-- Coerce a Fin 2 to Bool: 0 ↦ false, nonzero ↦ true. -/
instance : Coe (Fin 2) Bool where
  coe v := v.val != 0

/-- Coerce a Float to Bool via truthiness: 0.0 ↦ false, nonzero ↦ true. -/
instance : Coe Float Bool where
  coe f := f != 0.0

/-- Coerce a Bool to Float: false ↦ 0.0, true ↦ 1.0. -/
instance : Coe Bool Float where
  coe b := if b then 1.0 else 0.0

end Palladium.Util
