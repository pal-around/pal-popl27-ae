import Util.Denote
import Util.Discrete
import Util.Macros
import Std

namespace Palladium.TensorLib

open Palladium.Util

/-!
DType: tags for the scalar element types of Jay tensors, and their denotations.
-/

/-- Runtime tag identifying the scalar element type of an array. -/
inductive DType where
  /-- Floating-point scalar. -/
  | float
  /-- Unbounded natural number scalar (used as output of discrete sums). -/
  | nat
  /-- Discrete (finite-support) scalar. -/
  | disc : Discrete → DType
deriving Repr, BEq, Inhabited, DecidableEq

/-- Boolean dtype abbreviation: `disc (.fin 2)`. -/
abbrev DType.bool : DType := .disc (.fin 2)

@[reducible] instance : Util.Denote DType where
  Sem := Type
  denote
    | .float => Float
    | .nat => Nat
    | .disc d => ⟦d⟧

/-- Promote two dtypes to their common supertype (disc < nat < float). -/
@[reducible] def DType.promote : DType → DType → DType
  | .float, _ => .float
  | _, .float => .float
  | .nat, _ => .nat
  | _, .nat => .nat
  | .disc (.fin m _), .disc (.fin n _) => .disc (.fin (max m n))


/-! # Castable -/

/-- Evidence that a cast from `src` to `tgt` is valid. -/
inductive DType.Castable : DType → DType → Type where
  /-- Any dtype can be cast to float. -/
  | toFloat : DType.Castable src .float
  /-- Natural number can be cast to natural number. -/
  | nat : DType.Castable .nat .nat
  /-- A discrete type can be cast to nat. -/
  | discToNat : DType.Castable (.disc d) .nat
  /-- A discrete type can be cast to a larger discrete type. -/
  | disc : (h : m ≤ n) → DType.Castable (.disc (.fin m hm)) (.disc (.fin n hn))

/-- Casting to the promoted type is valid for the left operand. -/
def DType.Castable.promote_left (η₁ η₂ : DType) : DType.Castable η₁ (η₁.promote η₂) := by
  cases η₁ with
  | float => exact .toFloat
  | nat =>
    cases η₂ with
    | float => exact .toFloat
    | nat => exact .nat
    | disc _ => exact .nat
  | disc d =>
    cases η₂ with
    | float => exact .toFloat
    | nat => exact .discToNat
    | disc d' =>
      cases d with | fin m hm =>
      cases d' with | fin n hn =>
      exact .disc (Nat.le_max_left m n)

/-- Casting to the promoted type is valid for the right operand. -/
def DType.Castable.promote_right (η₁ η₂ : DType) : DType.Castable η₂ (η₁.promote η₂) := by
  cases η₁ with
  | float => exact .toFloat
  | nat =>
    cases η₂ with
    | float => exact .toFloat
    | nat => exact .nat
    | disc _ => exact .discToNat
  | disc d =>
    cases η₂ with
    | float => exact .toFloat
    | nat => exact .nat
    | disc d' =>
      cases d with | fin m hm =>
      cases d' with | fin n hn =>
      exact .disc (Nat.le_max_right m n)

/-- Cast a value from one dtype to a target dtype (promotion), given a validity proof. -/
def DType.cast (src tgt : DType) (v : ⟦src⟧) (h : DType.Castable src tgt) : ⟦tgt⟧ :=
  match src, tgt, h with
  | .float, .float, .toFloat => v
  | .nat, .float, .toFloat => Float.ofNat v
  | .disc (.fin _ _), .float, .toFloat => Float.ofNat v.val
  | .nat, .nat, .nat => v
  | .disc (.fin _ _), .nat, .discToNat => v.val
  | .disc (.fin _ _), .disc (.fin _ _), .disc hle =>
      ⟨v.val, Nat.lt_of_lt_of_le v.isLt hle⟩

/-- Truthiness: convert any DType value to Bool (nonzero = true). -/
def DType.toBool (η : DType) (v : ⟦η⟧) : Bool :=
  match η with
  | .float => ↑v
  | .nat => v != 0
  | .disc (.fin _ _) => v.val != 0

/-- Pointwise addition for any DType. -/
def DType.add : (η : DType) → ⟦η⟧ → ⟦η⟧ → ⟦η⟧
  | .float => (· + ·)
  | .nat => Nat.add
  | .disc (.fin _ _) => (· + ·)

/-- Zero element for any DType. -/
def DType.zero : (η : DType) → ⟦η⟧
  | .float => 0
  | .nat => (0 : Nat)
  | .disc (.fin _ h) => ⟨0, h⟩

/-- Output dtype for axis reduction: disc sums become nat, others are preserved. -/
def DType.sumOut : DType → DType
  | .disc _ => .nat
  | η       => η

/-- Iterate `sumOut` `k` times (output dtype after summing out `k` axes). -/
def DType.sumOutMany (η : DType) : Nat → DType
  | 0 => η
  | k + 1 => DType.sumOut (DType.sumOutMany η k)

@[simp] theorem DType.sumOutMany_float (k : Nat) : DType.sumOutMany DType.float k = DType.float := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp [DType.sumOutMany]
    rw [ih]
    rfl

@[simp] theorem DType.sumOutMany_nat (k : Nat) : DType.sumOutMany DType.nat k = DType.nat := by
  induction k with
  | zero => rfl
  | succ k ih =>
    simp [DType.sumOutMany]
    rw [ih]
    rfl

theorem DType.sumOutMany_disc_succ_eq_nat {n : Nat} {h : 0 < n} (k : Nat) :
    DType.sumOutMany (DType.disc (.fin n h)) (k + 1) = DType.nat := by
  induction k with
  | zero => simp [DType.sumOutMany, DType.sumOut]
  | succ k ih =>
    calc
      DType.sumOutMany (DType.disc (.fin n h)) (k + 1 + 1)
          = DType.sumOut (DType.sumOutMany (DType.disc (.fin n h)) (k + 1)) := rfl
      _ = DType.sumOut DType.nat := by rw [ih]
      _ = DType.nat := rfl

theorem DType.sumOutMany_disc_fin_eq_nat {n : Nat} {h : 0 < n} {k : Nat} (hk : k ≠ 0) :
    DType.sumOutMany (DType.disc (.fin n h)) k = DType.nat := by
  obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero hk
  exact DType.sumOutMany_disc_succ_eq_nat k'

instance (η : DType) : BEq ⟦η⟧ := by
  cases η
  exact inferInstanceAs (BEq Float)
  exact inferInstanceAs (BEq Nat)
  next d => cases d with | fin n _ => exact inferInstanceAs (BEq (Fin n))

instance (η : DType) : Inhabited ⟦η⟧ := by
  cases η with
  | float => exact inferInstanceAs (Inhabited Float)
  | nat => exact inferInstanceAs (Inhabited Nat)
  | disc d => exact Util.DenotedInhabited.instInhabited d


end Palladium.TensorLib
