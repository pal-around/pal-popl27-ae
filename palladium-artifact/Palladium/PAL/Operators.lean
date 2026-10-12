import Palladium.PAL.Typing

namespace Palladium.PAL

open Palladium.Util

inductive UOp where
  | neg
  | not
deriving Repr

inductive BOp where
  | add
  | sub
  | mul
  | div
  | and
  | or
  | eq
  | ne
  | lt
  | le
deriving Repr

/-- Evidence that negation applied to input type `T` produces output type `Tₒ`. -/
inductive UOp.Neg.Valid : Ty → Ty → Type where
  /-- Negation of a float. -/
  | f : Neg.Valid .float .float
  /-- Negation of a nat. -/
  | n : Neg.Valid .nat .float
  /-- Negation of a discrete value. -/
  | d : Neg.Valid (.disc (.fin n hn)) .float

/-- Evidence that logical not applied to input type `T` produces output type `Tₒ`. -/
inductive UOp.Not.Valid : Ty → Ty → Type where
  /-- Logical not of a float. -/
  | f : Not.Valid .float Ty.bool
  /-- Logical not of a nat. -/
  | n : Not.Valid .nat Ty.bool
  /-- Logical not of a discrete value. -/
  | d : Not.Valid (.disc (.fin n hn)) Ty.bool

/-- Evidence that a unary operator applied to input type `T` produces output type `Tₒ`. -/
inductive UOp.Valid : UOp → Ty → Ty → Type where
  /-- Negation validity. -/
  | ofNeg : UOp.Neg.Valid T Tₒ → Valid .neg T Tₒ
  /-- Logical not validity. -/
  | ofNot : UOp.Not.Valid T Tₒ → Valid .not T Tₒ


/-- Negation: any input type, result is float. -/
@[reducible] def UOp.Valid.neg {T : Ty} : Valid .neg T .float :=
  .ofNeg (match T with | .float => .f | .nat => .n | .disc (.fin _ _) => .d)

/-- Logical not: any input type, result is bool. -/
@[reducible] def UOp.Valid.not {T : Ty} : Valid .not T Ty.bool :=
  .ofNot (match T with | .float => .f | .nat => .n | .disc (.fin _ _) => .d)

instance : Repr (UOp.Neg.Valid T Tₒ) where
  reprPrec _ _ := "⟨UOp.Valid⟩"

instance : Repr (UOp.Not.Valid T Tₒ) where
  reprPrec _ _ := "⟨UOp.Valid⟩"

instance : Repr (UOp.Valid op T Tₒ) where
  reprPrec _ _ := "⟨UOp.Valid⟩"

/-! # Binary operator validity evidence -/

/-- Evidence that addition applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Add.Valid : Ty → Ty → Ty → Type where
  /-- Addition: float × float → float. -/
  | ff : BOp.Add.Valid .float .float .float
  /-- Addition: float × nat → float. -/
  | fn : BOp.Add.Valid .float .nat .float
  /-- Addition: float × disc → float. -/
  | fd : BOp.Add.Valid .float (.disc (.fin n hn)) .float
  /-- Addition: nat × float → float. -/
  | nf : BOp.Add.Valid .nat .float .float
  /-- Addition: nat × nat → nat. -/
  | nn : BOp.Add.Valid .nat .nat .nat
  /-- Addition: nat × disc → nat. -/
  | nd : BOp.Add.Valid .nat (.disc (.fin n hn)) .nat
  /-- Addition: disc × float → float. -/
  | df : BOp.Add.Valid (.disc (.fin m hm)) .float .float
  /-- Addition: disc × nat → nat. -/
  | dn : BOp.Add.Valid (.disc (.fin m hm)) .nat .nat
  /-- Addition: disc × disc → nat (arithmetic, no modular wrapping). -/
  | dd : BOp.Add.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) .nat

/-- Evidence that subtraction applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Sub.Valid : Ty → Ty → Ty → Type where
  /-- Subtraction: float × float → float. -/
  | ff : BOp.Sub.Valid .float .float .float
  /-- Subtraction: float × nat → float. -/
  | fn : BOp.Sub.Valid .float .nat .float
  /-- Subtraction: float × disc → float. -/
  | fd : BOp.Sub.Valid .float (.disc (.fin n hn)) .float
  /-- Subtraction: nat × float → float. -/
  | nf : BOp.Sub.Valid .nat .float .float
  /-- Subtraction: nat × nat → nat. -/
  | nn : BOp.Sub.Valid .nat .nat .nat
  /-- Subtraction: nat × disc → nat. -/
  | nd : BOp.Sub.Valid .nat (.disc (.fin n hn)) .nat
  /-- Subtraction: disc × float → float. -/
  | df : BOp.Sub.Valid (.disc (.fin m hm)) .float .float
  /-- Subtraction: disc × nat → nat. -/
  | dn : BOp.Sub.Valid (.disc (.fin m hm)) .nat .nat
  /-- Subtraction: disc × disc → disc. -/
  | dd : BOp.Sub.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) (.disc (.fin (max m n)))

/-- Evidence that multiplication applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Mul.Valid : Ty → Ty → Ty → Type where
  /-- Multiplication: float × float → float. -/
  | ff : BOp.Mul.Valid .float .float .float
  /-- Multiplication: float × nat → float. -/
  | fn : BOp.Mul.Valid .float .nat .float
  /-- Multiplication: float × disc → float. -/
  | fd : BOp.Mul.Valid .float (.disc (.fin n hn)) .float
  /-- Multiplication: nat × float → float. -/
  | nf : BOp.Mul.Valid .nat .float .float
  /-- Multiplication: nat × nat → nat. -/
  | nn : BOp.Mul.Valid .nat .nat .nat
  /-- Multiplication: nat × disc → nat. -/
  | nd : BOp.Mul.Valid .nat (.disc (.fin n hn)) .nat
  /-- Multiplication: disc × float → float. -/
  | df : BOp.Mul.Valid (.disc (.fin m hm)) .float .float
  /-- Multiplication: disc × nat → nat. -/
  | dn : BOp.Mul.Valid (.disc (.fin m hm)) .nat .nat
  /-- Multiplication: disc × disc → disc. -/
  | dd : BOp.Mul.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) (.disc (.fin (m * n) (Nat.mul_pos hm hn)))

/-- Evidence that division applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Div.Valid : Ty → Ty → Ty → Type where
  /-- Division: float × float → float. -/
  | ff : BOp.Div.Valid .float .float .float
  /-- Division: float × nat → float. -/
  | fn : BOp.Div.Valid .float .nat .float
  /-- Division: float × disc → float. -/
  | fd : BOp.Div.Valid .float (.disc (.fin n hn)) .float
  /-- Division: nat × float → float. -/
  | nf : BOp.Div.Valid .nat .float .float
  /-- Division: nat × nat → float. -/
  | nn : BOp.Div.Valid .nat .nat .float
  /-- Division: nat × disc → float. -/
  | nd : BOp.Div.Valid .nat (.disc (.fin n hn)) .float
  /-- Division: disc × float → float. -/
  | df : BOp.Div.Valid (.disc (.fin m hm)) .float .float
  /-- Division: disc × nat → float. -/
  | dn : BOp.Div.Valid (.disc (.fin m hm)) .nat .float
  /-- Division: disc × disc → float. -/
  | dd : BOp.Div.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) .float

/-- Evidence that boolean and applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.And.Valid : Ty → Ty → Ty → Type where
  /-- Boolean and: float × float → bool. -/
  | ff : BOp.And.Valid .float .float Ty.bool
  /-- Boolean and: float × nat → bool. -/
  | fn : BOp.And.Valid .float .nat Ty.bool
  /-- Boolean and: float × disc → bool. -/
  | fd : BOp.And.Valid .float (.disc (.fin n hn)) Ty.bool
  /-- Boolean and: nat × float → bool. -/
  | nf : BOp.And.Valid .nat .float Ty.bool
  /-- Boolean and: nat × nat → bool. -/
  | nn : BOp.And.Valid .nat .nat Ty.bool
  /-- Boolean and: nat × disc → bool. -/
  | nd : BOp.And.Valid .nat (.disc (.fin n hn)) Ty.bool
  /-- Boolean and: disc × float → bool. -/
  | df : BOp.And.Valid (.disc (.fin m hm)) .float Ty.bool
  /-- Boolean and: disc × nat → bool. -/
  | dn : BOp.And.Valid (.disc (.fin m hm)) .nat Ty.bool
  /-- Boolean and: disc × disc → bool. -/
  | dd : BOp.And.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) Ty.bool

/-- Evidence that boolean or applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Or.Valid : Ty → Ty → Ty → Type where
  /-- Boolean or: float × float → bool. -/
  | ff : BOp.Or.Valid .float .float Ty.bool
  /-- Boolean or: float × nat → bool. -/
  | fn : BOp.Or.Valid .float .nat Ty.bool
  /-- Boolean or: float × disc → bool. -/
  | fd : BOp.Or.Valid .float (.disc (.fin n hn)) Ty.bool
  /-- Boolean or: nat × float → bool. -/
  | nf : BOp.Or.Valid .nat .float Ty.bool
  /-- Boolean or: nat × nat → bool. -/
  | nn : BOp.Or.Valid .nat .nat Ty.bool
  /-- Boolean or: nat × disc → bool. -/
  | nd : BOp.Or.Valid .nat (.disc (.fin n hn)) Ty.bool
  /-- Boolean or: disc × float → bool. -/
  | df : BOp.Or.Valid (.disc (.fin m hm)) .float Ty.bool
  /-- Boolean or: disc × nat → bool. -/
  | dn : BOp.Or.Valid (.disc (.fin m hm)) .nat Ty.bool
  /-- Boolean or: disc × disc → bool. -/
  | dd : BOp.Or.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) Ty.bool

/-- Evidence that equality applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Eq.Valid : Ty → Ty → Ty → Type where
  /-- Equality: float × float → bool. -/
  | ff : BOp.Eq.Valid .float .float Ty.bool
  /-- Equality: float × nat → bool. -/
  | fn : BOp.Eq.Valid .float .nat Ty.bool
  /-- Equality: float × disc → bool. -/
  | fd : BOp.Eq.Valid .float (.disc (.fin n hn)) Ty.bool
  /-- Equality: nat × float → bool. -/
  | nf : BOp.Eq.Valid .nat .float Ty.bool
  /-- Equality: nat × nat → bool. -/
  | nn : BOp.Eq.Valid .nat .nat Ty.bool
  /-- Equality: nat × disc → bool. -/
  | nd : BOp.Eq.Valid .nat (.disc (.fin n hn)) Ty.bool
  /-- Equality: disc × float → bool. -/
  | df : BOp.Eq.Valid (.disc (.fin m hm)) .float Ty.bool
  /-- Equality: disc × nat → bool. -/
  | dn : BOp.Eq.Valid (.disc (.fin m hm)) .nat Ty.bool
  /-- Equality: disc × disc → bool. -/
  | dd : BOp.Eq.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) Ty.bool

/-- Evidence that inequality applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Ne.Valid : Ty → Ty → Ty → Type where
  /-- Inequality: float × float → bool. -/
  | ff : BOp.Ne.Valid .float .float Ty.bool
  /-- Inequality: float × nat → bool. -/
  | fn : BOp.Ne.Valid .float .nat Ty.bool
  /-- Inequality: float × disc → bool. -/
  | fd : BOp.Ne.Valid .float (.disc (.fin n hn)) Ty.bool
  /-- Inequality: nat × float → bool. -/
  | nf : BOp.Ne.Valid .nat .float Ty.bool
  /-- Inequality: nat × nat → bool. -/
  | nn : BOp.Ne.Valid .nat .nat Ty.bool
  /-- Inequality: nat × disc → bool. -/
  | nd : BOp.Ne.Valid .nat (.disc (.fin n hn)) Ty.bool
  /-- Inequality: disc × float → bool. -/
  | df : BOp.Ne.Valid (.disc (.fin m hm)) .float Ty.bool
  /-- Inequality: disc × nat → bool. -/
  | dn : BOp.Ne.Valid (.disc (.fin m hm)) .nat Ty.bool
  /-- Inequality: disc × disc → bool. -/
  | dd : BOp.Ne.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) Ty.bool

/-- Evidence that less-than applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Lt.Valid : Ty → Ty → Ty → Type where
  /-- Less-than: float × float → bool. -/
  | ff : BOp.Lt.Valid .float .float Ty.bool
  /-- Less-than: float × nat → bool. -/
  | fn : BOp.Lt.Valid .float .nat Ty.bool
  /-- Less-than: float × disc → bool. -/
  | fd : BOp.Lt.Valid .float (.disc (.fin n hn)) Ty.bool
  /-- Less-than: nat × float → bool. -/
  | nf : BOp.Lt.Valid .nat .float Ty.bool
  /-- Less-than: nat × nat → bool. -/
  | nn : BOp.Lt.Valid .nat .nat Ty.bool
  /-- Less-than: nat × disc → bool. -/
  | nd : BOp.Lt.Valid .nat (.disc (.fin n hn)) Ty.bool
  /-- Less-than: disc × float → bool. -/
  | df : BOp.Lt.Valid (.disc (.fin m hm)) .float Ty.bool
  /-- Less-than: disc × nat → bool. -/
  | dn : BOp.Lt.Valid (.disc (.fin m hm)) .nat Ty.bool
  /-- Less-than: disc × disc → bool. -/
  | dd : BOp.Lt.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) Ty.bool

/-- Evidence that less-or-equal applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Le.Valid : Ty → Ty → Ty → Type where
  /-- Less-or-equal: float × float → bool. -/
  | ff : BOp.Le.Valid .float .float Ty.bool
  /-- Less-or-equal: float × nat → bool. -/
  | fn : BOp.Le.Valid .float .nat Ty.bool
  /-- Less-or-equal: float × disc → bool. -/
  | fd : BOp.Le.Valid .float (.disc (.fin n hn)) Ty.bool
  /-- Less-or-equal: nat × float → bool. -/
  | nf : BOp.Le.Valid .nat .float Ty.bool
  /-- Less-or-equal: nat × nat → bool. -/
  | nn : BOp.Le.Valid .nat .nat Ty.bool
  /-- Less-or-equal: nat × disc → bool. -/
  | nd : BOp.Le.Valid .nat (.disc (.fin n hn)) Ty.bool
  /-- Less-or-equal: disc × float → bool. -/
  | df : BOp.Le.Valid (.disc (.fin m hm)) .float Ty.bool
  /-- Less-or-equal: disc × nat → bool. -/
  | dn : BOp.Le.Valid (.disc (.fin m hm)) .nat Ty.bool
  /-- Less-or-equal: disc × disc → bool. -/
  | dd : BOp.Le.Valid (.disc (.fin m hm)) (.disc (.fin n hn)) Ty.bool

/-- Evidence that a binary operator applied to `T₁` and `T₂` produces `Tₒ`. -/
inductive BOp.Valid : BOp → Ty → Ty → Ty → Type where
  /-- Addition validity. -/
  | ofAdd : BOp.Add.Valid T₁ T₂ Tₒ → BOp.Valid .add T₁ T₂ Tₒ
  /-- Subtraction validity. -/
  | ofSub : BOp.Sub.Valid T₁ T₂ Tₒ → BOp.Valid .sub T₁ T₂ Tₒ
  /-- Multiplication validity. -/
  | ofMul : BOp.Mul.Valid T₁ T₂ Tₒ → BOp.Valid .mul T₁ T₂ Tₒ
  /-- Division validity. -/
  | ofDiv : BOp.Div.Valid T₁ T₂ Tₒ → BOp.Valid .div T₁ T₂ Tₒ
  /-- Boolean and validity. -/
  | ofAnd : BOp.And.Valid T₁ T₂ Tₒ → BOp.Valid .and T₁ T₂ Tₒ
  /-- Boolean or validity. -/
  | ofOr  : BOp.Or.Valid T₁ T₂ Tₒ → BOp.Valid .or T₁ T₂ Tₒ
  /-- Equality validity. -/
  | ofEq  : BOp.Eq.Valid T₁ T₂ Tₒ → BOp.Valid .eq T₁ T₂ Tₒ
  /-- Inequality validity. -/
  | ofNe  : BOp.Ne.Valid T₁ T₂ Tₒ → BOp.Valid .ne T₁ T₂ Tₒ
  /-- Less-than validity. -/
  | ofLt  : BOp.Lt.Valid T₁ T₂ Tₒ → BOp.Valid .lt T₁ T₂ Tₒ
  /-- Less-or-equal validity. -/
  | ofLe  : BOp.Le.Valid T₁ T₂ Tₒ → BOp.Valid .le T₁ T₂ Tₒ

/-- Addition: any types, result uses additive promotion. -/
@[reducible] def BOp.Valid.add {T₁ T₂ : Ty} :
  BOp.Valid .add T₁ T₂
    (match T₁, T₂ with
      | .float, _ => .float
      | _, .float => .float
      | .nat, _ | _, .nat => .nat
      | .disc (.fin _ _), .disc (.fin _ _) => .nat) :=
  .ofAdd (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Subtraction: any types, result is promoted. -/
@[reducible] def BOp.Valid.sub {T₁ T₂ : Ty} :
    BOp.Valid .sub T₁ T₂
      (match T₁, T₂ with
       | .float, _ => .float
       | _, .float => .float
       | .nat, _ | _, .nat => .nat
       | .disc (.fin m _), .disc (.fin n _) => .disc (.fin (max m n))) :=
  .ofSub (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Multiplication: any types, result uses multiplicative promotion. -/
@[reducible] def BOp.Valid.mul {T₁ T₂ : Ty} :
    BOp.Valid .mul T₁ T₂
      (match T₁, T₂ with
       | .float, _ => .float
       | _, .float => .float
       | .nat, _ | _, .nat => .nat
       | .disc (.fin m hm), .disc (.fin n hn) => .disc (.fin (m * n) (Nat.mul_pos hm hn))) :=
  .ofMul (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Division: any types, result is float. -/
@[reducible] def BOp.Valid.div {T₁ T₂ : Ty} : BOp.Valid .div T₁ T₂ .float :=
  .ofDiv (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Boolean and: any operand types, result is bool. -/
@[reducible] def BOp.Valid.and {T₁ T₂ : Ty} : BOp.Valid .and T₁ T₂ Ty.bool :=
  .ofAnd (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Boolean or: any operand types, result is bool. -/
@[reducible] def BOp.Valid.or {T₁ T₂ : Ty} : BOp.Valid .or T₁ T₂ Ty.bool :=
  .ofOr (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Equality: any operand types, result is bool. -/
@[reducible] def BOp.Valid.eq {T₁ T₂ : Ty} : BOp.Valid .eq T₁ T₂ Ty.bool :=
  .ofEq (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Inequality: any operand types, result is bool. -/
@[reducible] def BOp.Valid.ne {T₁ T₂ : Ty} : BOp.Valid .ne T₁ T₂ Ty.bool :=
  .ofNe (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Less-than: any operand types, result is bool. -/
@[reducible] def BOp.Valid.lt {T₁ T₂ : Ty} : BOp.Valid .lt T₁ T₂ Ty.bool :=
  .ofLt (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

/-- Less-or-equal: any operand types, result is bool. -/
@[reducible] def BOp.Valid.le {T₁ T₂ : Ty} : BOp.Valid .le T₁ T₂ Ty.bool :=
  .ofLe (match T₁, T₂ with
  | .float, .float => .ff
  | .float, .nat => .fn
  | .float, .disc (.fin _ _) => .fd
  | .nat, .float => .nf
  | .nat, .nat => .nn
  | .nat, .disc (.fin _ _) => .nd
  | .disc (.fin _ _), .float => .df
  | .disc (.fin _ _), .nat => .dn
  | .disc (.fin _ _), .disc (.fin _ _) => .dd)

instance : Repr (BOp.Add.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Sub.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Mul.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Div.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.And.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Or.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Eq.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Ne.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Lt.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Le.Valid T₁ T₂ Tₒ) where reprPrec _ _ := "⟨BOp.Valid⟩"
instance : Repr (BOp.Valid op T₁ T₂ Tₒ) where
  reprPrec _ _ := "⟨BOp.Valid⟩"

end Palladium.PAL
