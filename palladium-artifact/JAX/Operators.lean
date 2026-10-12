import JAX.Typing

namespace Palladium.JAX

open Palladium.TensorLib

/- Unary operators. -/
inductive UOp where
  /-- Numeric negation (`-x`). -/
  | neg
  /-- Boolean negation (`not x`). -/
  | not
deriving Repr

/- Binary operators with JAX-style broadcasting. -/
inductive BOp where
  /-- Numeric addition (`x + y`). -/
  | add
  /-- Numeric subtraction (`x - y`). -/
  | sub
  /-- Numeric multiplication (`x * y`). -/
  | mul
  /-- Numeric division (`x / y`). -/
  | div
  /-- Boolean conjunction (`x and y`). -/
  | and
  /-- Boolean disjunction (`x or y`). -/
  | or
  /-- Equality comparison (`x == y`). -/
  | eq
  /-- Inequality comparison (`x != y`). -/
  | ne
  /-- Less-than comparison (`x < y`). -/
  | lt
  /-- Less-than-or-equal comparison (`x ≤ y`). -/
  | le
  /-- Greater-than comparison (`x > y`). -/
  | gt
  /-- Greater-than-or-equal comparison (`x ≥ y`). -/
  | ge
deriving Repr

/-- Evidence that a unary operator applied to input dtype `η` produces output dtype `ηₒ`. -/
inductive UOp.Valid : UOp → DType → DType → Type where
  /-- Negation of a float. -/
  | neg_f : Valid .neg .float .float
  /-- Negation of a nat. -/
  | neg_n : Valid .neg .nat .float
  /-- Negation of a discrete value. -/
  | neg_d : Valid .neg (.disc (.fin n hn)) .float
  /-- Logical not of a float. -/
  | not_f : Valid .not .float DType.bool
  /-- Logical not of a nat. -/
  | not_n : Valid .not .nat DType.bool
  /-- Logical not of a discrete value. -/
  | not_d : Valid .not (.disc (.fin n hn)) DType.bool

/-- Negation: any input type, result is float. -/
@[reducible] def UOp.Valid.neg {η : DType} : Valid .neg η .float :=
  match η with | .float => .neg_f | .nat => .neg_n | .disc (.fin _ _) => .neg_d

/-- Logical not: any input type, result is bool. -/
@[reducible] def UOp.Valid.not {η : DType} : Valid .not η DType.bool :=
  match η with | .float => .not_f | .nat => .not_n | .disc (.fin _ _) => .not_d

instance : Repr (UOp.Valid op η ηₒ) where
  reprPrec _ _ := "⟨UOp.Valid⟩"

/-! # Binary operator validity evidence -/

/-- Evidence that a binary operator applied to `η₁` and `η₂` produces `ηₒ`. -/
inductive BOp.Valid : BOp → DType → DType → DType → Type where
  /-- Addition: float × float → float. -/
  | add_ff : BOp.Valid .add .float .float .float
  /-- Addition: float × nat → float. -/
  | add_fn : BOp.Valid .add .float .nat .float
  /-- Addition: float × disc → float. -/
  | add_fd : BOp.Valid .add .float (.disc (.fin n hn)) .float
  /-- Addition: nat × float → float. -/
  | add_nf : BOp.Valid .add .nat .float .float
  /-- Addition: nat × nat → nat. -/
  | add_nn : BOp.Valid .add .nat .nat .nat
  /-- Addition: nat × disc → nat. -/
  | add_nd : BOp.Valid .add .nat (.disc (.fin n hn)) .nat
  /-- Addition: disc × float → float. -/
  | add_df : BOp.Valid .add (.disc (.fin m hm)) .float .float
  /-- Addition: disc × nat → nat. -/
  | add_dn : BOp.Valid .add (.disc (.fin m hm)) .nat .nat
  /-- Addition: disc × disc → nat (arithmetic, no modular wrapping). -/
  | add_dd : BOp.Valid .add (.disc (.fin m hm)) (.disc (.fin n hn)) .nat
  /-- Subtraction: float × float → float. -/
  | sub_ff : BOp.Valid .sub .float .float .float
  /-- Subtraction: float × nat → float. -/
  | sub_fn : BOp.Valid .sub .float .nat .float
  /-- Subtraction: float × disc → float. -/
  | sub_fd : BOp.Valid .sub .float (.disc (.fin n hn)) .float
  /-- Subtraction: nat × float → float. -/
  | sub_nf : BOp.Valid .sub .nat .float .float
  /-- Subtraction: nat × nat → nat. -/
  | sub_nn : BOp.Valid .sub .nat .nat .nat
  /-- Subtraction: nat × disc → nat. -/
  | sub_nd : BOp.Valid .sub .nat (.disc (.fin n hn)) .nat
  /-- Subtraction: disc × float → float. -/
  | sub_df : BOp.Valid .sub (.disc (.fin m hm)) .float .float
  /-- Subtraction: disc × nat → nat. -/
  | sub_dn : BOp.Valid .sub (.disc (.fin m hm)) .nat .nat
  /-- Subtraction: disc × disc → disc. -/
  | sub_dd : BOp.Valid .sub (.disc (.fin m hm)) (.disc (.fin n hn)) (.disc (.fin (max m n)))
  /-- Multiplication: float × float → float. -/
  | mul_ff : BOp.Valid .mul .float .float .float
  /-- Multiplication: float × nat → float. -/
  | mul_fn : BOp.Valid .mul .float .nat .float
  /-- Multiplication: float × disc → float. -/
  | mul_fd : BOp.Valid .mul .float (.disc (.fin n hn)) .float
  /-- Multiplication: nat × float → float. -/
  | mul_nf : BOp.Valid .mul .nat .float .float
  /-- Multiplication: nat × nat → nat. -/
  | mul_nn : BOp.Valid .mul .nat .nat .nat
  /-- Multiplication: nat × disc → nat. -/
  | mul_nd : BOp.Valid .mul .nat (.disc (.fin n hn)) .nat
  /-- Multiplication: disc × float → float. -/
  | mul_df : BOp.Valid .mul (.disc (.fin m hm)) .float .float
  /-- Multiplication: disc × nat → nat. -/
  | mul_dn : BOp.Valid .mul (.disc (.fin m hm)) .nat .nat
  /-- Multiplication: disc × disc → disc. -/
  | mul_dd : BOp.Valid .mul (.disc (.fin m hm)) (.disc (.fin n hn)) (.disc (.fin (m * n) (Nat.mul_pos hm hn)))
  /-- Division: float × float → float. -/
  | div_ff : BOp.Valid .div .float .float .float
  /-- Division: float × nat → float. -/
  | div_fn : BOp.Valid .div .float .nat .float
  /-- Division: float × disc → float. -/
  | div_fd : BOp.Valid .div .float (.disc (.fin n hn)) .float
  /-- Division: nat × float → float. -/
  | div_nf : BOp.Valid .div .nat .float .float
  /-- Division: nat × nat → float. -/
  | div_nn : BOp.Valid .div .nat .nat .float
  /-- Division: nat × disc → float. -/
  | div_nd : BOp.Valid .div .nat (.disc (.fin n hn)) .float
  /-- Division: disc × float → float. -/
  | div_df : BOp.Valid .div (.disc (.fin m hm)) .float .float
  /-- Division: disc × nat → float. -/
  | div_dn : BOp.Valid .div (.disc (.fin m hm)) .nat .float
  /-- Division: disc × disc → float. -/
  | div_dd : BOp.Valid .div (.disc (.fin m hm)) (.disc (.fin n hn)) .float
  /-- Boolean and: float × float → bool. -/
  | and_ff : BOp.Valid .and .float .float DType.bool
  /-- Boolean and: float × nat → bool. -/
  | and_fn : BOp.Valid .and .float .nat DType.bool
  /-- Boolean and: float × disc → bool. -/
  | and_fd : BOp.Valid .and .float (.disc (.fin n hn)) DType.bool
  /-- Boolean and: nat × float → bool. -/
  | and_nf : BOp.Valid .and .nat .float DType.bool
  /-- Boolean and: nat × nat → bool. -/
  | and_nn : BOp.Valid .and .nat .nat DType.bool
  /-- Boolean and: nat × disc → bool. -/
  | and_nd : BOp.Valid .and .nat (.disc (.fin n hn)) DType.bool
  /-- Boolean and: disc × float → bool. -/
  | and_df : BOp.Valid .and (.disc (.fin m hm)) .float DType.bool
  /-- Boolean and: disc × nat → bool. -/
  | and_dn : BOp.Valid .and (.disc (.fin m hm)) .nat DType.bool
  /-- Boolean and: disc × disc → bool. -/
  | and_dd : BOp.Valid .and (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool
  /-- Boolean or: float × float → bool. -/
  | or_ff : BOp.Valid .or .float .float DType.bool
  /-- Boolean or: float × nat → bool. -/
  | or_fn : BOp.Valid .or .float .nat DType.bool
  /-- Boolean or: float × disc → bool. -/
  | or_fd : BOp.Valid .or .float (.disc (.fin n hn)) DType.bool
  /-- Boolean or: nat × float → bool. -/
  | or_nf : BOp.Valid .or .nat .float DType.bool
  /-- Boolean or: nat × nat → bool. -/
  | or_nn : BOp.Valid .or .nat .nat DType.bool
  /-- Boolean or: nat × disc → bool. -/
  | or_nd : BOp.Valid .or .nat (.disc (.fin n hn)) DType.bool
  /-- Boolean or: disc × float → bool. -/
  | or_df : BOp.Valid .or (.disc (.fin m hm)) .float DType.bool
  /-- Boolean or: disc × nat → bool. -/
  | or_dn : BOp.Valid .or (.disc (.fin m hm)) .nat DType.bool
  /-- Boolean or: disc × disc → bool. -/
  | or_dd : BOp.Valid .or (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool
  /-- Equality: float × float → bool. -/
  | eq_ff : BOp.Valid .eq .float .float DType.bool
  /-- Equality: float × nat → bool. -/
  | eq_fn : BOp.Valid .eq .float .nat DType.bool
  /-- Equality: float × disc → bool. -/
  | eq_fd : BOp.Valid .eq .float (.disc (.fin n hn)) DType.bool
  /-- Equality: nat × float → bool. -/
  | eq_nf : BOp.Valid .eq .nat .float DType.bool
  /-- Equality: nat × nat → bool. -/
  | eq_nn : BOp.Valid .eq .nat .nat DType.bool
  /-- Equality: nat × disc → bool. -/
  | eq_nd : BOp.Valid .eq .nat (.disc (.fin n hn)) DType.bool
  /-- Equality: disc × float → bool. -/
  | eq_df : BOp.Valid .eq (.disc (.fin m hm)) .float DType.bool
  /-- Equality: disc × nat → bool. -/
  | eq_dn : BOp.Valid .eq (.disc (.fin m hm)) .nat DType.bool
  /-- Equality: disc × disc → bool. -/
  | eq_dd : BOp.Valid .eq (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool
  /-- Inequality: float × float → bool. -/
  | ne_ff : BOp.Valid .ne .float .float DType.bool
  /-- Inequality: float × nat → bool. -/
  | ne_fn : BOp.Valid .ne .float .nat DType.bool
  /-- Inequality: float × disc → bool. -/
  | ne_fd : BOp.Valid .ne .float (.disc (.fin n hn)) DType.bool
  /-- Inequality: nat × float → bool. -/
  | ne_nf : BOp.Valid .ne .nat .float DType.bool
  /-- Inequality: nat × nat → bool. -/
  | ne_nn : BOp.Valid .ne .nat .nat DType.bool
  /-- Inequality: nat × disc → bool. -/
  | ne_nd : BOp.Valid .ne .nat (.disc (.fin n hn)) DType.bool
  /-- Inequality: disc × float → bool. -/
  | ne_df : BOp.Valid .ne (.disc (.fin m hm)) .float DType.bool
  /-- Inequality: disc × nat → bool. -/
  | ne_dn : BOp.Valid .ne (.disc (.fin m hm)) .nat DType.bool
  /-- Inequality: disc × disc → bool. -/
  | ne_dd : BOp.Valid .ne (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool
  /-- Less-than: float × float → bool. -/
  | lt_ff : BOp.Valid .lt .float .float DType.bool
  /-- Less-than: float × nat → bool. -/
  | lt_fn : BOp.Valid .lt .float .nat DType.bool
  /-- Less-than: float × disc → bool. -/
  | lt_fd : BOp.Valid .lt .float (.disc (.fin n hn)) DType.bool
  /-- Less-than: nat × float → bool. -/
  | lt_nf : BOp.Valid .lt .nat .float DType.bool
  /-- Less-than: nat × nat → bool. -/
  | lt_nn : BOp.Valid .lt .nat .nat DType.bool
  /-- Less-than: nat × disc → bool. -/
  | lt_nd : BOp.Valid .lt .nat (.disc (.fin n hn)) DType.bool
  /-- Less-than: disc × float → bool. -/
  | lt_df : BOp.Valid .lt (.disc (.fin m hm)) .float DType.bool
  /-- Less-than: disc × nat → bool. -/
  | lt_dn : BOp.Valid .lt (.disc (.fin m hm)) .nat DType.bool
  /-- Less-than: disc × disc → bool. -/
  | lt_dd : BOp.Valid .lt (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool
  /-- Less-or-equal: float × float → bool. -/
  | le_ff : BOp.Valid .le .float .float DType.bool
  /-- Less-or-equal: float × nat → bool. -/
  | le_fn : BOp.Valid .le .float .nat DType.bool
  /-- Less-or-equal: float × disc → bool. -/
  | le_fd : BOp.Valid .le .float (.disc (.fin n hn)) DType.bool
  /-- Less-or-equal: nat × float → bool. -/
  | le_nf : BOp.Valid .le .nat .float DType.bool
  /-- Less-or-equal: nat × nat → bool. -/
  | le_nn : BOp.Valid .le .nat .nat DType.bool
  /-- Less-or-equal: nat × disc → bool. -/
  | le_nd : BOp.Valid .le .nat (.disc (.fin n hn)) DType.bool
  /-- Less-or-equal: disc × float → bool. -/
  | le_df : BOp.Valid .le (.disc (.fin m hm)) .float DType.bool
  /-- Less-or-equal: disc × nat → bool. -/
  | le_dn : BOp.Valid .le (.disc (.fin m hm)) .nat DType.bool
  /-- Less-or-equal: disc × disc → bool. -/
  | le_dd : BOp.Valid .le (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool
  /-- Greater-than: float × float → bool. -/
  | gt_ff : BOp.Valid .gt .float .float DType.bool
  /-- Greater-than: float × nat → bool. -/
  | gt_fn : BOp.Valid .gt .float .nat DType.bool
  /-- Greater-than: float × disc → bool. -/
  | gt_fd : BOp.Valid .gt .float (.disc (.fin n hn)) DType.bool
  /-- Greater-than: nat × float → bool. -/
  | gt_nf : BOp.Valid .gt .nat .float DType.bool
  /-- Greater-than: nat × nat → bool. -/
  | gt_nn : BOp.Valid .gt .nat .nat DType.bool
  /-- Greater-than: nat × disc → bool. -/
  | gt_nd : BOp.Valid .gt .nat (.disc (.fin n hn)) DType.bool
  /-- Greater-than: disc × float → bool. -/
  | gt_df : BOp.Valid .gt (.disc (.fin m hm)) .float DType.bool
  /-- Greater-than: disc × nat → bool. -/
  | gt_dn : BOp.Valid .gt (.disc (.fin m hm)) .nat DType.bool
  /-- Greater-than: disc × disc → bool. -/
  | gt_dd : BOp.Valid .gt (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool
  /-- Greater-or-equal: float × float → bool. -/
  | ge_ff : BOp.Valid .ge .float .float DType.bool
  /-- Greater-or-equal: float × nat → bool. -/
  | ge_fn : BOp.Valid .ge .float .nat DType.bool
  /-- Greater-or-equal: float × disc → bool. -/
  | ge_fd : BOp.Valid .ge .float (.disc (.fin n hn)) DType.bool
  /-- Greater-or-equal: nat × float → bool. -/
  | ge_nf : BOp.Valid .ge .nat .float DType.bool
  /-- Greater-or-equal: nat × nat → bool. -/
  | ge_nn : BOp.Valid .ge .nat .nat DType.bool
  /-- Greater-or-equal: nat × disc → bool. -/
  | ge_nd : BOp.Valid .ge .nat (.disc (.fin n hn)) DType.bool
  /-- Greater-or-equal: disc × float → bool. -/
  | ge_df : BOp.Valid .ge (.disc (.fin m hm)) .float DType.bool
  /-- Greater-or-equal: disc × nat → bool. -/
  | ge_dn : BOp.Valid .ge (.disc (.fin m hm)) .nat DType.bool
  /-- Greater-or-equal: disc × disc → bool. -/
  | ge_dd : BOp.Valid .ge (.disc (.fin m hm)) (.disc (.fin n hn)) DType.bool

/-- Addition: any types, result uses additive promotion. -/
@[reducible] def BOp.Valid.add {η₁ η₂ : DType} :
  BOp.Valid .add η₁ η₂
    (match η₁, η₂ with
      | .float, _ => .float
      | _, .float => .float
      | .nat, _ | _, .nat => .nat
      | .disc (.fin _ _), .disc (.fin _ _) => .nat) :=
  match η₁, η₂ with
  | .float, .float => .add_ff
  | .float, .nat => .add_fn
  | .float, .disc (.fin _ _) => .add_fd
  | .nat, .float => .add_nf
  | .nat, .nat => .add_nn
  | .nat, .disc (.fin _ _) => .add_nd
  | .disc (.fin _ _), .float => .add_df
  | .disc (.fin _ _), .nat => .add_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .add_dd

/-- Subtraction: any types, result is promoted. -/
@[reducible] def BOp.Valid.sub {η₁ η₂ : DType} :
    BOp.Valid .sub η₁ η₂
      (match η₁, η₂ with
       | .float, _ => .float
       | _, .float => .float
       | .nat, _ | _, .nat => .nat
       | .disc (.fin m hm), .disc (.fin n hn) => .disc (.fin (max m n))) :=
  match η₁, η₂ with
  | .float, .float => .sub_ff
  | .float, .nat => .sub_fn
  | .float, .disc (.fin _ _) => .sub_fd
  | .nat, .float => .sub_nf
  | .nat, .nat => .sub_nn
  | .nat, .disc (.fin _ _) => .sub_nd
  | .disc (.fin _ _), .float => .sub_df
  | .disc (.fin _ _), .nat => .sub_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .sub_dd

/-- Multiplication: any types, result uses multiplicative promotion. -/
@[reducible] def BOp.Valid.mul {η₁ η₂ : DType} :
    BOp.Valid .mul η₁ η₂
      (match η₁, η₂ with
       | .float, _ => .float
       | _, .float => .float
       | .nat, _ | _, .nat => .nat
       | .disc (.fin m hm), .disc (.fin n hn) => .disc (.fin (m * n) (Nat.mul_pos hm hn))) :=
  match η₁, η₂ with
  | .float, .float => .mul_ff
  | .float, .nat => .mul_fn
  | .float, .disc (.fin _ _) => .mul_fd
  | .nat, .float => .mul_nf
  | .nat, .nat => .mul_nn
  | .nat, .disc (.fin _ _) => .mul_nd
  | .disc (.fin _ _), .float => .mul_df
  | .disc (.fin _ _), .nat => .mul_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .mul_dd

/-- Division: any types, result is float. -/
@[reducible] def BOp.Valid.div {η₁ η₂ : DType} : BOp.Valid .div η₁ η₂ .float :=
  match η₁, η₂ with
  | .float, .float => .div_ff
  | .float, .nat => .div_fn
  | .float, .disc (.fin _ _) => .div_fd
  | .nat, .float => .div_nf
  | .nat, .nat => .div_nn
  | .nat, .disc (.fin _ _) => .div_nd
  | .disc (.fin _ _), .float => .div_df
  | .disc (.fin _ _), .nat => .div_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .div_dd

/-- Boolean and: any operand types, result is bool. -/
@[reducible] def BOp.Valid.and {η₁ η₂ : DType} : BOp.Valid .and η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .and_ff
  | .float, .nat => .and_fn
  | .float, .disc (.fin _ _) => .and_fd
  | .nat, .float => .and_nf
  | .nat, .nat => .and_nn
  | .nat, .disc (.fin _ _) => .and_nd
  | .disc (.fin _ _), .float => .and_df
  | .disc (.fin _ _), .nat => .and_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .and_dd

/-- Boolean or: any operand types, result is bool. -/
@[reducible] def BOp.Valid.or {η₁ η₂ : DType} : BOp.Valid .or η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .or_ff
  | .float, .nat => .or_fn
  | .float, .disc (.fin _ _) => .or_fd
  | .nat, .float => .or_nf
  | .nat, .nat => .or_nn
  | .nat, .disc (.fin _ _) => .or_nd
  | .disc (.fin _ _), .float => .or_df
  | .disc (.fin _ _), .nat => .or_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .or_dd

/-- Equality: any operand types, result is bool. -/
@[reducible] def BOp.Valid.eq {η₁ η₂ : DType} : BOp.Valid .eq η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .eq_ff
  | .float, .nat => .eq_fn
  | .float, .disc (.fin _ _) => .eq_fd
  | .nat, .float => .eq_nf
  | .nat, .nat => .eq_nn
  | .nat, .disc (.fin _ _) => .eq_nd
  | .disc (.fin _ _), .float => .eq_df
  | .disc (.fin _ _), .nat => .eq_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .eq_dd

/-- Inequality: any operand types, result is bool. -/
@[reducible] def BOp.Valid.ne {η₁ η₂ : DType} : BOp.Valid .ne η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .ne_ff
  | .float, .nat => .ne_fn
  | .float, .disc (.fin _ _) => .ne_fd
  | .nat, .float => .ne_nf
  | .nat, .nat => .ne_nn
  | .nat, .disc (.fin _ _) => .ne_nd
  | .disc (.fin _ _), .float => .ne_df
  | .disc (.fin _ _), .nat => .ne_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .ne_dd

/-- Less-than: any operand types, result is bool. -/
@[reducible] def BOp.Valid.lt {η₁ η₂ : DType} : BOp.Valid .lt η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .lt_ff
  | .float, .nat => .lt_fn
  | .float, .disc (.fin _ _) => .lt_fd
  | .nat, .float => .lt_nf
  | .nat, .nat => .lt_nn
  | .nat, .disc (.fin _ _) => .lt_nd
  | .disc (.fin _ _), .float => .lt_df
  | .disc (.fin _ _), .nat => .lt_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .lt_dd

/-- Less-or-equal: any operand types, result is bool. -/
@[reducible] def BOp.Valid.le {η₁ η₂ : DType} : BOp.Valid .le η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .le_ff
  | .float, .nat => .le_fn
  | .float, .disc (.fin _ _) => .le_fd
  | .nat, .float => .le_nf
  | .nat, .nat => .le_nn
  | .nat, .disc (.fin _ _) => .le_nd
  | .disc (.fin _ _), .float => .le_df
  | .disc (.fin _ _), .nat => .le_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .le_dd

/-- Greater-than: any operand types, result is bool. -/
@[reducible] def BOp.Valid.gt {η₁ η₂ : DType} : BOp.Valid .gt η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .gt_ff
  | .float, .nat => .gt_fn
  | .float, .disc (.fin _ _) => .gt_fd
  | .nat, .float => .gt_nf
  | .nat, .nat => .gt_nn
  | .nat, .disc (.fin _ _) => .gt_nd
  | .disc (.fin _ _), .float => .gt_df
  | .disc (.fin _ _), .nat => .gt_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .gt_dd

/-- Greater-or-equal: any operand types, result is bool. -/
@[reducible] def BOp.Valid.ge {η₁ η₂ : DType} : BOp.Valid .ge η₁ η₂ DType.bool :=
  match η₁, η₂ with
  | .float, .float => .ge_ff
  | .float, .nat => .ge_fn
  | .float, .disc (.fin _ _) => .ge_fd
  | .nat, .float => .ge_nf
  | .nat, .nat => .ge_nn
  | .nat, .disc (.fin _ _) => .ge_nd
  | .disc (.fin _ _), .float => .ge_df
  | .disc (.fin _ _), .nat => .ge_dn
  | .disc (.fin _ _), .disc (.fin _ _) => .ge_dd

end Palladium.JAX
