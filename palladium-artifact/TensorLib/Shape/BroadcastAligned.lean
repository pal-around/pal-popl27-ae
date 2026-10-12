import TensorLib.Shape.Basics

namespace Palladium.TensorLib
namespace Shape

/-! ## Section 1: BroadcastAlignedCompatible — symmetric aligned join predicate -/

/-- Symmetric aligned join: both shapes have the same length, and each pair of dimensions
    is either equal, or one of them is `1`. -/
inductive BroadcastAlignedCompatible : Shape → Shape → Prop
  /-- Two empty shapes are joinable. -/
  | nil : BroadcastAlignedCompatible [] []
  /-- Equal dimensions are joinable. -/
  | eq : BroadcastAlignedCompatible lhs rhs → BroadcastAlignedCompatible (d :: lhs) (d :: rhs)
  /-- A left dimension of `1` joins with any right dimension. -/
  | left : BroadcastAlignedCompatible lhs rhs → BroadcastAlignedCompatible (1 :: lhs) (d :: rhs)
  /-- A right dimension of `1` joins with any left dimension. -/
  | right : BroadcastAlignedCompatible lhs rhs → BroadcastAlignedCompatible (d :: lhs) (1 :: rhs)

/-- Extract the tail joinability proof from cons-cons shapes. -/
theorem BroadcastAlignedCompatible.tail {a : Nat} {lhs : Shape} {b : Nat} {rhs : Shape}
    (h : BroadcastAlignedCompatible (a :: lhs) (b :: rhs)) :
    BroadcastAlignedCompatible lhs rhs := by
  cases h with | eq h | left h | right h => exact h

def broadcastAlignedCompatible :
    (lhs rhs : Shape) → Decidable (BroadcastAlignedCompatible lhs rhs)
  | [], [] => .isTrue .nil
  | [], _ :: _ => .isFalse fun h => nomatch h
  | _ :: _, [] => .isFalse fun h => nomatch h
  | da :: lhs, db :: rhs =>
    match broadcastAlignedCompatible lhs rhs with
    | .isFalse hf => .isFalse fun | .eq h | .left h | .right h => hf h
    | .isTrue ht =>
      if hab : da = db then by subst hab; exact .isTrue (.eq ht)
      else if ha1 : da = 1 then by subst ha1; exact .isTrue (.left ht)
      else if hb1 : db = 1 then by subst hb1; exact .isTrue (.right ht)
      else .isFalse fun | .eq _ => hab rfl | .left _ => ha1 rfl | .right _ => hb1 rfl

instance : Decidable (BroadcastAlignedCompatible lhs rhs) :=
  broadcastAlignedCompatible lhs rhs

/-! ## Section 2: broadcastAligned and broadcastAligned? -/

/-- Proof-free aligned join: compute element-wise broadcast dimensions.
    For mismatched-length inputs, returns `[]` as a harmless default. -/
@[reducible] def broadcastDims : Shape → Shape → Shape
  | [], [] => []
  | a :: lhs, b :: rhs => (if a = 1 then b else a) :: broadcastDims lhs rhs
  | _, _ => []

/-- Total aligned join of two same-length shapes, using the joinability proof to rule out
    mismatched-length cases. -/
def broadcastAligned : (lhs rhs : Shape) → BroadcastAlignedCompatible lhs rhs → Shape
  | [], [], _ => []
  | a :: lhs, b :: rhs, h => (if a = 1 then b else a) :: broadcastAligned lhs rhs h.tail
  | [], _ :: _, h => nomatch h
  | _ :: _, [], h => nomatch h

/-- `broadcastDims` agrees with `broadcastAligned` for compatible shapes. -/
theorem broadcastDims_eq_broadcastAligned {lhs rhs : Shape}
    (h : BroadcastAlignedCompatible lhs rhs) :
    broadcastDims lhs rhs = broadcastAligned lhs rhs h := by
  induction h with
  | nil => rfl
  | eq _ ih => simp [broadcastDims, broadcastAligned, ih]
  | left _ ih => simp [broadcastDims, broadcastAligned, ih]
  | right _ ih => simp [broadcastDims, broadcastAligned, ih]

/-- Convenience: aligned join via a joinability proof. -/
abbrev BroadcastAlignedCompatible.broadcastAligned {lhs rhs : Shape}
    (h : BroadcastAlignedCompatible lhs rhs) : Shape :=
  Shape.broadcastAligned lhs rhs h

/-- When both dimensions are equal, the join takes that dimension. -/
@[simp] theorem broadcastAligned_eq {d : Nat} {lhs rhs : Shape}
    (h : BroadcastAlignedCompatible (d :: lhs) (d :: rhs)) :
    broadcastAligned (d :: lhs) (d :: rhs) h = d :: broadcastAligned lhs rhs h.tail := by
  show (if d = 1 then d else d) :: broadcastAligned lhs rhs h.tail = _
  split <;> rfl

/-- When the left dimension is `1`, the join takes the right dimension. -/
@[simp] theorem broadcastAligned_left {d : Nat} {lhs rhs : Shape}
    (h : BroadcastAlignedCompatible (1 :: lhs) (d :: rhs)) :
    broadcastAligned (1 :: lhs) (d :: rhs) h = d :: broadcastAligned lhs rhs h.tail := rfl

/-- When the right dimension is `1`, the join takes the left dimension. -/
@[simp] theorem broadcastAligned_right {d : Nat} {lhs rhs : Shape}
    (h : BroadcastAlignedCompatible (d :: lhs) (1 :: rhs)) :
    broadcastAligned (d :: lhs) (1 :: rhs) h = d :: broadcastAligned lhs rhs h.tail := by
  show (if d = 1 then 1 else d) :: broadcastAligned lhs rhs h.tail = _
  split <;> simp_all

/-- The aligned join has the same rank as the inputs. -/
theorem broadcastAligned_rank {lhs rhs : Shape} (h : BroadcastAlignedCompatible lhs rhs) :
    (broadcastAligned lhs rhs h).rank = lhs.rank := by
  induction h with
  | nil => rfl
  | eq _ ih => simp [ih]
  | left _ ih => simp [ih]
  | right _ ih => simp [ih]

/-! ## Section 3: BroadcastableAligned — same-rank broadcastability -/

/-- Aligned broadcastability for same-length shapes: each dim of `σₛ` is either equal to the
    corresponding dim of `σₜ`, or is `1`. -/
inductive BroadcastableAligned : Shape → Shape → Prop
  /-- Two empty shapes are broadcastable when aligned. -/
  | nil : BroadcastableAligned [] []
  /-- Equal dimensions are broadcastable. -/
  | eq : BroadcastableAligned σₛ σₜ → BroadcastableAligned (d :: σₛ) (d :: σₜ)
  /-- A dimension of size 1 broadcasts to any dimension. -/
  | one : BroadcastableAligned σₛ σₜ → BroadcastableAligned (1 :: σₛ) (d :: σₜ)

/-- Every shape is broadcastable (aligned) with itself. -/
theorem BroadcastableAligned.refl (σₛ : Shape) : BroadcastableAligned σₛ σₛ := by
  induction σₛ with
  | nil => exact .nil
  | cons d ds ih => exact .eq ih

/-- Transitivity for aligned broadcastability. -/
theorem BroadcastableAligned.trans {σₛ mid σₜ : Shape}
    (h1 : BroadcastableAligned σₛ mid) (h2 : BroadcastableAligned mid σₜ) :
    BroadcastableAligned σₛ σₜ := by
  induction h1 generalizing σₜ with
  | nil => exact h2
  | eq _ ih =>
    cases h2 with
    | eq h2' => exact .eq (ih h2')
    | one h2' => exact .one (ih h2')
  | one _ ih =>
    cases h2 with
    | eq h2' => exact .one (ih h2')
    | one h2' => exact .one (ih h2')

/-- Aligned-broadcastable shapes have equal rank. -/
theorem BroadcastableAligned.rank_eq {σₛ σₜ : Shape}
    (h : BroadcastableAligned σₛ σₜ) : σₛ.rank = σₜ.rank := by
  induction h with
  | nil => rfl
  | eq _ ih => simp [List.length_cons, ih]
  | one _ ih => simp [List.length_cons, ih]

/-- Extract the tail proof from a cons-cons aligned broadcastability. -/
theorem BroadcastableAligned.tail {d d' : Nat} {σₛ σₜ : Shape}
    (h : BroadcastableAligned (d :: σₛ) (d' :: σₜ)) : BroadcastableAligned σₛ σₜ := by
  cases h with | eq h | one h => exact h

/-- Antisymmetry for aligned broadcastability: mutual broadcastability implies equality. -/
theorem BroadcastableAligned.antisymm {σ₁ σ₂ : Shape}
    (h1 : BroadcastableAligned σ₁ σ₂) (h2 : BroadcastableAligned σ₂ σ₁) :
    σ₁ = σ₂ := by
  induction h1 with
  | nil => rfl
  | eq _ ih => exact congrArg _ (ih h2.tail)
  | @one σₛ σₜ d _ ih =>
    cases h2 with
    | eq h2' => exact congrArg (1 :: ·) (ih h2')
    | one h2' => exact congrArg (1 :: ·) (ih h2')

/-- If the head dimensions differ, the source dimension must be `1`. -/
theorem BroadcastableAligned.one_of_ne {d d' : Nat} {σₛ σₜ : Shape}
    (h : BroadcastableAligned (d :: σₛ) (d' :: σₜ)) (hne : d ≠ d') : d = 1 := by
  cases h with | eq _ => exact absurd rfl hne | one _ => rfl

/-! ## Section 4: Aligned bridging theorems -/

/-- Both inputs of a join are broadcastable (aligned) to the join result. -/
theorem BroadcastAlignedCompatible.toBroadcastableAligned {lhs rhs : Shape}
    (h : BroadcastAlignedCompatible lhs rhs) :
    BroadcastableAligned lhs h.broadcastAligned ∧
    BroadcastableAligned rhs h.broadcastAligned := by
  induction h with
  | nil => exact ⟨.nil, .nil⟩
  | eq _ ih => simp only [broadcastAligned_eq]; exact ⟨.eq ih.1, .eq ih.2⟩
  | left _ ih => simp only [broadcastAligned_left]; exact ⟨.one ih.1, .eq ih.2⟩
  | right _ ih => simp only [broadcastAligned_right]; exact ⟨.eq ih.1, .one ih.2⟩

/-- Aligned broadcastability implies aligned joinability. -/
theorem BroadcastableAligned.toBroadcastAlignedCompatible {σₛ σₜ : Shape}
    (h : BroadcastableAligned σₛ σₜ) : BroadcastAlignedCompatible σₛ σₜ := by
  induction h with
  | nil => exact .nil
  | eq _ ih => exact .eq ih
  | one _ ih => exact .left ih

/-- When `BroadcastableAligned σₛ σₜ`, the aligned join equals `σₜ`. -/
theorem broadcastAligned_eq_right {σₛ σₜ : Shape}
    (h : BroadcastableAligned σₛ σₜ) :
    h.toBroadcastAlignedCompatible.broadcastAligned = σₜ := by
  induction h with
  | nil => rfl
  | @eq σₛ σₜ d _ ih => simp [ih]
  | @one σₛ σₜ d _ ih => simp [ih]


instance : Decidable (BroadcastableAligned σₛ σₜ) :=
  if hc : BroadcastAlignedCompatible σₛ σₜ then
    if heq : broadcastAligned σₛ σₜ hc = σₜ then
      .isTrue (by rw [← heq]; exact hc.toBroadcastableAligned.1)
    else
      .isFalse (fun hb => heq (broadcastAligned_eq_right hb))
  else
    .isFalse (fun hb => hc hb.toBroadcastAlignedCompatible)

end Shape
end Palladium.TensorLib
