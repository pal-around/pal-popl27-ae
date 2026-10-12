import Palladium.Compiler.Compile.Expr

/-! ## Sample Compilation Helpers

Shape lemmas, shape casts, and the compile-time macro `REDUCE(e, m, 0)` used by
`compileSample` (rule C-Sample). `REDUCE(e, m, 0)` unrolls to `m` consecutive sums
along axis 0. -/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.PAL
open Palladium.TensorLib
open Shape

/-- `σ₁ ⊔ σ₂ = σ₁` when `σ₂` broadcasts to `σ₁`. -/
theorem broadcast2_eq_left_of_le {σ₁ σ₂ : Shape} (h : σ₂ ≤ σ₁) (hc : σ₁ ⋈ σ₂) :
    σ₁ ⊔ σ₂ ‖ hc = σ₁ :=
  Shape.Broadcastable.antisymm
    (Shape.Broadcastable.broadcast2_common_data (Shape.Broadcastable.refl σ₁) h).property.2.2
    (Shape.Broadcastable.of_broadcastCompatible_left hc)

/-- A suffix broadcasts to the whole shape: `σ ≤ pre ++ σ`. -/
theorem Broadcastable.suffix : (pre σ : Shape) → σ ≤ pre ++ σ
  | [], σ => Shape.Broadcastable.refl σ
  | n :: pre, σ => Shape.Broadcastable.cons_left n (Broadcastable.suffix pre σ)

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Transport a JAX expression along a shape equality. -/
def castShape {Γ : JAX.Context} {σ₁ σ₂ : Shape} {η : DType} (h : σ₁ = σ₂)
    (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₁ η)) : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ₂ η) :=
  cast (congrArg (fun σ => JAX.Expr (L := L) Γ (JAX.Ty.tensor σ η)) h) e

/-- The type of `sum(e, 0)` for a float tensor `e` of shape `n :: σ`. -/
theorem sum_axis0_ty (n : Nat) (σ : Shape) :
    JAX.Ty.tensor (Shape.removeAxes (n :: σ) [⟨0, Nat.succ_pos _⟩])
        (DType.float.sumOutMany 1) =
      JAX.Ty.tensor σ .float := by
  rw [removeAxes_singleton, DType.sumOutMany_float]
  rfl

/-- `sum(e, 0)` on a float tensor of shape `n :: σ`, typed at shape `σ`. -/
def sumAxis0 {Γ : JAX.Context} {n : Nat} {σ : Shape}
    (e : JAX.Expr (L := L) Γ (JAX.Ty.tensor (n :: σ) .float)) :
    JAX.Expr (L := L) Γ (JAX.Ty.tensor σ .float) :=
  cast (congrArg (JAX.Expr (L := L) Γ) (sum_axis0_ty n σ))
    (JAX.Expr.sum e [⟨0, Nat.succ_pos _⟩] (List.nodup_singleton _))

/-- `REDUCE(e, m, 0)` with `m = |pre|`: sum out the leading axes `pre` of `e`, axis 0 first.
    `REDUCE(e, 0, 0) = e` and `REDUCE(e, m, 0) = REDUCE(sum(e, 0), m - 1, 0)`. -/
def reduce {Γ : JAX.Context} :
    (pre : Shape) → {σ : Shape} → JAX.Expr (L := L) Γ (JAX.Ty.tensor (pre ++ σ) .float) →
      JAX.Expr (L := L) Γ (JAX.Ty.tensor σ .float)
  | [], _, e => e
  | _ :: pre, _, e => reduce pre (sumAxis0 e)

end

end Palladium.Compiler
