import Palladium.Compiler.Compile.CompileResults
import Palladium.Compiler.Compile.EnumSupport

/-! ## Expression Compilation -/

namespace Palladium.PAL.Expr

open Palladium.Util
open Palladium.TensorLib
open Palladium.TensorLib.Shape
open Palladium.PAL
open Palladium.Compiler

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-- Compile a PAL expression into a JAX expression.
    Returns the shape, expression, and broadcastability to the target shape.
    Rules: C-Const (`fin`, `float`), C-Var (`var`), C-Param (`param`), C-Unary (`uop`),
    C-Binary (`bop`), C-Ternary (`ite`). -/
def compile {Δ : WellFormedRandomContext}
    (e : Expr (Θ := Θ) ↑Δ T) {Γ : JAX.Context} :
    CompileResult (L := L) ↑Δ (Γ ++ 𝒞⟦Θ⟧) T :=
  match e with
  | Expr.fin i =>
      ⟨[], JAX.Expr.array (.disc (.fin _ _))
        (Tensor.ofTabulated (L := L) (σ := []) i),
       Shape.Broadcastable.nil_left 𝒞⟦Δ⟧⟩
  | Expr.float r =>
      ⟨[], JAX.Expr.array .float
        (Tensor.ofTabulated (L := L) (σ := []) r),
       Shape.Broadcastable.nil_left 𝒞⟦Δ⟧⟩
  | Expr.var x (.fin n hn) h =>
      ⟨enumShape ↑Δ x h,
       enumExpr h Δ.wf,
       enumShape_broadcastable h Δ.wf⟩
  | Expr.param x T h =>
      ⟨[], JAX.Expr.var x _ (StaticContext.toJAXContext_mem_append Γ h),
       Shape.Broadcastable.nil_left 𝒞⟦Δ⟧⟩
  | Expr.uop op e hv =>
      let ⟨σ, e', hbc⟩ := e.compile (Γ := Γ)
      ⟨σ, JAX.Expr.uop (UOp.compile op) e' (UOp.Valid.compile hv), hbc⟩
  | Expr.bop op a b hv =>
      let ⟨σ₁, a', ha_bc⟩ := a.compile (Γ := Γ)
      let ⟨σ₂, b', hb_bc⟩ := b.compile (Γ := Γ)
      let bcd := Shape.Broadcastable.broadcast2_common_data ha_bc hb_bc
      ⟨_, JAX.Expr.bop (BOp.compile op) a' b' (BOp.Valid.compile hv)
        bcd.val, bcd.property.2.2⟩
  | @Expr.ite _ _ T₁ T₂ c t e =>
      let cr := c.compile (Γ := Γ)
      let tr := t.compile (Γ := Γ)
      let er := e.compile (Γ := Γ)
      let bcd1 := Shape.Broadcastable.broadcast2_common_data tr.broadcastable er.broadcastable
      let bcd2 := Shape.Broadcastable.broadcast2_common_data cr.broadcastable bcd1.property.2.2
      let h := Shape.BroadcastNCompatible.of_triple bcd1.val bcd2.val
      let jaxExpr := JAX.Expr.select cr.expr tr.expr er.expr h
      ⟨(⊔ₙ #v[cr.σ, tr.σ, er.σ] ‖ h),
       by show JAX.Expr _ (JAX.Ty.tensor _ 𝒞⟦T₁ ⊔ T₂⟧)
          exact (Ty.compile_promote T₁ T₂).symm ▸ jaxExpr,
       by exact Shape.broadcastN_triple_eq bcd1.val bcd2.val ▸ bcd2.property.2.2⟩

section

variable {L : Tensor.Layout} [Tensor.OfTabulated L]
variable {Θ : ValidStaticContext}

/-- PAL expression compilation never emits JAX `letIn`. -/
theorem compile_res_expr_hasLetIn_false [Tensor.Operations L]
    {Δ : WellFormedRandomContext} {Γ : JAX.Context} {T : PAL.Ty}
    (e : PAL.Expr (Θ := Θ) ↑Δ T) :
    (e.compile (L := L) (Γ := Γ)).expr.hasLetIn = false := by
  match e with
  | .fin .. => simp [compile, JAX.Expr.hasLetIn]
  | .float .. => simp [compile, JAX.Expr.hasLetIn]
  | .var x (.fin n hn) h =>
      simp [compile, JAX.Expr.hasLetIn, Palladium.Compiler.enumExpr,
        Palladium.Compiler.finSupport]
  | .param .. => simp [compile, JAX.Expr.hasLetIn]
  | .uop _ e .. =>
      simp [compile, JAX.Expr.hasLetIn]
      exact compile_res_expr_hasLetIn_false e
  | .bop _ a b .. =>
      simp [compile, JAX.Expr.hasLetIn]
      rw [compile_res_expr_hasLetIn_false a, compile_res_expr_hasLetIn_false b]
      simp
  | @Expr.ite _ _ T₁ T₂ c t e' =>
      let cr := c.compile (L := L) (Γ := Γ)
      let tr := t.compile (L := L) (Γ := Γ)
      let er := e'.compile (L := L) (Γ := Γ)
      let bcd1 := Shape.Broadcastable.broadcast2_common_data tr.broadcastable er.broadcastable
      let bcd2 := Shape.Broadcastable.broadcast2_common_data cr.broadcastable bcd1.property.2.2
      let hbc := Shape.BroadcastNCompatible.of_triple bcd1.val bcd2.val
      let jaxExpr := JAX.Expr.select cr.expr tr.expr er.expr hbc
      simp [compile]
      rw [JAX.Expr.hasLetIn_cast_dtype (Ty.compile_promote T₁ T₂).symm jaxExpr]
      have hs :
          jaxExpr.hasLetIn =
            (cr.expr.hasLetIn || tr.expr.hasLetIn || er.expr.hasLetIn) := rfl
      rw [hs, compile_res_expr_hasLetIn_false c, compile_res_expr_hasLetIn_false t,
        compile_res_expr_hasLetIn_false e']
      simp

end

open Palladium.Util Palladium.TensorLib Palladium.Compiler

/-- PAL expressions compile to JAX compile results. -/
@[reducible] instance {Δ : WellFormedRandomContext} :
    Util.Compile (Expr (Θ := Θ) ↑Δ T) where
  Target := ∀ {L : Tensor.Layout} [Tensor.Operations L] [Tensor.OfTabulated L] {Γ : JAX.Context},
    CompileResult (L := L) ↑Δ (Γ ++ 𝒞⟦Θ⟧) T
  compile e := fun {_} [_] [_] {Γ} => e.compile (Δ := Δ) (Γ := Γ)

end Palladium.PAL.Expr
