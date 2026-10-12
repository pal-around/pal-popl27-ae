import Palladium.Compiler.Compile.Reduce

/-! ## Statement Compilation -/

namespace Palladium.PAL.Stmt

open Palladium.Util
open Palladium.PAL
open Palladium.Compiler
open Palladium.TensorLib
open Shape

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-! ### Sampling

`x⃗ ← sample (wpp e)` compiles (rule C-Sample, with `m = |x⃗|`) to

    letIn PT ptExpr (comment ("Sample " ++ x⃗)
      (letIn w (broadcast_to(𝒞⟦e⟧, 𝒞⟦x⃗, Δ⟧))
        (letIn p (w ⊘ REDUCE(w, m, 0)) (PT ⊙ p))))

The weight `w` is the compiled weight expression at the full shape `𝒞⟦x⃗, Δ⟧`,
`REDUCE(w, m, 0)` is the normalizing constant over all valuations of `x⃗`, and the
result scales the normalized weights by the prior mass `PT`. -/

section

variable {Δ : WellFormedRandomContext} (vars : WellFormedRandomContext)

/-- Type of the probability tensor before sampling. -/
abbrev samplePtTy (Δ : WellFormedRandomContext) : JAX.Ty := JAX.Ty.tensor 𝒞⟦Δ⟧ .float

/-- Type of the weight, normalized-weight, and output tensors of a sample. -/
abbrev sampleTy (Δ : WellFormedRandomContext) : JAX.Ty :=
  JAX.Ty.tensor 𝒞⟦(↑vars ++ ↑Δ : RandomContext)⟧ .float

/-- `𝒞⟦x⃗, Δ⟧ = 𝒞⟦x⃗⟧ ++ 𝒞⟦Δ⟧`. -/
theorem sampleShape_eq :
    𝒞⟦(↑vars ++ ↑Δ : RandomContext)⟧ = 𝒞⟦(↑vars : RandomContext)⟧ ++ 𝒞⟦Δ⟧ :=
  RandomContext.toShape_append _ _

/-- `𝒞⟦Δ⟧ ≤ 𝒞⟦x⃗, Δ⟧`. -/
theorem sampleShape_suffix : 𝒞⟦Δ⟧ ≤ 𝒞⟦(↑vars ++ ↑Δ : RandomContext)⟧ :=
  (sampleShape_eq vars (Δ := Δ)).symm ▸ Broadcastable.suffix _ _

/-- The weight `broadcast_to(𝒞⟦e⟧, 𝒞⟦x⃗, Δ⟧)`, in the context where `PT` is bound. -/
def sampleWeight {Γ : JAX.Context} (hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys)
    (e : Expr (Θ := ↑Θ) (↑vars ++ ↑Δ) Ty.float) :
    JAX.Expr (L := L) (⟨probabilityTensor, samplePtTy Δ⟩ :: (Γ ++ 𝒞⟦Θ⟧))
      (sampleTy vars Δ) :=
  let cr := e.compile (L := L) (Δ := ⟨↑vars ++ ↑Δ, hwf⟩)
    (Γ := ⟨probabilityTensor, samplePtTy Δ⟩ :: Γ)
  let w : JAX.Expr (L := L) (⟨probabilityTensor, samplePtTy Δ⟩ :: (Γ ++ 𝒞⟦Θ⟧))
      (JAX.Ty.tensor 𝒞⟦(↑vars ++ ↑Δ : RandomContext)⟧ 𝒞⟦Ty.float⟧) :=
    JAX.Expr.broadcastTo cr.expr _ cr.broadcastable
  w

/-- The normalized weights `w ⊘ REDUCE(w, m, 0)`, where `w` is the head binding. -/
def sampleNormalize {Γ' : JAX.Context} (w : String) :
    JAX.Expr (L := L) (⟨w, sampleTy vars Δ⟩ :: Γ') (sampleTy vars Δ) :=
  let wRef : JAX.Expr (L := L) (⟨w, sampleTy vars Δ⟩ :: Γ') (sampleTy vars Δ) :=
    JAX.Expr.var w _ (.head _)
  let zExpr := reduce 𝒞⟦(↑vars : RandomContext)⟧ (castShape (sampleShape_eq vars) wRef)
  let hc := (Shape.Broadcastable.broadcast2_common_data
    (Shape.Broadcastable.refl _) (sampleShape_suffix vars)).val
  castShape (broadcast2_eq_left_of_le (sampleShape_suffix vars) hc)
    (JAX.Expr.bop .div wRef zExpr .div_ff hc)

/-- The output `PT ⊙ p`, where `p` is the head binding and `PT` is bound two levels below. -/
def sampleUpdate {Γ' : JAX.Context} (p w : String) :
    JAX.Expr (L := L)
      (⟨p, sampleTy vars Δ⟩ :: ⟨w, sampleTy vars Δ⟩ :: ⟨probabilityTensor, samplePtTy Δ⟩ :: Γ')
      (sampleTy vars Δ) :=
  let hc := (Shape.Broadcastable.broadcast2_common_data
    (sampleShape_suffix vars) (Shape.Broadcastable.refl _)).val
  castShape (Shape.Broadcastable.broadcast2_eq_right (sampleShape_suffix vars) hc)
    (JAX.Expr.bop .mul (JAX.Expr.var probabilityTensor _ (.tail _ (.tail _ (.head _))))
      (JAX.Expr.var p _ (.head _)) .mul_ff hc)

/-- Compile `x⃗ ← sample wppc` (rule C-Sample). -/
def compileSample {Γ : JAX.Context}
    (hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys)
    (wppc : Wppc (Θ := Θ) Δ vars hwf)
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (samplePtTy Δ)) :
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (sampleTy vars Δ) :=
  match wppc with
  | .wpp e =>
    let name := sampleName vars.keys
    JAX.Expr.letIn probabilityTensor ptExpr
      (JAX.Expr.comment ("Sample " ++ name)
        (JAX.Expr.letIn (weightName name) (sampleWeight vars hwf e)
          (JAX.Expr.letIn (probName name) (sampleNormalize vars (weightName name))
            (sampleUpdate vars (probName name) (weightName name)))))

end

end Palladium.PAL.Stmt
