import Palladium.Compiler.Compile.CompileResults

/-! ## Marginalization Compilation -/

namespace Palladium.PAL.Stmt

open Palladium.Util
open Palladium.PAL
open Palladium.Compiler
open Palladium.TensorLib
open Shape

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-! ### Marginalization -/

/-- Compile a marginalization statement into a JAX expression (rule C-Marg).
    Takes the current probability tensor expression and wraps it in a `letIn`:
    `letIn PT ptExpr (comment ("Marginalize " ++ name) (sum(PT, axis)))`. -/
def compileMarginalize {Δ : WellFormedRandomContext}
    {Γ : JAX.Context} (name : String) (D : Discrete)
    (h : ⟨name, D⟩ ∈ Δ)
    (ptExpr : JAX.Expr (L := L) Γ (JAX.Ty.tensor 𝒞⟦Δ⟧ .float)) :
    JAX.Expr (L := L) Γ (JAX.Ty.tensor 𝒞⟦Δ.kerase name⟧ .float) :=
      let idx := randomAxis ↑Δ name D h
      have hIdxLt : idx < Δ.keys.length := randomAxis_lt ↑Δ name D h
      have hIdxLt' : idx < 𝒞⟦Δ⟧.rank := by rw [Δ.toShape_length]; exact hIdxLt
      let newPtShape := 𝒞⟦Δ⟧.slice ⟨idx, hIdxLt'⟩
      -- Bind PT in a letIn, then sum along the marginalized axis
      let ptTy := JAX.Ty.tensor 𝒞⟦Δ⟧ .float
      let ptCtx := (⟨probabilityTensor, ptTy⟩ : Σ _ : String, JAX.Ty) :: Γ
      let ptRef : JAX.Expr (L := L) ptCtx (JAX.Ty.tensor 𝒞⟦Δ⟧ .float) :=
        JAX.Expr.var probabilityTensor _ (.head _)
      let sumRaw :=
        JAX.Expr.sum ptRef [⟨idx, hIdxLt'⟩] (List.nodup_singleton _)
      have hSumTy :
          JAX.Ty.tensor (𝒞⟦Δ⟧.removeAxes [⟨idx, hIdxLt'⟩]) (DType.float.sumOutMany 1) =
            JAX.Ty.tensor newPtShape DType.float := by
        simp [newPtShape, removeAxes_singleton, DType.sumOutMany_float]
      let sumExpr : JAX.Expr (L := L) ptCtx (JAX.Ty.tensor newPtShape .float) :=
        hSumTy ▸ sumRaw
      -- Key: Δ.keys[idx] = name (from randomAxis)
      have hKeysIdx : Δ.keys[idx]'hIdxLt = name :=
        randomAxis_keysGetElem ↑Δ name D h
      -- Prove newPtShape = RandomContext.toShape (List.kerase name Δ)
      have hNewPtEq : newPtShape = 𝒞⟦Δ.kerase name⟧ :=
        Δ.toShape_delete name idx hIdxLt hKeysIdx
      -- Bind PT, then comment + sum (comment inside for pretty-print order)
      hNewPtEq ▸ JAX.Expr.letIn probabilityTensor ptExpr
        (JAX.Expr.comment ("Marginalize " ++ name) sumExpr)

end Palladium.PAL.Stmt
