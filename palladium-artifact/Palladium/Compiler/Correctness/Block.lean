import Palladium.Compiler.Correctness.Stmt
import Palladium.Compiler.Compile.Basics
import Palladium.Compiler.Compile.Block

/-! # Block Correctness -/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL

section

variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]
  [Tensor.OfTabulated L] [Tensor.LawfulOfTabulated L]

/-- The denotation of a compiled block represents the PMF of the block. This is the
    `compileAs`/`evalAs` correctness theorem at the block's declared return context. -/
theorem Block.correctness
    {Θ : ValidStaticContext} {Δ Δ' Δ_ret : WellFormedRandomContext}
    {Γ : JAX.Context}
    (blk : PAL.Block (Θ := Θ) Δ Δ' Δ_ret)
    {θ : ⟦Θ⟧}
    (μ : ℙ⟦Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧)
        (JAX.Ty.tensor 𝒞⟦Δ⟧ .float))
    (hpt : probabilityTensor ∉ (Γ ++ 𝒞⟦Θ⟧).keys) :
    Block.Correct (θ := θ) blk μ γ ptExpr := by
  match blk with
  | PAL.Block.mk (Δ_ret_var := Δ_ret_var) hperm hwf_ext body =>
      simp only [Block.Correct]
      intro hpc hμ
      simpa [PAL.Block.compile, PAL.Block.eval, Util.Denote.denote] using
        Block.correctnessAs (L := L) (Γ := Γ)
          (PAL.Block.mk (Δ_ret_var := Δ_ret_var) hperm hwf_ext body)
          (lhs := Δ_ret) (htypes := rfl) hwf_ext μ γ ptExpr hpt hpc hμ

end

end Palladium.Compiler
