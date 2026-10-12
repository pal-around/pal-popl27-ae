import Palladium.Compiler.Correctness.Block
import Palladium.Compiler.Compile.Basics

/-! # Program Correctness -/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL

section

variable {L : Tensor.Layout} [Tensor.Operations L] [Tensor.LawfulOperations L]
  [Tensor.OfTabulated L] [Tensor.LawfulOfTabulated L]

/-- Whole-program correctness: `γ ~ θ → ∀ δ, ⟦𝒞⟦prog⟧⟧ γ [idx(δ)] = ⟦prog⟧ θ δ`.

    `⟦𝒞⟦prog⟧⟧` is `JAX.Program.denote`, the denotational semantics of Jay, and `γ` is a
    valuation of the parameters. -/
theorem Program.correctness
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext}
    (prog : PAL.Program (Θ := Θ) Δ)
    {θ : ⟦Θ⟧}
    (γ : JAX.Valuation 𝒞⟦Θ⟧) :
    Program.Correct (L := L) (θ := θ) prog γ := by
  match prog with
  | PAL.Program.mk name body =>
  simp only [Program.Correct]
  intro hpc
  -- Local definitions matching compile.
  let ΓCtx := 𝒞⟦Θ⟧
  let initPtExpr : JAX.Expr (L := L) ΓCtx (.tensor [] .float) :=
    .array .float (Tensor.ofTabulated (1.0 : Float))
  -- Step 1: The initial probability tensor represents the trivial PMF.
  have hμ_init : IdxFn.RepresentsPMF (ctx := (∅ : RandomContext)) (initPtExpr.denote γ)
      (fun _ => 1.0) := by
    intro δ
    exact Tensor.scalar_ofTabulated (L := L) (1.0 : Float)
  -- Step 2: Block correctness over the block's combined output `↑Δ ++ []`.
  let extCtx : WellFormedRandomContext := body.outputContext
  have htypes : (↑extCtx : RandomContext).map Sigma.snd =
      (↑Δ : RandomContext).map Sigma.snd := by
    cases body
    change (((↑Δ : RandomContext) ++ ([] : RandomContext)).map Sigma.snd =
      (↑Δ : RandomContext).map Sigma.snd)
    simp
  have hpt : probabilityTensor ∉ ΓCtx.keys :=
    probabilityTensor_not_in_append (Γ := []) (hΓ := by simp)
  have h_block := (Block.correctness (Γ := []) body
    (fun _ => 1.0) γ initPtExpr hpt) hpc hμ_init
  have h_block_cast := h_block.of_shape_cast htypes extCtx.wf
  -- Step 3: The program returns the denotation of the compiled block.
  simp only [Util.Compile.compile]
  show ((PAL.Program.mk name body).compile (L := L)).denote γ ~ _
  intro δ
  simp only [PAL.Program.compile, JAX.Program.denote,
    JAX.Stmts.denote, JAX.Stmt.denote, Environment.get_kinsert_eq]
  rw [denote_shape_cast]
  exact h_block_cast δ

/-- Whole-program correctness for the evaluation of Jay in a tensor layout `L`:
    `γ ~ θ → ∀ δ, (𝒞⟦prog⟧.eval γ)[idx(δ)] = ⟦prog⟧ θ δ`, for a layout environment `γ`
    of the parameters. Follows from `Program.correctness` and the adequacy theorem
    `JAX.Program.eval_toSem`. -/
theorem Program.correctness_eval
    {Θ : ValidStaticContext} {Δ : WellFormedRandomContext}
    (prog : PAL.Program (Θ := Θ) Δ)
    {θ : ⟦Θ⟧}
    (γ : JAX.Environment (L := L) 𝒞⟦Θ⟧)
    (hγ : JAX.Environment.RepresentsParams (Γ := []) γ θ) :
    Tensor.RepresentsPMF ((𝒞⟦prog⟧ (L := L)).eval γ) (⟦prog⟧ θ) := by
  intro δ
  have hcorr := Program.correctness (L := L) prog (θ := θ) (JAX.Environment.toValuation γ)
    hγ.toValuation δ
  have hadeq := JAX.Program.eval_toSem (Util.Compile.compile prog (L := L)) γ
  rw [Environment.getElem_renv, Tensor.getElem_eq_index]
  exact (congrFun hadeq δ.toIndices).trans hcorr

end

end Palladium.Compiler
