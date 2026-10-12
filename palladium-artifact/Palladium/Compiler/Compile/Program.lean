import Palladium.Compiler.Compile.Block

/-! ## Program Compilation -/

namespace Palladium.PAL.Program

open Palladium.Util
open Palladium.Compiler
open Palladium.TensorLib

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-- Compile a PAL program to a Jay program (rule C-Prog): the body block is compiled
    starting from the probability tensor `1.0`, and the program returns
    `__pal_probability_tensor`. -/
def compile {Δ : WellFormedRandomContext}
    (decl : Program (Θ := Θ) Δ) :
    JAX.Program (L := L) 𝒞⟦Θ⟧ (.tensor 𝒞⟦Δ⟧ .float) :=
  match decl with
  | Program.mk name body =>
      let ΓCtx := 𝒞⟦Θ⟧
      -- Build initial probability tensor expression: scalar 1.0
      -- Note: [] ++ 𝒞⟦Θ⟧ = 𝒞⟦Θ⟧ = ΓCtx definitionally
      let initPtExpr : JAX.Expr (L := L) ΓCtx (JAX.Ty.tensor [] .float) :=
        JAX.Expr.array .float (Tensor.ofTabulated (1.0 : Float))
      -- Compile the return block and cast `↑Δ ++ []` back to the program's
      -- declared output context.
      have hshape : 𝒞⟦body.outputContext⟧ = 𝒞⟦Δ⟧ := by
        cases body
        apply RandomContext.toShape_eq_of_map_snd_eq
        change (((↑Δ : RandomContext) ++ ([] : RandomContext)).map Sigma.snd =
          (↑Δ : RandomContext).map Sigma.snd)
        simp
      let transposeExpr' := hshape ▸ body.compile (Γ := []) initPtExpr
      -- Wrap in single assign + return
      let transposeTy := JAX.Ty.tensor 𝒞⟦Δ⟧ .float
      let finalCtx := List.kinsert probabilityTensor transposeTy ΓCtx
      let assignBody : JAX.Stmts (L := L) ΓCtx finalCtx :=
        .cons (.assign probabilityTensor transposeExpr') .nil
      JAX.Program.mk name assignBody probabilityTensor (.head _)

end Palladium.PAL.Program

namespace Palladium.PAL.Program

open Palladium.Util Palladium.Compiler Palladium.TensorLib

/-- PAL programs compile to JAX programs. -/
@[reducible] instance {Θ' : ValidStaticContext} {Δ : WellFormedRandomContext} :
    Util.Compile (Program (Θ := Θ') Δ) where
  Target := ∀ {L : Tensor.Layout} [Tensor.Operations L] [Tensor.OfTabulated L],
    JAX.Program (L := L) 𝒞⟦Θ'⟧ (.tensor 𝒞⟦Δ⟧ .float)
  compile p := fun {_} [_] [_] => p.compile

end Palladium.PAL.Program
