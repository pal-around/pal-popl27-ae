import Palladium.Compiler.Compile.Basics

/-! ### Compilation Results

Output-bundle structures (`CompileResult`) for each compilation stage. -/

namespace Palladium.PAL

open Palladium.Util
open Palladium.TensorLib
open Palladium.Compiler
open Shape

section CompileResults

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Result of compiling a PAL expression. -/
structure Expr.CompileResult
    (Δ : RandomContext) (Γ : JAX.Context) (Tₒ : PAL.Ty) where
  /-- Output shape of the compiled expression. -/
  σ : Shape
  /-- The compiled JAX expression. -/
  expr : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ 𝒞⟦Tₒ⟧)
  /-- Output shape broadcasts to the random context shape. -/
  broadcastable : σ ≤ 𝒞⟦Δ⟧



end CompileResults

end Palladium.PAL
