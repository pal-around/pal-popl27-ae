import Palladium.Compiler.Compile.Stmt

/-! ## Block Compilation -/

namespace Palladium.PAL.Block

open Palladium.Util Palladium.Compiler Palladium.TensorLib

/-- PAL blocks compile to JAX expression transformers.
    Output tensor is over the block's declared `Δ_ret ++ Δ'` context. -/
@[reducible] instance {Θ : ValidStaticContext} {Δ Δ' Δ_ret : WellFormedRandomContext} :
    Util.Compile (Block (Θ := Θ) Δ Δ' Δ_ret) where
  Target := ∀ {L : Tensor.Layout} [Tensor.Operations L] [Tensor.OfTabulated L] {Γ : JAX.Context}
    (hwf_ext : (↑Δ_ret ++ ↑Δ' : RandomContext).NodupKeys),
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float) →
    JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧)
      (JAX.Ty.tensor 𝒞⟦(⟨↑Δ_ret ++ ↑Δ', hwf_ext⟩ : WellFormedRandomContext)⟧ .float)
  compile blk := fun {_} [_] [_] {_} hwf_ext e =>
    have hshape :
        𝒞⟦blk.outputContext⟧ =
        𝒞⟦(⟨↑Δ_ret ++ ↑Δ', hwf_ext⟩ : WellFormedRandomContext)⟧ := by
      cases blk
      rfl
    hshape ▸ blk.compile e

end Palladium.PAL.Block
