import Palladium.PAL.Syntax
import Palladium.PAL.Denote
import JAX.Syntax
import JAX.Denote
import JAX.Semantics
import Palladium.Compiler.Compile
import Palladium.Compiler.Encoding
import TensorLib.TensorLemmas
import JAX.Adequacy

/-!
## Parameter Representation

A Jay valuation represents the PAL static environment: for each parameter `x : T ∈ Θ`,
the rank-0 value of `x` in `γ` equals the PAL denotation `θ(x)`. This is independent of
the random context `Δ` and random environment `δ`. `Environment.RepresentsParams` states
the same for a layout environment.
-/

namespace Palladium.JAX

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL
open Palladium.Compiler
open Tensor

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- The JAX environment represents the PAL static environment at the scalar
    level: `∀ x ∈ Θ, γ(x).scalar = ↑(θ(x))`. It does not depend on `Δ` or `δ`. -/
def Environment.RepresentsParams
    {Θ : StaticContext}
    {Γ : JAX.Context}
    (γ : ⟦Γ ++ 𝒞⟦Θ⟧⟧ (L := L))
    (θ : ⟦Θ⟧) : Prop :=
  ∀ (x : String) (T : PAL.Ty) (h : ⟨x, T⟩ ∈ Θ),
    (γ⁅x⁆(StaticContext.toJAXContext_mem_append Γ h)).scalar =
    ↑(θ⁅x⁆h)

end

section

/-- The valuation represents the PAL static environment: `∀ x ∈ Θ, γ(x)[] = ↑(θ(x))`. -/
def Valuation.RepresentsParams
    {Θ : StaticContext}
    {Γ : JAX.Context}
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (θ : ⟦Θ⟧) : Prop :=
  ∀ (x : String) (T : PAL.Ty) (h : ⟨x, T⟩ ∈ Θ),
    (γ⁅x⁆(StaticContext.toJAXContext_mem_append Γ h)) Indices.nil = ↑(θ⁅x⁆h)

/-- `RepresentsParams` is preserved when extending the valuation with a reserved-name
    binding: parameter names are not reserved, so lookups skip the new head binding. -/
theorem Valuation.RepresentsParams.cons_reserved
    {Θ : ValidStaticContext} {Γ : JAX.Context}
    {γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧)}
    {θ : ⟦Θ⟧}
    (hpc : Valuation.RepresentsParams γ θ)
    (name : String) (ty : JAX.Ty) (val : ty.Sem)
    (hReserved : isReserved name = true) :
    Valuation.RepresentsParams (Γ := ⟨name, ty⟩ :: Γ) (Θ := ↑Θ) (γ.cons name ty val) θ := by
  intro y T h_param
  have hne := Ne.symm (ne_of_isReserved hReserved (Θ.paramsNotReserved y h_param))
  rw [Environment.get_cons_ne γ val _ hne]
  exact hpc y T h_param

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- A layout environment that represents the parameters gives a valuation that does. -/
theorem Environment.RepresentsParams.toValuation
    {Θ : StaticContext} {Γ : JAX.Context}
    {γ : ⟦Γ ++ 𝒞⟦Θ⟧⟧ (L := L)} {θ : ⟦Θ⟧}
    (h : Environment.RepresentsParams γ θ) :
    Valuation.RepresentsParams (Environment.toValuation γ) θ :=
  fun x T hmem => h x T hmem

end

/-- Tilde notation for `Valuation.RepresentsParams`. Write `γ ~ θ` in place of
    `Valuation.RepresentsParams γ θ`, matching the PAL-to-JAX parameter correspondence. -/
scoped notation:50 γ " ~ " θ:51 => Valuation.RepresentsParams γ θ

/-- Annotated tilde notation with explicit JAX prefix context.
    Use `γ ~ θ ‖ Γ` when Lean cannot infer `Γ` from `γ`'s type. -/
scoped notation:50 val " ~ " params:51 " ‖ " pfx:51 => Valuation.RepresentsParams (Γ := pfx) val params

end Palladium.JAX
