import JAX.Syntax
import Util.Denote
import Util.Macros
import TensorLib.Tensor
import Std

/-!
JAX denotation and environments.

This module connects JAX types to their Lean denotations, parameterized by a
layout `L` (any `L` with `[Tensor.Operations L]`), and provides the typed
environment used in evaluation.
-/

namespace Palladium.JAX

open Palladium.Util
open Palladium.TensorLib

/-- JAX types denote their tensor types parameterized by layout `L`.
    Use as `⟦T⟧ (L := L)` to get the concrete tensor type. -/
@[reducible] instance : Util.Denote Ty where
  Sem := (L : Tensor.Layout) → [Tensor.Operations L] → Type
  denote
    | .tensor σ η => fun (L : Tensor.Layout) [Tensor.Operations L] => Tensor (L := L) σ ⟦η⟧

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- Maps JAX types to Lean types, parameterized by layout `L`.
    `Ty.denote (L := L) T = ⟦T⟧ (L := L)`.
    Retained for dot notation and `env{ Ty.denote (L := L) | ... }` macro. -/
@[reducible] def Ty.denote : Ty → Type
  | .tensor σ η => Tensor (L := L) σ ⟦η⟧

/-- Build a `BEq` instance for a layout-parameterized denotation. -/
def Ty.instBEq (T : Ty) :
    BEq (⟦T⟧ (L := L)) := by
  cases T with
  | tensor σ η => exact inferInstance

/-- Build an `Inhabited` instance for a layout-parameterized denotation. -/
def Ty.instInhabited (T : Ty) :
    Inhabited (⟦T⟧ (L := L)) := by
  cases T with
  | tensor σ η => exact inferInstance

instance {T : Ty} : BEq (⟦T⟧ (L := L)) :=
  Ty.instBEq T

instance {T : Ty} : Inhabited (⟦T⟧ (L := L)) :=
  Ty.instInhabited T

/-- Environment parameterized by tensor representation. -/
abbrev Environment (Γ : Context) : Type :=
  Util.Environment (Ty.denote (L := L)) Γ

/-- Default environment using parameterized Inhabited instances. -/
instance {Γ : Context} : Inhabited (Environment (L := L) Γ) where
  default := fun _ τ _ =>
    let _ := Ty.instInhabited (L := L) τ
    default

end

/-! ## Index-wise semantic domains

The denotational semantics of Jay interprets a tensor type as the space of tensors given
extensionally by their indexing functions, `⟦τ[σ]⟧ = I(σ) → ⟦τ⟧`, and a context `Γ` as the
space of valuations `γ` mapping each `x : τ[σ] ∈ Γ` to an element of `⟦τ[σ]⟧`. -/

/-- `⟦τ[σ]⟧ = I(σ) → ⟦τ⟧`: a tensor denotes its indexing function. -/
@[reducible] def Ty.Sem : Ty → Type
  | .tensor σ η => Indices σ → ⟦η⟧

/-- `⟦Γ⟧`: valuations mapping each variable of `Γ` to an indexing function. -/
abbrev Valuation (Γ : Context) : Type := Util.Environment Ty.Sem Γ

/-- JAX contexts denote their environment types, parameterized by layout `L`.
    Use as `⟦Γ⟧ (L := L)` to get `JAX.Environment (L := L) Γ`. -/
@[reducible] instance : Util.Denote Context where
  Sem := (L : Tensor.Layout) → [Tensor.Operations L] → Type
  denote Γ (L : Tensor.Layout) [Tensor.Operations L] := Environment (L := L) Γ

/-- Well-formed JAX contexts denote the environment of their underlying context.
    Use as `⟦Γ⟧ (L := L)` to get `JAX.Environment (L := L) Γ`. -/
@[reducible] instance : Util.Denote WellFormedContext where
  Sem := (L : Tensor.Layout) → [Tensor.Operations L] → Type
  denote Γ (L : Tensor.Layout) [Tensor.Operations L] := Environment (L := L) Γ

end Palladium.JAX
