import Palladium.Compiler.Compile.Basics
import Palladium.Compiler.Encoding.ContextShape

/-! ## Enum Support

Consolidates all enum-support-related definitions: `finSupport` (finite support literal),
`enumShape` (canonical shape for enum tensors), and `enumExpr` (inline enum expression builder).

The tensor of values of a random variable is the inline expression
`reshape(finSupport(n), enumShape)` (rule C-Var). It does not reference any JAX context
variable. -/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL

/-! ### Enum Shape

The canonical shape for the enum tensor of a random variable `x` in context `Δ`.
For variable `x` at position `j` with `.fin n`, the shape is
`n :: List.replicate rest.length 1` — support size followed by trailing ones matching
the number of older variables.

This shape is fully determined by the context structure and the variable's position,
so it can be *computed* rather than carried as an abstract field. -/

/-- Canonical enum shape for variable `x` in random context `Δ`.
    For the variable at position `j` with `.fin n`, the shape is
    `n :: List.replicate rest.keys.length 1` — support size followed by trailing ones. -/
def enumShape : (Δ : RandomContext) → (x : String) → {D : Discrete} → ⟨x, D⟩ ∈ Δ → Shape
  | [], _, _, h => nomatch h
  | ⟨y, .fin n _⟩ :: rest, x, _, h =>
    if hxy : x = y then n :: List.replicate rest.keys.length 1
    else enumShape rest x (Context.mem_cons_ne hxy h)

/-- When the variable matches the head, `enumShape` returns `n :: replicate Δ.keys.length 1`. -/
@[simp] theorem enumShape_self {x : String} {n : Nat} {hn : 0 < n}
    {Δ : RandomContext} {D : Discrete}
    {h : ⟨x, D⟩ ∈ (⟨x, .fin n hn⟩ :: Δ : RandomContext)} :
    enumShape (⟨x, .fin n hn⟩ :: Δ) x h = n :: List.replicate Δ.keys.length 1 := by
  simp [enumShape]

/-- When the variable does not match the head, `enumShape` delegates to the tail. -/
theorem enumShape_ne {x y : String} {D_y : Discrete} {D : Discrete}
    {Δ : RandomContext}
    {h : ⟨x, D⟩ ∈ (⟨y, D_y⟩ :: Δ : RandomContext)}
    (hne : x ≠ y) :
    enumShape (⟨y, D_y⟩ :: Δ) x h = enumShape Δ x (Context.mem_cons_ne hne h) := by
  cases D_y with
  | fin n hn => simp [enumShape, dif_neg hne]

open Shape in
/-- Every enum shape is broadcastable to the random context shape. -/
theorem enumShape_broadcastable
    {Δ : RandomContext} {x : String} {D : Discrete}
    (h : ⟨x, D⟩ ∈ Δ) (hnd : Δ.keys.Nodup) :
    enumShape Δ x h ≤ 𝒞⟦Δ⟧ := by
  induction Δ with
  | nil => exact nomatch h
  | cons hd rest ih =>
    match hd with
    | ⟨y, .fin n hn⟩ =>
    by_cases hxy : x = y
    · subst hxy
      simp only [enumShape_self]
      rw [show rest.keys.length = (↑rest : Shape).rank
        from (RandomContext.toShape_length rest).symm]
      exact Shape.Broadcastable.replicate_ones_cons n ↑rest
    · simp only [enumShape_ne hxy]
      exact Shape.Broadcastable.cons_left n
        (ih (Context.mem_cons_ne hxy h) (List.nodup_cons.mp hnd).2)

/-! ### Reshapeability of enum shapes -/

/-- The flat support `[n]` is reshapeable to the canonical `enumShape`. -/
theorem enumShape_reshapeable {Δ : RandomContext} {x : String} {n : Nat} {hn : 0 < n}
    (h : ⟨x, .fin n hn⟩ ∈ Δ) (hnd : Δ.keys.Nodup) :
    Shape.Reshapeable [n] (enumShape Δ x h) := by
  induction Δ with
  | nil => exact nomatch h
  | cons hd rest ih =>
    match hd with
    | ⟨y, .fin m hm⟩ =>
      by_cases hxy : x = y
      · subst hxy
        have hmn : m = n := by
          cases h with
          | head => rfl
          | tail _ h' =>
            exact absurd (List.mem_keys.mpr ⟨_, h'⟩) (List.nodup_cons.mp hnd).1
        subst hmn
        simp only [enumShape_self]
        exact Shape.Reshapeable.enum m rest.keys.length
      · simp only [enumShape_ne hxy]
        exact ih (Context.mem_cons_ne hxy h) (List.nodup_cons.mp hnd).2

/-! ### Finite support literal -/

variable {L : Tensor.Layout} [Tensor.Operations L]
  [Tensor.OfTabulated L]

/-- Compile a finite support as a JAX array literal with discrete type. -/
def finSupport {Γ : JAX.Context} (n : Nat) (pos : 0 < n) :
    JAX.Expr (L := L) Γ (JAX.Ty.tensor [n] (.disc (.fin n pos))) :=
  JAX.Expr.array (.disc (.fin n pos))
    (Tensor.arange (L := L) n)

/-! ### Inline enum expression builder -/

/-- Build an inline enum expression `reshape(finSupport(n), enumShape)`.
    This expression does NOT reference any JAX context variable — `Γ` is unused
    in the expression body. -/
def enumExpr {Γ : JAX.Context} {Δ : RandomContext} {x : String} {n : Nat} {hn : 0 < n}
    (h : ⟨x, .fin n hn⟩ ∈ Δ) (hnd : Δ.keys.Nodup) :
    JAX.Expr (L := L) Γ (JAX.Ty.tensor (enumShape Δ x h) (.disc (.fin n hn))) :=
  JAX.Expr.reshape (finSupport n hn) (enumShape Δ x h) (enumShape_reshapeable h hnd)

end Palladium.Compiler
