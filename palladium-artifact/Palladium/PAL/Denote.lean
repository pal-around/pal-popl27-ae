import Palladium.PAL.Syntax
import Util.Denote
import Util.Macros
import Util.PMF
import Std

namespace Palladium.PAL

open Palladium.Util

/-! # Denote instances for PAL type descriptors -/

@[reducible] def Ty.denote : Ty → Type
  | Ty.float => Float
  | Ty.nat => Nat
  | Ty.disc d => Discrete.denote d

/-- PAL types denote their runtime scalar/Support types. -/
@[reducible] instance : Util.Denote Ty where
  Sem := Type; denote := Ty.denote

/-- Inhabited instance for PAL type denotations. -/
instance : Util.DenotedInhabited Ty Ty.denote where
  instInhabited T := match T with
    | .float => ⟨(0.0 : Float)⟩
    | .nat => ⟨(0 : Nat)⟩
    | .disc d => Util.DenotedInhabited.instInhabited d

/-- Direct `Inhabited` instance for `PAL.Ty.denote` (used by `Environment`). -/
instance (T : Ty) : Inhabited (Ty.denote T) :=
  Util.DenotedInhabited.instInhabited T

/-! # Type casts with validity proofs -/

/-- Cast a PAL value from source type to target type, given a validity proof. -/
def Ty.cast (src tgt : Ty) (v : ⟦src⟧) (h : Ty.Castable src tgt) : ⟦tgt⟧ :=
  match src, tgt with
  | .float, .float => v
  | .float, .nat   => absurd h.to_castable (by simp [Ty.castable])
  | .float, .disc _ => absurd h.to_castable (by simp [Ty.castable])
  | .nat, .float => Float.ofNat v
  | .nat, .nat   => v
  | .nat, .disc _ => absurd h.to_castable (by simp [Ty.castable])
  | .disc (.fin _ _), .float => Float.ofNat v.val
  | .disc (.fin _ _), .nat   => v.val
  | .disc (.fin m _), .disc (.fin n _) =>
      ⟨v.val, Nat.lt_of_lt_of_le v.isLt (Ty.Castable.le_of_disc h)⟩

/-- Truthiness: convert any PAL value to Bool (nonzero = true). -/
def Ty.toBool (T : Ty) (v : ⟦T⟧) : Bool :=
  match T with
  | .float => ↑v
  | .nat => v != 0
  | .disc (.fin _ _) => v.val != 0

/- Environments map contexts to runtime values. -/
/-- Static environment: values aligned with the static context. -/
abbrev StaticEnvironment (ctx : StaticContext) := Util.Environment Ty.denote ctx

/-- Random environment: values aligned with the random context. -/
abbrev RandomEnvironment (ctx : RandomContext) := Util.Environment Discrete.denote ctx

/- PMF maps a random-variable assignment to a probability mass value. -/
/-- Probability mass function over assignments. -/
abbrev PMF (ctx : RandomContext) := Util.PMF Discrete.denote ctx

/-- Probability-bracket notation for `PAL.PMF`. Write `ℙ⟦Δ⟧` in place of
    `PMF Δ` to match the standard probability-measure convention.
    Uses U+2119 (double-struck P) with U+27E6 / U+27E7. -/
scoped notation "ℙ⟦" Δ "⟧" => Palladium.PAL.PMF Δ

/-- Conditional probability: maps a conditioning assignment to a PMF. -/
abbrev ConditionalPMF (cond rand : RandomContext) := Util.ConditionalPMF Discrete.denote cond rand

/-- Conditional-probability-bracket notation for `PAL.ConditionalPMF`.
    Write `ℙ⟦Γ | Δ⟧` in place of `ConditionalPMF Δ Γ` to match the standard
    conditional-probability convention P(Γ | Δ). Note the argument swap:
    `Γ` (random variables) appears first in the notation but second in the function.
    Uses U+2119 (double-struck P) with U+27E6 / U+27E7 and ASCII `|`. -/
scoped notation "ℙ⟦" Γ " | " Δ "⟧" => Palladium.PAL.ConditionalPMF Δ Γ

/-- Static contexts denote their environment types. -/
@[reducible] instance : Util.Denote StaticContext where
  Sem := Type; denote := StaticEnvironment

/-- Random contexts denote their environment types. -/
@[reducible] instance : Util.Denote RandomContext where
  Sem := Type; denote := RandomEnvironment

/-- Well-formed random contexts denote their environment types. -/
@[reducible] instance : Util.Denote WellFormedRandomContext where
  Sem := Type; denote Δ := RandomEnvironment Δ

/-- Well-formed static contexts denote their environment types. -/
@[reducible] instance : Util.Denote WellFormedStaticContext where
  Sem := Type; denote Θ := StaticEnvironment Θ

/-- Valid static contexts denote their environment types. -/
@[reducible] instance : Util.Denote ValidStaticContext where
  Sem := Type; denote Θ := StaticEnvironment Θ

end Palladium.PAL
