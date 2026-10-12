import Util.Context
import Util.Discrete

namespace Palladium.PAL

open Palladium.Util
/-- PAL types: floating-point, natural number, or discrete. -/
inductive Ty where
  /-- Floating-point scalar. -/
  | float : Ty
  /-- Unbounded natural number scalar. -/
  | nat : Ty
  /-- Discrete (finite-support) type. -/
  | disc : Discrete → Ty
deriving Repr, DecidableEq, BEq, Inhabited

/-- Boolean type abbreviation: `disc (.fin 2)`. -/
abbrev Ty.bool : Ty := .disc (.fin 2)

abbrev StaticContext := Util.Context Ty
abbrev RandomContext := Util.Context Discrete
abbrev WellFormedRandomContext := Util.WellFormedContext Discrete
/-- Well-formed static context: a PAL static context with unique keys. -/
abbrev WellFormedStaticContext := Util.WellFormedContext PAL.Ty

/-- Promote two types to their common supertype (disc < nat < float). -/
def Ty.promote : Ty → Ty → Ty
  | .float, _ => .float
  | _, .float => .float
  | .nat, _ => .nat
  | _, .nat => .nat
  | .disc (.fin m hm), .disc (.fin n hn) => .disc (.fin (max m n))

/-- Notation for PAL type promotion. -/
scoped infixl:68 " ⊔ " => Ty.promote

/-! # Castable -/

/-- Evidence that a cast from `src` to `tgt` is valid. -/
inductive Ty.Castable : Ty → Ty → Prop where
  /-- Any type can be cast to float. -/
  | toFloat : Ty.Castable src .float
  /-- Natural number can be cast to natural number. -/
  | nat : Ty.Castable .nat .nat
  /-- A discrete type can be cast to nat. -/
  | discToNat : Ty.Castable (.disc d) .nat
  /-- A discrete type can be cast to a larger discrete type. -/
  | disc : (h : m ≤ n) → Ty.Castable (.disc (.fin m hm)) (.disc (.fin n hn))

/-- Extract the Nat inequality proof from a discrete-to-discrete cast. -/
theorem Ty.Castable.le_of_disc (h : Ty.Castable (.disc (.fin m hm)) (.disc (.fin n hn))) : m ≤ n := by
  cases h; assumption

/-- Boolean check for whether a cast from `src` to `tgt` is valid. -/
def Ty.castable (src tgt : Ty) : Bool :=
  match src, tgt with
  | _, .float => true
  | .float, .disc _ | .float, .nat => false
  | .nat, .nat => true
  | .nat, .disc _ => false
  | .disc (.fin m _), .disc (.fin n _) => m ≤ n
  | .disc _, .nat => true

/-- Soundness: `castable` returning true implies `Castable` holds. -/
theorem Ty.Castable.of_castable {src tgt : Ty} (h : src.castable tgt = true) : Ty.Castable src tgt := by
  match src, tgt with
  | _, .float => exact .toFloat
  | .float, .disc _ => simp [Ty.castable] at h
  | .float, .nat => simp [Ty.castable] at h
  | .nat, .nat => exact .nat
  | .nat, .disc _ => simp [Ty.castable] at h
  | .disc (.fin _ _), .nat => exact .discToNat
  | .disc (.fin _ _), .disc (.fin _ _) => exact .disc (by simpa [Ty.castable] using h)

/-- Completeness: `Castable` holding implies `castable` returns true. -/
theorem Ty.Castable.to_castable {src tgt : Ty} (h : Ty.Castable src tgt) : src.castable tgt = true := by
  cases h with
  | toFloat => simp [Ty.castable]
  | nat => simp [Ty.castable]
  | discToNat => simp [Ty.castable]
  | disc hle => simpa [Ty.castable] using hle

/-- Decidability of cast validity between PAL types. -/
instance Ty.Castable.instDecidable (src tgt : Ty) : Decidable (Ty.Castable src tgt) :=
  if h : src.castable tgt then
    .isTrue (Ty.Castable.of_castable h)
  else
    .isFalse fun hc => absurd hc.to_castable h

/-- Casting to the promoted type is valid for the left operand. -/
def Ty.Castable.promote_left (T₁ T₂ : Ty) : Ty.Castable T₁ (T₁ ⊔ T₂) := by
  cases T₁ with
  | float => exact .toFloat
  | nat =>
    cases T₂ with
    | float => exact .toFloat
    | nat => exact .nat
    | disc _ => exact .nat
  | disc d =>
    cases T₂ with
    | float => exact .toFloat
    | nat => exact .discToNat
    | disc d' =>
      cases d with | fin m _ =>
      cases d' with | fin n _ =>
      exact .disc (Nat.le_max_left m n)

/-- Casting to the promoted type is valid for the right operand. -/
def Ty.Castable.promote_right (T₁ T₂ : Ty) : Ty.Castable T₂ (T₁ ⊔ T₂) := by
  cases T₁ with
  | float => exact .toFloat
  | nat =>
    cases T₂ with
    | float => exact .toFloat
    | nat => exact .nat
    | disc _ => exact .discToNat
  | disc d =>
    cases T₂ with
    | float => exact .toFloat
    | nat => exact .nat
    | disc d' =>
      cases d with | fin m _ =>
      cases d' with | fin n _ =>
      exact .disc (Nat.le_max_right m n)

/-- Reserved prefix for compiler-generated names. -/
def reserved : String := "__pal"

/-- Return true when a PAL variable name uses the reserved prefix. -/
def isReserved (name : String) : Bool :=
  reserved.toList.isPrefixOf name.toList

/-- Invariant for PAL static parameter names: all names in `Θ` are non-reserved. -/
def NameInvariant (Θ : StaticContext) :=
  ∀ x {T}, ⟨x, T⟩ ∈ Θ → isReserved x = false

instance (Θ : StaticContext) : Decidable (NameInvariant Θ) :=
  decidable_of_iff (∀ p ∈ Θ, isReserved p.1 = false)
    ⟨fun h x {T} hmem => h ⟨x, T⟩ hmem, fun h ⟨x, T⟩ hmem => @h x T hmem⟩

/-- Valid static context: well-formed static context with non-reserved parameter names.
    Extends `WellFormedStaticContext` with `NameInvariant`. -/
structure ValidStaticContext extends WellFormedStaticContext where
  /-- Parameter names are not reserved. -/
  paramsNotReserved : NameInvariant ctx

instance : Coe ValidStaticContext WellFormedStaticContext where
  coe v := v.toWellFormedContext

instance : Coe ValidStaticContext StaticContext where
  coe v := v.ctx

/-- The empty static context is trivially valid. -/
def ValidStaticContext.empty : ValidStaticContext :=
  ⟨⟨∅, List.nodup_nil⟩, fun _ _ h => nomatch h⟩

/-- Construct a valid static context from a list with decidable proofs. -/
@[reducible] def ValidStaticContext.ofList (Θ : StaticContext)
    (hnd : Θ.NodupKeys := by native_decide)
    (hni : NameInvariant Θ := by native_decide) : ValidStaticContext :=
  ⟨⟨Θ, hnd⟩, hni⟩

namespace ValidStaticContext

instance : EmptyCollection ValidStaticContext where
  emptyCollection := ValidStaticContext.empty

instance : Inhabited ValidStaticContext where
  default := ∅

instance : Membership (Σ _ : String, Ty) ValidStaticContext where
  mem Θ x := x ∈ (↑Θ : StaticContext)

/-- Keys of the valid static context. -/
abbrev keys (Θ : ValidStaticContext) : List String := Θ.toWellFormedContext.keys

/-- The keys have no duplicates. -/
theorem keys_nodup (Θ : ValidStaticContext) : Θ.keys.Nodup := Θ.toWellFormedContext.keys_nodup

end ValidStaticContext

end Palladium.PAL
