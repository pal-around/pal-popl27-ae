import Lean
import Util.Denote

namespace Palladium.Util

/-! # Context and environment literal macros -/

open Lean

abbrev UtilMacroTerm := Lean.TSyntax `term

private def mkList (xs : Array UtilMacroTerm) : Lean.MacroM UtilMacroTerm := do
  let mut acc ← `(List.nil)
  for x in xs.reverse do
    acc ← `(List.cons $x $acc)
  pure acc

declare_syntax_cat utilKey
syntax ident : utilKey
syntax str : utilKey

declare_syntax_cat ctxBinding
syntax utilKey ":" term : ctxBinding
syntax "ctx{" ctxBinding,* "}" : term

declare_syntax_cat envBinding
syntax utilKey ":" term ":=" term : envBinding
syntax "env{" term "|" envBinding,* "}" : term
syntax "env{" envBinding,* "}" : term

private def expandKey (stx : Lean.Syntax) : Lean.MacroM UtilMacroTerm := do
  match stx with
  | `(utilKey| $k:ident) =>
      pure (Lean.quote k.getId.toString)
  | `(utilKey| $k:str) =>
      pure (Lean.quote k.getString)
  | _ =>
      Lean.Macro.throwErrorAt stx "unsupported key syntax"

private partial def stripParens (stx : Lean.Syntax) : Lean.Syntax :=
  match stx with
  | `(term| ( $t:term )) => stripParens t
  | _ => stx

private def discreteFinNat? (stx : Lean.Syntax) : Option UtilMacroTerm :=
  let stx := stripParens stx
  match stx with
  | `(term| Discrete.fin $n) => some n
  | `(term| Discrete.fin $n $_) => some n
  | `(term| Palladium.Util.Discrete.fin $n) => some n
  | `(term| Palladium.Util.Discrete.fin $n $_) => some n
  | `(term| .fin $n) => some n
  | `(term| .fin $n $_) => some n
  | _ => none

private def isPalTyFloat (stx : Lean.Syntax) : Bool :=
  let stx := stripParens stx
  match stx with
  | `(term| Ty.float) => true
  | `(term| PAL.Ty.float) => true
  | `(term| Palladium.PAL.Ty.float) => true
  | _ => false

private def palTyDiscFinNat? (stx : Lean.Syntax) : Option UtilMacroTerm :=
  let stx := stripParens stx
  match stx with
  | `(term| Ty.disc $d) => discreteFinNat? d
  | `(term| PAL.Ty.disc $d) => discreteFinNat? d
  | `(term| Palladium.PAL.Ty.disc $d) => discreteFinNat? d
  | _ => none

private def expandEnvValue (payload : Lean.Syntax) (value : Lean.Syntax) : Lean.MacroM UtilMacroTerm := do
  let payload := stripParens payload
  let value' := stripParens value
  match value' with
  | `(term| $n:num) =>
      if let some m := discreteFinNat? payload then
        `((⟨$n, by omega⟩ : Fin $m))
      else if isPalTyFloat payload then
        `(Float.ofNat $n)
      else if let some m := palTyDiscFinNat? payload then
        `((⟨$n, by omega⟩ : Fin $m))
      else
        pure ⟨value⟩
  | `(term| $s:scientific) =>
      if isPalTyFloat payload then
        pure (← `(($s : Float)))
      else
        pure ⟨value⟩
  | _ => pure ⟨value⟩

private def mkEnvBindings (bindings : Array (Lean.TSyntax `envBinding)) :
    Lean.MacroM UtilMacroTerm := do
  let mut entries : Array UtilMacroTerm := #[]
  for binding in bindings do
    match binding with
    | `(envBinding| $key:utilKey : $payload:term := $value:term) =>
      let keyTerm ← expandKey key
      let valueTerm ← expandEnvValue payload value
      entries := entries.push (← `(⟨$keyTerm, $payload, $valueTerm⟩))
    | _ =>
      Lean.Macro.throwErrorAt binding "unsupported environment binding"
  mkList entries

macro "ctx{" bindings:ctxBinding,* "}" : term => do
  let mut entries : Array UtilMacroTerm := #[]
  for binding in bindings.getElems do
    match binding with
    | `(ctxBinding| $key:utilKey : $payload:term) =>
        let keyTerm ← expandKey key
        entries := entries.push (← `(⟨$keyTerm, $payload⟩))
    | _ =>
        Lean.Macro.throwErrorAt binding "unsupported context binding"
  mkList entries

declare_syntax_cat wfctxBinding
syntax ident ":" term : wfctxBinding

macro "wfctx{" bindings:wfctxBinding,* "}" : term => do
  let mut entries : Array UtilMacroTerm := #[]
  for binding in bindings.getElems do
    match binding with
    | `(wfctxBinding| $key:ident : $payload:term) =>
        let keyTerm := Lean.quote key.getId.toString
        entries := entries.push (← `(⟨$keyTerm, $payload⟩))
    | _ =>
        Lean.Macro.throwErrorAt binding "unsupported wfctx binding"
  let listTerm ← mkList entries
  `(WellFormedContext.ofList $listTerm)

macro "env{" denote:term "|" bindings:envBinding,* "}" : term => do
  let bindingsTerm ← mkEnvBindings bindings.getElems
  `(Environment.ofList $denote $bindingsTerm)

macro "env{" bindings:envBinding,* "}" : term => do
  if bindings.getElems.isEmpty then
    `(default)
  else
    let bindingsTerm ← mkEnvBindings bindings.getElems
    `(Environment.ofList Denote.denote $bindingsTerm)

/-- Quill-bracket notation for `Environment.get`. Write `env⁅name⁆h` in place
    of `env.get name h`; the type `τ` is inferred from the proof `h : ⟨name, τ⟩ ∈ ctx`.
    Works for any string-valued expression: variable (`x`), constant
    (`probabilityTensor`), or application (`enumName y`).

    Uses U+2045/U+2046 (square brackets with quill) to avoid conflict with
    comparison operators and anonymous constructor brackets. The proof argument `h`
    is parsed at maximum precedence (identifier or parenthesized expression).
    Proof-carrying tensor indexing (`t[i]'h`) works unchanged via the built-in
    `getElem` macro. -/
scoped notation:max env "⁅" name "⁆" h:max => Palladium.Util.Environment.get env name h

/-- Double-bracket notation for `Denote.denote`. Write `⟦T⟧` in place of
    `Denote.denote T` to match the standard denotational-semantics convention.
    Uses U+27E6 / U+27E7 (mathematical left/right white square bracket).

    We use `syntax` with `(priority := high)` so this wins over Mathlib's global
    `notation3` for `Quotient.mk` (`Mathlib.Data.Quot`), which would otherwise
    shadow `⟦·⟧` in any file that imports Mathlib transitively. -/
scoped syntax:max (priority := high) "⟦" term "⟧" : term
macro_rules
  | `(term|⟦$t:term⟧) => `(Denote.denote $t)

/-- Semantic-universe notation for `Denote.Sem`. Write `Sem⟦α⟧` in place of
    `Denote.Sem (α := α)` to extract the target `Sort` of a denotable type.
    Uses U+27E6 / U+27E7 with a `Sem` prefix. -/
scoped notation "Sem⟦" α "⟧" => Denote.sem α

/-- Compilation-bracket notation for `Compile.compile`. Write `𝒞⟦x⟧` in place
    of `Compile.compile x`. Uses U+27E6 / U+27E7 with `𝒞` prefix. -/
scoped notation "𝒞⟦" T "⟧" => Compile.compile T

end Palladium.Util
