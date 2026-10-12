import Util.Context
import Std

namespace Palladium.Util

/-! # Denote typeclass

Universe-polymorphic denotation class. For type descriptors (e.g. `PAL.Ty`, `DType`),
`Sem = Type` and `denote` maps descriptors to Lean types. For AST nodes (e.g. `Expr`),
`Sem` can be a function type and `denote` maps syntax to semantic functions.
-/

/-- Maps descriptors or syntax to their semantic denotation. -/
class Denote.{u} (α : Type) where
  /-- The semantic universe (e.g. `Type` for descriptors, a function type for AST). -/
  Sem : outParam (Sort u)
  /-- The denotation function. -/
  denote : α → Sem

attribute [reducible] Denote.denote
attribute [reducible] Denote.Sem

/-- Helper abbreviation for the `Sem⟦α⟧` notation. Extracts the semantic universe
    from the `Denote` instance for `α`. -/
abbrev Denote.sem (α : Type) [inst : Denote α] := inst.Sem

/-! # Compile typeclass

Compilation class mapping PAL constructs to JAX representations.
-/

/-- Maps PAL constructs to their compiled JAX representations. -/
class Compile.{u} (α : Type) where
  /-- The compilation target type. -/
  Target : outParam (Sort u)
  /-- The compilation function. -/
  compile : α → Target

attribute [reducible] Compile.compile
attribute [reducible] Compile.Target

/-- Helper abbreviation that extracts the compilation target `Sort` from the
    `Compile` instance for `α`. -/
abbrev Compile.target (α : Type) [inst : Compile α] := inst.Target

/-!
# Denotation-based Inhabited

Some type descriptors have default values in their denotation. This class
packages that capability so it can be reused uniformly.
-/

/-- Provide `Inhabited` instances for denotations. -/
class DenotedInhabited (α : Type) (denote : outParam (α → Type)) where
  /-- `Inhabited` instance for the denotation of a specific type descriptor. -/
  instInhabited : (T : α) → Inhabited (denote T)

/-- Use `DenotedInhabited` to supply `Inhabited (f T)`. -/
instance (T : α) [DenotedInhabited α f] : Inhabited (f T) :=
  DenotedInhabited.instInhabited T

/-!
# Runtime environments

Runtime environments aligned with contexts. Environments provide a value for
any name that is provably present in the context with a given payload type.

Binding proofs use list membership (`⟨x, τ⟩ ∈ ctx`) rather than `List.dlookup`.
This simplifies operations like `kerase` (no `DenotedInhabited` needed) and
`tail` (no freshness proof needed).
-/

/-- Typed environment aligned with a context. -/
def Environment {α : Type} (denote : α → Type) (ctx : Context α) : Type :=
  ∀ x τ, ⟨x, τ⟩ ∈ ctx → denote τ

/-- Empty-context environments are always inhabited. -/
instance {α : Type} {denote : α → Type} :
    Inhabited (Environment denote ([] : Context α)) where
  default := fun _ _ h => nomatch h

/-- Environments are inhabited when all denotations are inhabited. -/
instance {α : Type} {denote : α → Type} [(τ : α) → Inhabited (denote τ)]
    {ctx : Context α} : Inhabited (Environment denote ctx) where
  default := fun _ _ _ => default

namespace Environment

/-- Fetch a value using a proof that the name is bound. -/
def get {α : Type} {denote : α → Type} {ctx : Context α}
    (env : Environment denote ctx) (name : String) {τ : α}
    (h : ⟨name, τ⟩ ∈ ctx) : denote τ :=
  env name τ h

/-- Update/insert a binding in the environment. -/
def kinsert {α : Type} {denote : α → Type} [DecidableEq α] {ctx : Context α}
    (env : Environment denote ctx) (name : String) (payload : α)
    (value : denote payload) : Environment denote (List.kinsert name payload ctx) :=
  fun x τ h =>
    if heq : (⟨x, τ⟩ : Σ _ : String, α) = ⟨name, payload⟩ then
      (Context.sigma_snd_eq heq) ▸ value
    else
      env x τ ((List.kerase_sublist name ctx).subset (by
        cases h with
        | head => exact absurd rfl heq
        | tail _ h' => exact h'))

/-- Drop a binding from the environment. -/
def kerase {α : Type} {denote : α → Type} {ctx : Context α}
    (env : Environment denote ctx) (name : String) :
    Environment denote (List.kerase name ctx) :=
  fun x τ h => env x τ ((List.kerase_sublist name ctx).subset h)

/-- Insert a binding into a smaller environment. -/
def insert {α : Type} {denote : α → Type} [DecidableEq α] {ctx : Context α} (key : String)
    (env : Environment denote (List.kerase key ctx)) (payload : α)
    (value : denote payload) (hkey : ⟨key, payload⟩ ∈ ctx)
    (hnd : ctx.NodupKeys) :
    Environment denote ctx :=
  fun x τ h =>
    if heq : (⟨x, τ⟩ : Σ _ : String, α) = ⟨key, payload⟩ then
      (Context.sigma_snd_eq heq) ▸ value
    else
      have hne : x ≠ key := by
        intro hxk; subst hxk
        have := hnd.eq_of_mk_mem h hkey; subst this
        exact heq rfl
      env x τ (Context.mem_kerase_of_ne hne h)

/-- Restrict a cons-extended environment to the tail context. -/
def tail {α : Type} {denote : α → Type} {name : String} {payload : α} {ctx : Context α}
    (env : Environment denote (⟨name, payload⟩ :: ctx)) : Environment denote ctx :=
  fun x τ h => env x τ (.tail _ h)

/-- Extend an environment with a leading (cons) binding. -/
def cons {α : Type} {denote : α → Type} [DecidableEq α] {ctx : Context α}
    (env : Environment denote ctx) (name : String) (payload : α)
    (value : denote payload) : Environment denote (⟨name, payload⟩ :: ctx) :=
  fun x τ h =>
    if heq : (⟨x, τ⟩ : Σ _ : String, α) = ⟨name, payload⟩ then
      (Context.sigma_snd_eq heq) ▸ value
    else
      env x τ (by
        cases h with
        | head => exact absurd rfl heq
        | tail _ h' => exact h')

/-- Adapt an environment from a permuted context back to the original.
    Requires a Perm proof and NodupKeys. -/
def unpermute {α : Type} {denote : α → Type} {ctx : Context α} {names : List String}
    {hperm : names.Perm ctx.keys}
    (env : Environment denote (ctx.permute names hperm))
    (hnd : ctx.NodupKeys) :
    Environment denote ctx :=
  fun x τ h => env x τ (Context.mem_permute ctx names hperm hnd h)

/-- Getting the head binding from a cons-extended environment returns the inserted value. -/
theorem get_cons_eq {α : Type} {denote : α → Type} [DecidableEq α] {ctx : Context α}
    (env : Environment denote ctx) (name : String) (payload : α)
    (value : denote payload) :
    (env.cons name payload value).get name (.head _) = value := by
  simp [get, cons]

/-- Getting a different name from a cons-extended environment skips to the tail. -/
theorem get_cons_ne {α : Type} {denote : α → Type} [DecidableEq α] {ctx : Context α}
    {name₁ : String} {τ₁ : α}
    (env : Environment denote ctx) (value : denote τ₁)
    {name₂ : String} {τ₂ : α} (h : ⟨name₂, τ₂⟩ ∈ ctx)
    (hne : name₂ ≠ name₁) :
    (env.cons name₁ τ₁ value).get name₂ (List.Mem.tail _ h) = env.get name₂ h := by
  simp only [get, cons,
    dif_neg (show (⟨name₂, τ₂⟩ : Σ _ : String, α) ≠ ⟨name₁, τ₁⟩ from
      fun heq => hne (congrArg Sigma.fst heq))]

/-- Getting the name that was just inserted returns the inserted value. -/
theorem get_kinsert_eq {α : Type} {denote : α → Type} [DecidableEq α] {ctx : Context α}
    (env : Environment denote ctx) (name : String) (payload : α)
    (value : denote payload) :
    (env.kinsert name payload value).get name (.head _) = value := by
  simp [get, kinsert]

/-- Convert an environment from one context to a permuted context.
    Given a `List.Perm` proof, looks up each binding via the permutation. -/
def ofPerm {α : Type} {denote : α → Type} {ctx₁ ctx₂ : Context α}
    (env : Environment denote ctx₁) (hperm : ctx₁.Perm ctx₂) :
    Environment denote ctx₂ :=
  fun x τ h => env x τ (hperm.mem_iff.mpr h)

def empty {α} (denote : α → Type) :
  Environment denote (∅ : Context α) :=
  fun _ _ h => nomatch h

def ofListCtx {α}
  (denote : α → Type)
  [DecidableEq α]
  (bindings :
    List (Σ _name : String, Σ payload : α, denote payload)) :
  Σ ctx : Context α, Environment denote ctx :=
bindings.foldr
  (fun ⟨name, payload, value⟩ ⟨ctx, env⟩ =>
    ⟨⟨name, payload⟩ :: ctx,
     cons env name payload value⟩)
  ⟨(∅ : Context α), empty denote⟩

def ofList {α}
  (denote : α → Type)
  [DecidableEq α]
  (bindings :
    List (Σ _name : String, Σ payload : α, denote payload)) :
  Environment denote (ofListCtx denote bindings).fst :=
  (ofListCtx denote bindings).snd

/-- Rename an environment by positional type matching: given `env : Environment denote ctx₁`
    where `ctx₁` and `ctx₂` have the same types at corresponding positions,
    produce `Environment denote ctx₂`.
    Each position in `ctx₂` gets the value from the corresponding position in `ctx₁`. -/
def renameByTypes {α : Type} {denote : α → Type} [DecidableEq α]
    {ctx₁ ctx₂ : Context α}
    (htypes : ctx₁.map Sigma.snd = ctx₂.map Sigma.snd)
    (env : Environment denote ctx₁) :
    Environment denote ctx₂ :=
  renameByTypesGo ctx₁ ctx₂ htypes env
where
  renameByTypesGo : (from_ to_ : Context α) →
      (htypes : from_.map Sigma.snd = to_.map Sigma.snd) →
      Environment denote from_ →
      Environment denote to_
    | [], [], _, _ => fun _ _ h => nomatch h
    | ⟨_fn, fτ⟩ :: from_rest, ⟨tn, tτ⟩ :: to_rest, htypes, env =>
        have h_eq : fτ = tτ := by
          have := congrArg (fun l => l.head?) htypes; simp at this; exact this
        have htypes_rest : from_rest.map Sigma.snd = to_rest.map Sigma.snd := by
          have := congrArg List.tail htypes; simp at this; exact this
        let headVal : denote tτ := h_eq ▸ env.get _ (.head _)
        let restEnv := renameByTypesGo from_rest to_rest htypes_rest env.tail
        restEnv.cons tn tτ headVal

/-! ### Environments over `pre ++ ctx` -/

/-- Restrict an environment over `pre ++ ctx` to the prefix `pre`. -/
def takePrefix {α : Type} {denote : α → Type} {pre ctx : Context α}
    (env : Environment denote (pre ++ ctx)) : Environment denote pre :=
  fun x τ h => env x τ (List.mem_append_left ctx h)

/-- Restrict an environment over `pre ++ ctx` to the suffix `ctx`. -/
def dropPrefix {α : Type} {denote : α → Type} {pre ctx : Context α}
    (env : Environment denote (pre ++ ctx)) : Environment denote ctx :=
  fun x τ h => env x τ (List.mem_append_right pre h)

/-- Extend `env : Environment denote ctx` with the bindings `vals` for the prefix `pre`,
    giving an environment over `pre ++ ctx`. -/
def prepend {α : Type} {denote : α → Type} [DecidableEq α] {ctx : Context α} :
    (pre : Context α) → Environment denote pre → Environment denote ctx →
      Environment denote (pre ++ ctx)
  | [], _, env => env
  | ⟨x, τ⟩ :: rest, vals, env => (prepend rest vals.tail env).cons x τ (vals x τ (.head _))

/-- Splitting an environment over `pre ++ ctx` and prepending the parts gives it back. -/
theorem prepend_takePrefix_dropPrefix {α : Type} {denote : α → Type} [DecidableEq α]
    {ctx : Context α} :
    (pre : Context α) → (env : Environment denote (pre ++ ctx)) →
      prepend pre env.takePrefix env.dropPrefix = env
  | [], _ => rfl
  | ⟨x, τ⟩ :: rest, env => by
      funext y υ h
      simp only [prepend, cons]
      split
      · next heq =>
        cases heq
        rfl
      · next hne =>
        have ih := prepend_takePrefix_dropPrefix rest (env.tail)
        exact congrFun (congrFun (congrFun ih y) υ) _

/-- The prefix of a prepended environment is the prepended part. -/
theorem takePrefix_prepend {α : Type} {denote : α → Type} [DecidableEq α]
    {ctx : Context α} :
    (pre : Context α) → (vals : Environment denote pre) → (env : Environment denote ctx) →
      (prepend pre vals env).takePrefix = vals
  | [], _, _ => funext fun _ => funext fun _ => funext fun h => nomatch h
  | ⟨x, τ⟩ :: rest, vals, env => by
      funext y υ h
      simp only [takePrefix, prepend, cons]
      split
      · next heq =>
        cases heq
        rfl
      · next hne =>
        have hmem : ⟨y, υ⟩ ∈ rest := by
          cases h with
          | head => exact absurd rfl hne
          | tail _ h' => exact h'
        have ih := takePrefix_prepend rest vals.tail env
        exact congrFun (congrFun (congrFun ih y) υ) hmem

/-- The suffix of a prepended environment is the original environment, provided the
    names of `pre` are fresh for `ctx`. -/
theorem dropPrefix_prepend {α : Type} {denote : α → Type} [DecidableEq α]
    {ctx : Context α} :
    (pre : Context α) → (hfresh : ∀ x ∈ pre.keys, x ∉ ctx.keys) →
      (vals : Environment denote pre) → (env : Environment denote ctx) →
      (prepend pre vals env).dropPrefix = env
  | [], _, _, _ => rfl
  | ⟨x, τ⟩ :: rest, hfresh, vals, env => by
      funext y υ h
      simp only [dropPrefix, prepend, cons]
      split
      · next heq =>
        have hx : y = x := congrArg Sigma.fst heq
        subst hx
        exact absurd (List.mem_keys.mpr ⟨υ, h⟩) (hfresh y (by simp))
      · next =>
        have ih := dropPrefix_prepend rest
          (fun z hz => hfresh z (List.mem_cons_of_mem _ hz)) vals.tail env
        exact congrFun (congrFun (congrFun ih y) υ) h


end Environment

end Palladium.Util
