import Mathlib.Data.List.Sigma


namespace Palladium.Util

/-!
Context infrastructure: ordered lists of named bindings.

Contexts represent parameter lists, random variable declarations, and similar
structures. This module is separate from Environment, which pairs contexts
with aligned runtime values.

The underlying representation is `List (Σ _ : String, α)`, using Mathlib's
`NodupKeys` for uniqueness proofs. Lookup, insertion, and erasure use Mathlib's
`List.dlookup`, `List.kinsert`, and `List.kerase` respectively.
-/

/-- Contexts are ordered association lists of named bindings. -/
abbrev Context (α : Type) := List (Σ _ : String, α)

instance : Inhabited (Context α) := ⟨[]⟩
instance : EmptyCollection (Context α) := ⟨[]⟩
instance [Repr α] : Repr (Context α) where
  reprPrec ctx _ := repr ctx

/-- `NodupKeys` is decidable because it reduces to `Nodup` on a list of strings. -/
instance (ctx : Context α) : Decidable ctx.NodupKeys :=
  List.nodupDecidable ctx.keys

/-- A context with a proof that its keys are unique. -/
structure WellFormedContext (α : Type) where
  /-- The underlying association list. -/
  ctx : Context α
  /-- All keys are distinct. -/
  wf : ctx.NodupKeys

namespace WellFormedContext

instance [Repr α] : Repr (WellFormedContext α) where
  reprPrec wfc n := reprPrec wfc.ctx n

instance : Coe (WellFormedContext α) (Context α) where
  coe wfc := wfc.ctx

instance : Membership (Σ _ : String, α) (WellFormedContext α) where
  mem wfc x := x ∈ (↑wfc : Context α)

instance [DecidableEq α] (x : Σ _ : String, α) (wfc : WellFormedContext α) :
    Decidable (x ∈ wfc) :=
  inferInstanceAs (Decidable (x ∈ (↑wfc : Context α)))

/-- Keys of the well-formed context. -/
abbrev keys (wfc : WellFormedContext α) : List String := (↑wfc : Context α).keys

/-- The keys have no duplicates. -/
theorem keys_nodup (wfc : WellFormedContext α) : wfc.keys.Nodup := wfc.wf

/-- Empty well-formed context. -/
instance : EmptyCollection (WellFormedContext α) where
  emptyCollection := ⟨[], List.nodup_nil⟩

instance : Inhabited (WellFormedContext α) where
  default := ∅

/-- Erase a key, preserving well-formedness. -/
@[reducible] def kerase (wfc : WellFormedContext α) (name : String) : WellFormedContext α :=
  ⟨List.kerase name ↑wfc, List.NodupKeys.kerase name wfc.wf⟩

/-- Insert/replace a key-value pair, preserving well-formedness. -/
@[reducible] def kinsert (wfc : WellFormedContext α)
    (name : String) (val : α) : WellFormedContext α :=
  ⟨List.kinsert name val ↑wfc, List.kinsert_nodupKeys name val wfc.wf⟩

/-- Cons a fresh binding, preserving well-formedness. -/
@[reducible] def cons (wfc : WellFormedContext α) (name : String) (val : α)
    (hfresh : name ∉ wfc.keys) : WellFormedContext α :=
  ⟨⟨name, val⟩ :: (↑wfc : Context α), List.nodupKeys_cons.mpr ⟨hfresh, wfc.wf⟩⟩

/-- Lookup by name. -/
def dlookup (wfc : WellFormedContext α) (name : String) : Option α :=
  List.dlookup name (↑wfc : Context α)

/-- Construct from a context and a NodupKeys proof. -/
def ofContext (ctx : Context α) (h : ctx.NodupKeys) : WellFormedContext α :=
  ⟨ctx, h⟩

/-- Construct a well-formed context from a list with a decidable uniqueness proof. -/
def ofList (ctx : Context α) (h : ctx.NodupKeys := by native_decide) : WellFormedContext α :=
  ⟨ctx, h⟩

/-- Membership in a well-formed context is membership in the underlying context. -/
@[simp] theorem mem_coe {wfc : WellFormedContext α} {x : Σ _ : String, α} :
    x ∈ wfc ↔ x ∈ (↑wfc : Context α) := Iff.rfl

/-- Permutation relation, delegating to `List.Perm`. -/
abbrev Perm (a : WellFormedContext α) (b : Context α) : Prop :=
  (↑a : Context α).Perm b

end WellFormedContext

namespace Context

-- instance : Coe (List (String × α)) (Context α) where
--   coe l := l.map fun (k, v) => ⟨k, v⟩

-- /-- Coercion from `String × α` to `Σ _ : String, α` for tuple cons syntax. -/
-- instance : Coe (String × α) (Σ _ : String, α) where
--   coe | (k, v) => ⟨k, v⟩

/-! ## dlookup helpers for constant-family sigma lists

These wrap Mathlib's `List.dlookup_cons_ne`, `List.dlookup_kinsert`, etc.
for our `Context α = List (Σ _ : String, α)` (constant family) to avoid
dependent-type unification issues with the general `Sigma β` versions.
-/

/-- Lookup the head name returns the head value. -/
@[simp] protected theorem dlookup_cons_eq (ctx : Context α) (name : String) (payload : α) :
    List.dlookup name ((⟨name, payload⟩ : Σ _ : String, α) :: ctx) = some payload := by
  simp

/-- Lookup a different name in a cons context delegates to the tail. -/
protected theorem dlookup_cons_ne (ctx : Context α) (name : String) (payload : α)
    (x : String) (hne : x ≠ name) :
    List.dlookup x ((⟨name, payload⟩ : Σ _ : String, α) :: ctx) = List.dlookup x ctx := by
  simp [hne]

/-- Lookup after kinsert returns the inserted value for the same key. -/
protected theorem dlookup_kinsert (name : String) (payload : α) (ctx : Context α) :
    List.dlookup name (List.kinsert name payload ctx) = some payload := by
  simp

/-- Lookup distributes over append: left match wins. -/
protected theorem dlookup_append (ctx₁ ctx₂ : Context α) (x : String) :
    List.dlookup x (ctx₁ ++ ctx₂) = (List.dlookup x ctx₁).or (List.dlookup x ctx₂) := by
  exact List.dlookup_append ctx₁ ctx₂ x

/-- Internal helper for `permute`: reorder using subset proof. -/
private def permuteAux (ctx : Context α) :
    (names : List String) → (∀ (x : String), x ∈ names → x ∈ ctx.keys) → Context α
  | [], _ => []
  | n :: ns, hsub =>
    ⟨n, (List.dlookup n ctx).get (List.dlookup_isSome.mpr (hsub n List.mem_cons_self))⟩ ::
    permuteAux ctx ns (fun x hx => hsub x (List.mem_cons_of_mem n hx))

/-- Reorder context entries according to a permutation of names.
    Requires a proof that `names` is a permutation of `ctx.keys`. -/
def permute (ctx : Context α) (names : List String) (h : names.Perm ctx.keys) : Context α :=
  permuteAux ctx names (fun _x hx => h.subset hx)

/-! ## Permute lemmas -/

private theorem dlookup_permuteAux
    (ctx : Context α) (names : List String)
    (hsub : ∀ (x : String), x ∈ names → x ∈ ctx.keys) (x : String) :
    List.dlookup x (permuteAux ctx names hsub) =
      if x ∈ names then List.dlookup x ctx else none := by
  induction names with
  | nil => simp [permuteAux]
  | cons n ns ih =>
      unfold permuteAux
      by_cases hx : x = n
      · subst hx
        simp only [Context.dlookup_cons_eq, List.mem_cons, true_or, ↓reduceIte]
        exact Option.some_get (List.dlookup_isSome.mpr (hsub x List.mem_cons_self))
      · rw [Context.dlookup_cons_ne _ n _ x hx]
        simp only [List.mem_cons, hx, false_or]
        exact ih _

/-- Lookup in original succeeds implies lookup in permuted succeeds. -/
theorem dlookup_permute_of_perm (ctx : Context α) (names : List String)
    (hperm : names.Perm ctx.keys)
    (x : String) (τ : α)
    (h : List.dlookup x ctx = some τ) :
    List.dlookup x (ctx.permute names hperm) = some τ := by
  have hxKeys : x ∈ ctx.keys := by
    rw [List.mem_keys]; exact ⟨τ, List.of_mem_dlookup h⟩
  have hxNames : x ∈ names := hperm.symm.subset hxKeys
  simp only [permute, dlookup_permuteAux, hxNames, ↓reduceIte, h]

/-! ## Membership helpers for constant-family sigma lists

These provide `∈`-based reasoning for `Context α = List (Σ _ : String, α)`.
-/

/-- Membership in cons with a different key implies membership in the tail. -/
protected theorem mem_cons_ne {ctx : Context α} {name : String} {payload : α}
    {x : String} {τ : α} (hne : x ≠ name)
    (h : ⟨x, τ⟩ ∈ (⟨name, payload⟩ :: ctx : Context α)) :
    ⟨x, τ⟩ ∈ ctx := by
  cases h with
  | head => exact absurd rfl hne
  | tail _ h' => exact h'

/-- Membership with a different key is preserved by kerase. -/
protected theorem mem_kerase_of_ne {ctx : Context α} {name : String}
    {x : String} {τ : α} (hne : x ≠ name)
    (h : ⟨x, τ⟩ ∈ ctx) :
    ⟨x, τ⟩ ∈ List.kerase name ctx := by
  induction ctx with
  | nil => cases h
  | cons hd tl ih =>
      by_cases heq : name = hd.1
      · subst heq
        rw [List.kerase_cons_eq rfl]
        exact Context.mem_cons_ne hne h
      · rw [List.kerase_cons_ne heq]
        cases h with
        | head => exact .head _
        | tail _ h' => exact .tail _ (ih h')

/-- Membership in original context implies membership in permuted context
    (requires NodupKeys). -/
protected theorem mem_permute (ctx : Context α) (names : List String)
    (hperm : names.Perm ctx.keys) (hnd : ctx.NodupKeys)
    {x : String} {τ : α}
    (h : ⟨x, τ⟩ ∈ ctx) :
    ⟨x, τ⟩ ∈ ctx.permute names hperm := by
  have hdl : List.dlookup x ctx = some τ := List.mem_dlookup hnd h
  have h' := dlookup_permute_of_perm ctx names hperm x τ hdl
  exact List.of_mem_dlookup h'

/-- Extract the second component equality from a constant-family sigma equality. -/
protected theorem sigma_snd_eq {x y : String} {a b : α}
    (h : (⟨x, a⟩ : Σ _ : String, α) = ⟨y, b⟩) : a = b :=
  eq_of_heq (Sigma.mk.inj h).2

end Context

end Palladium.Util
