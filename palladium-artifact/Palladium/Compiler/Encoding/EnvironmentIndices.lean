import Palladium.PAL.Syntax
import Palladium.PAL.Denote
import Palladium.Compiler.Compile
import Palladium.Compiler.Encoding.ContextShape
import TensorLib.Indices

namespace Palladium.Compiler

open Palladium.Util
open Palladium.PAL

/-!
## Building environments from indices

To relate tensor entries to PMF values, we need to construct a random
environment from a list of indices into the tensor.

With the newest-first convention, the first index in the list corresponds
to the newest variable (head of the context).

Key equations:
- `Indices.toRandomEnvironment(∅, []) = ∅`
- `Indices.toRandomEnvironment(x:fin(n) · Σ, idx) = σ[x ↦ idx.head] ∪ Indices.toRandomEnvironment(Σ, idx.tail)`
-/

/-- Total reconstruction of a random environment from proof-carrying indices. -/
def Indices.toRandomEnvironment :
    (ctx : RandomContext) → Palladium.TensorLib.Indices ↑ctx → ⟦ctx⟧
  | [], _ => fun _ _ h => nomatch h
  | ⟨x, .fin n hn⟩ :: rest, indices =>
      let i := indices.head
      fun name D h =>
        if heq : (⟨name, D⟩ : Σ _ : String, Discrete) = ⟨x, .fin n hn⟩ then
          (Context.sigma_snd_eq heq) ▸ i
        else
          (Indices.toRandomEnvironment rest indices.tail) name D (by
            cases h with
            | head => exact absurd rfl heq
            | tail _ h' => exact h')

/-!
## Converting random environments to indices

To index into a tensor using a random environment, we extract the index
for each random variable from the environment. This is the inverse of
`Indices.toRandomEnvironment`.

Indices are ordered newest-first (matching `RandomContext.toShape`): the head
index corresponds to the newest variable (head of context).

Key equations:
- `idx(σ) : RandomEnvironment Σ → Indices (toShape Σ)`
- `idx(σ).values[i] = σ(keys(Σ)[i]).val`
- `Indices.toRandomEnvironment(Σ, idx(σ)) = σ`  (round-trip)
-/

/-- σ|_Σ : Restrict σ from x:fin(n) · Σ to Σ. Requires Nodup to exclude shadowing. -/
def tailRandomEnv {x : String} {n : Nat} {hn : 0 < n} {rest : RandomContext}
    (hnd : (x :: rest.keys).Nodup)
    (δ : ⟦⟨x, .fin n hn⟩ :: rest⟧) : ⟦rest⟧ :=
  fun name D h =>
    if hne : name = x then
      absurd (hne ▸ (List.mem_keys.mpr ⟨D, h⟩))
        ((List.nodup_cons.mp hnd).1)
    else
      δ name D (.tail _ h)

end Palladium.Compiler

namespace Palladium.Util.Environment

open Palladium.Compiler
open Palladium.Util
open Palladium.PAL
open Palladium.TensorLib

/-- Raw index list extraction (internal). Returns the raw `List Nat`; use `Environment.toIndices`
    for the proof-carrying version. -/
def toIndicesRaw : (ctx : RandomContext) → ⟦ctx⟧ → List Nat
  | [], _ => []
  | ⟨x, .fin n hn⟩ :: _, δ =>
      let xVal : Fin n := δ x (.fin n hn) (.head _)
      xVal.val :: toIndicesRaw _ δ.tail

theorem toIndicesRaw_length : (ctx : RandomContext) →
    (δ : ⟦ctx⟧) →
    (toIndicesRaw ctx δ).length = ctx.keys.length
  | [], _ => rfl
  | ⟨_x, .fin _n _⟩ :: _, δ => by
      simp only [toIndicesRaw, List.keys, List.map, List.length_cons]
      exact congrArg (· + 1) (toIndicesRaw_length _ δ.tail)

theorem toIndicesRaw_valid :
    (ctx : RandomContext) → (δ : ⟦ctx⟧) →
    ∀ j (hj : j < ctx.toShape.rank),
      (toIndicesRaw ctx δ)[j]'(by rw [toIndicesRaw_length]; rw [ctx.toShape_length] at hj; exact hj) <
      ctx.toShape[j]'hj
  | [], _, _, hj => absurd hj (by simp [RandomContext.toShape])
  | ⟨x, .fin n hn⟩ :: _, δ, j, hj => by
      let xVal : Fin n := δ x (.fin n hn) (.head _)
      cases j with
      | zero =>
        simp only [toIndicesRaw, RandomContext.toShape, List.getElem_cons_zero]
        exact xVal.isLt
      | succ j' =>
        simp only [toIndicesRaw, RandomContext.toShape, List.getElem_cons_succ,
                    List.length_cons, Shape.rank] at hj ⊢
        exact toIndicesRaw_valid _ δ.tail j' (by
          show j' < (RandomContext.toShape _).length; omega)

/-- idx : RandomEnvironment Σ → Indices (toShape Σ). Proof-carrying index extraction.
    Newest-first: idx(x:fin(n) · Σ, σ) has σ(x).val :: idx(Σ, σ|_Σ). -/
def toIndices {ctx : RandomContext} (δ : ⟦ctx⟧) :
    Indices ↑ctx :=
  ⟨toIndicesRaw ctx δ,
   by rw [toIndicesRaw_length, ctx.toShape_length],
   toIndicesRaw_valid ctx δ⟩

/-- `Coe` instance for ergonomic coercion from `RandomEnvironment` to `Indices`. -/
instance {ctx : RandomContext} : Coe ⟦ctx⟧ (Indices ↑ctx) where
  coe := toIndices

/-- Direct bracket indexing of tensors by random environments.
    `tensor[δ]` stands for `tensor[δ.toIndices]`. -/
instance {L : Tensor.Layout} [Tensor.Operations L] {ctx : RandomContext} {α : Type} :
    GetElem (L ↑ctx α) ⟦ctx⟧ α (fun _ _ => True) where
  getElem tensor δ _ := tensor[δ.toIndices]

/-- `tensor[δ]` reduces to `tensor[δ.toIndices]`. -/
@[simp]
theorem getElem_renv {L : Tensor.Layout} [Tensor.Operations L]
    {ctx : RandomContext} {α : Type}
    (tensor : L ↑ctx α)
    (δ : ⟦ctx⟧) (h : True := trivial) :
    tensor[δ] = tensor[δ.toIndices] := rfl

/-- The raw values of `toIndices` are `toIndicesRaw`. -/
theorem toIndices_values
    {ctx : RandomContext} (δ : ⟦ctx⟧) :
    (δ.toIndices).values = toIndicesRaw ctx δ := rfl

/-- Backward-compat: length of the raw values. -/
theorem toIndices_length {ctx : RandomContext}
    (δ : ⟦ctx⟧) :
    (δ.toIndices).values.length = ctx.keys.length :=
  toIndicesRaw_length ctx δ

/-- Round-trip: Indices.toRandomEnvironment(Σ, idx(σ)) = σ. -/
theorem toIndices_toRandomEnvironment :
    (ctx : RandomContext) → (δ : ⟦ctx⟧) →
    Palladium.Compiler.Indices.toRandomEnvironment ctx (δ.toIndices) = δ
  | [], _ => by
    simp only [Palladium.Compiler.Indices.toRandomEnvironment]
    funext name D h; exact nomatch h
  | ⟨x, .fin n hn⟩ :: rest, δ => by
    simp only [Palladium.Compiler.Indices.toRandomEnvironment, toIndices, Palladium.TensorLib.Indices.head, Palladium.TensorLib.Indices.tail]
    funext name D h
    split
    · rename_i heq
      have hD' := Context.sigma_snd_eq heq
      have hname' := congrArg Sigma.fst heq
      subst hname'; subst hD'
      simp only [toIndicesRaw, List.head_cons]
    · rename_i hne
      have ih := toIndices_toRandomEnvironment rest δ.tail
      have hmem : ⟨name, D⟩ ∈ rest := by
        cases h with
        | head => exact absurd rfl hne
        | tail _ h' => exact h'
      have : (Palladium.Compiler.Indices.toRandomEnvironment rest (δ.tail.toIndices)) name D hmem =
        δ name D h := by
        rw [ih]; rfl
      exact this

/-- `toIndicesRaw` is invariant under `renameByTypes`: renaming an environment
    by positional type matching does not change the extracted index values.
    Requires `NodupKeys` on the target context to ensure `cons`/`tail`
    interactions are well-behaved. -/
theorem toIndicesRaw_renameByTypes :
    (ctx₁ ctx₂ : RandomContext) →
    (htypes : ctx₁.map Sigma.snd = ctx₂.map Sigma.snd) →
    (hnd₂ : ctx₂.NodupKeys) →
    (δ : ⟦ctx₁⟧) →
    toIndicesRaw ctx₂ (δ.renameByTypes htypes) = toIndicesRaw ctx₁ δ
  | [], [], _, _, _ => rfl
  | ⟨x₁, .fin n₁ hn₁⟩ :: rest₁, ⟨x₂, .fin n₂ hn₂⟩ :: rest₂, htypes, hnd₂, δ => by
    have hn_eq : n₁ = n₂ := by
      have := congrArg (fun l => l.head?) htypes; simp at this; exact this
    subst hn_eq
    -- After subst, hn₁ and hn₂ are proofs of the same Prop (0 < n₂)
    -- so Discrete.fin n₂ hn₁ = Discrete.fin n₂ hn₂ by proof irrelevance
    have htypes_rest : rest₁.map Sigma.snd = rest₂.map Sigma.snd := by
      have := congrArg List.tail htypes; simp at this; exact this
    have hnd₂_rest : rest₂.NodupKeys :=
      (List.nodup_cons.mp hnd₂).2
    simp only [toIndicesRaw]
    congr 1
    · -- Head: show (renamed env).get x₂ (.head _) has same .val as δ.get x₁ (.head _)
      -- After subst and simp, the goal becomes:
      -- (cons ... x₂ D headVal) x₂ D (.head _) = δ x₁ D (.head _)
      -- cons at .head _ returns headVal, which is (h_eq ▸ δ.get x₁ (.head _))
      -- With proof irrelevance, h_eq is rfl, so headVal = δ.get x₁ (.head _)
      simp only [Environment.renameByTypes, Environment.renameByTypes.renameByTypesGo,
        Environment.get, Environment.cons]
      simp
    · -- Tail: inductive step
      -- Need: toIndicesRaw rest₂ ((δ.renameByTypes htypes).tail) = toIndicesRaw rest₁ (δ.tail)
      -- Since renameByTypes builds (restEnv.cons x₂ D headVal), and x₂ ∉ rest₂.keys (NodupKeys),
      -- tail accesses go through to restEnv = δ.tail.renameByTypes htypes_rest
      have h_tail : ∀ (name : String) (D : Discrete) (h : ⟨name, D⟩ ∈ rest₂),
          (δ.renameByTypes htypes).tail name D h =
          (δ.tail.renameByTypes htypes_rest) name D h := by
        intro name D hmem
        -- (δ.renameByTypes htypes).tail name D hmem
        -- = (δ.renameByTypes htypes) name D (.tail _ hmem)
        -- = (restEnv.cons x₂ _ headVal) name D (.tail _ hmem)
        -- Since name ∈ rest₂.keys and x₂ ∉ rest₂.keys (NodupKeys), name ≠ x₂
        -- So cons falls through to restEnv name D hmem
        simp only [Environment.tail, Environment.renameByTypes,
          Environment.renameByTypes.renameByTypesGo]
        have hne : (⟨name, D⟩ : Σ _ : String, Discrete) ≠ ⟨x₂, Discrete.fin n₁ hn₂⟩ := by
          intro heq
          have : name = x₂ := congrArg Sigma.fst heq
          subst this
          exact (List.nodup_cons.mp hnd₂).1 (List.mem_keys.mpr ⟨D, hmem⟩)
        simp only [Environment.cons, dif_neg hne]
      -- Use toIndices_ext to reduce to pointwise equality, then apply h_tail
      have := toIndicesRaw_renameByTypes rest₁ rest₂ htypes_rest hnd₂_rest δ.tail
      -- We need: toIndicesRaw rest₂ ((δ.renameByTypes htypes).tail) = toIndicesRaw rest₁ δ.tail
      -- We know: toIndicesRaw rest₂ (δ.tail.renameByTypes htypes_rest) = toIndicesRaw rest₁ δ.tail
      -- So we need: toIndicesRaw rest₂ ((δ.renameByTypes htypes).tail) =
      --             toIndicesRaw rest₂ (δ.tail.renameByTypes htypes_rest)
      -- By h_tail, the two environments agree on all membership proofs
      rw [← this]
      exact congrArg (toIndicesRaw rest₂) (funext fun name => funext fun D => funext fun h =>
        h_tail name D h)

end Palladium.Util.Environment
