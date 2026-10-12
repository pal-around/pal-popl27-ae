import Palladium.PAL.Syntax
import Palladium.PAL.Denote
import Palladium.Compiler.Compile
import Palladium.Compiler.Encoding.EnvironmentIndices

namespace Palladium.Compiler

open Palladium.Util
open Palladium.PAL
open Palladium.TensorLib

/-!
## Bridge lemmas: relating indices across context operations

These lemmas connect `Environment.toIndices` across context operations:
- `Environment.toIndices` for `⟨x, .fin n⟩ :: rest` decomposes as
  `xVal.val :: toIndicesRaw rest (tailRandomEnv δ)`, definitionally.
- `Environment.toIndices_insert`: indices for `δ'.insert x D v h` equal
  `(Environment.toIndices δ' hnd').insertIdx idx v.val`

Key results:
- `σ₁ ≡ₑₓₜ σ₂ → idx(σ₁) = idx(σ₂)`
- `idx(Σ, σ'.insert(x, v)) = idx(kerase x Σ, σ').insertIdx(ax, v.val)`  ← key bridge
-/

end Palladium.Compiler

namespace Palladium.Util.Environment

open Palladium.Compiler
open Palladium.Util
open Palladium.PAL
open Palladium.TensorLib

/-- (∀ x D h, σ₁(x) = σ₂(x)) → idx(Σ, σ₁) = idx(Σ, σ₂). -/
theorem toIndices_ext :
    (ctx : RandomContext) →
    (δ₁ δ₂ : ⟦ctx⟧) →
    (∀ (name : String) (D : Discrete) (h : ⟨name, D⟩ ∈ ctx),
      δ₁ name D h = δ₂ name D h) →
    δ₁.toIndices = δ₂.toIndices
  | [], _, _, _ => rfl
  | ⟨y, .fin m hm⟩ :: rest, δ₁, δ₂, hext => by
      apply Indices.ext
      simp only [toIndices, toIndicesRaw]
      congr 1
      · exact congrArg Fin.val (hext y (.fin m hm) (.head _))
      · exact congrArg Indices.values
          (toIndices_ext rest _ _ (fun name D h_rest =>
            hext name D (.tail _ h_rest)))

/-- Σ₁ = Σ₂ → idx(Σ₁, σ₁).values = idx(Σ₂, σ₂).values when σ₁ ≡_ext σ₂. -/
private theorem toIndices_eq_of_ctx_eq
    {ctx₁ ctx₂ : RandomContext} (heq : ctx₁ = ctx₂)
    (δ₁ : RandomEnvironment ctx₁) (δ₂ : RandomEnvironment ctx₂)
    (henvEq : ∀ name D (h₁ : ⟨name, D⟩ ∈ ctx₁)
                        (h₂ : ⟨name, D⟩ ∈ ctx₂),
              δ₁ name D h₁ = δ₂ name D h₂) :
    (δ₁.toIndices).values = (δ₂.toIndices).values := by
  subst heq
  exact congrArg Indices.values
    (toIndices_ext ctx₁ δ₁ δ₂ (fun name D h => henvEq name D h h))

/-- Σ₁ = Σ₂ → σ(x) = (Σ₁=Σ₂ ▸ σ)(x). -/
private theorem env_transport_apply
    {ctx₁ ctx₂ : RandomContext} (heq : ctx₁ = ctx₂)
    (δ : RandomEnvironment ctx₁) (name : String) (D : Discrete)
    (h₁ : ⟨name, D⟩ ∈ ctx₁)
    (h₂ : ⟨name, D⟩ ∈ ctx₂) :
    δ name D h₁ = (heq ▸ δ) name D h₂ := by
  subst heq; rfl

/-- (a :: as).insertIdx(p+1, v) = a :: as.insertIdx(p, v). -/
private theorem insertIdx_cons_succ {α : Type} (x : α) (xs : List α) (p : Nat) (a : α) :
    (x :: xs).insertIdx (p + 1) a = x :: xs.insertIdx p a := by
  simp [List.insertIdx_succ_cons]

/-- Negate a sigma equality from a first-component inequality. -/
private theorem sigma_ne_of_fst_ne {x y : String} {a b : Discrete}
    (hne : x ≠ y) : ¬((⟨x, a⟩ : Σ _ : String, Discrete) = ⟨y, b⟩) :=
  fun heq => hne (congrArg Sigma.fst heq)

/-- **Key bridge**: idx(Σ, σ'.insert(x, v)).values = idx(Σ\x, σ').values.insertIdx(ax, v.val).
    Used by Marginalize.correctness. -/
theorem toIndices_insert
    (ctx : RandomContext) (x : String) (n : Nat) (hn : 0 < n)
    (h : ⟨x, .fin n hn⟩ ∈ ctx)
    (hnd : ctx.keys.Nodup)
    (hnd' : (List.kerase x ctx).keys.Nodup)
    (δ' : RandomEnvironment (List.kerase x ctx))
    (v : Fin n)
    (idx : Nat) (hIdxLt : idx < ctx.keys.length)
    (hKeysIdx : ctx.keys[idx]'hIdxLt = x) :
    ((δ'.insert x (.fin n hn) v h hnd).toIndices).values =
      (δ'.toIndices).values.insertIdx idx v.val := by
  induction ctx generalizing idx with
  | nil =>
    nomatch h
  | cons s rest ih =>
    obtain ⟨y, D⟩ := s
    cases D with
    | fin m hm =>
      by_cases hyx : y = x
      · -- Case y = x: the variable is at the head of the context
        subst hyx -- eliminates x, uses y everywhere
        have hmn : m = n := by
          have := List.NodupKeys.eq_of_mk_mem hnd (.head _) h
          cases this; rfl
        subst hmn
        simp only [List.keys, List.map, List.length_cons] at hIdxLt hKeysIdx ⊢
        have hidx0 : idx = 0 := by
          cases idx with
          | zero => rfl
          | succ k =>
            exfalso
            have hmem : y ∈ rest.keys := hKeysIdx ▸ List.getElem_mem ..
            exact (List.nodup_cons.mp hnd).1 hmem
        subst hidx0
        have hyNotIn : y ∉ rest.keys := (List.nodup_cons.mp hnd).1
        have hDelEq : List.kerase y (⟨y, Discrete.fin m hm⟩ :: rest : RandomContext) = rest := by
          simp [List.kerase_cons_eq]
        simp only [toIndices, toIndicesRaw, List.insertIdx_zero]
        congr 1
        · -- Head: insert at head gives v
          show (Environment.insert y δ' (Discrete.fin m hm) v h hnd
                  y (Discrete.fin m hm) _).val = v.val
          unfold Environment.insert
          simp
        · -- Tail: toIndices rest agrees
          exact toIndices_eq_of_ctx_eq hDelEq.symm _ _
            (fun name D h₁ h₂ => by
              have hne : name ≠ y :=
                fun heq => hyNotIn (heq ▸ List.mem_keys.mpr ⟨D, h₁⟩)
              show (Environment.insert y δ' (Discrete.fin m hm) v h hnd).tail name D h₁ =
                   δ' name D h₂
              simp only [Environment.tail]
              show (Environment.insert y δ' (Discrete.fin m hm) v h hnd name D _) = δ' name D h₂
              unfold Environment.insert
              simp only [dif_neg (sigma_ne_of_fst_ne (a := D) (b := Discrete.fin m hm) hne)])
      · -- Case y ≠ x: variable is in the rest
        have hfindRest : ⟨x, .fin n hn⟩ ∈ rest :=
          Context.mem_cons_ne (Ne.symm hyx) h
        simp only [List.keys, List.map, List.length_cons] at hIdxLt hKeysIdx ⊢
        have hidxPos : idx > 0 := by
          cases idx with
          | zero => exact absurd hKeysIdx hyx
          | succ k => omega
        obtain ⟨k, rfl⟩ : ∃ k, idx = k + 1 := ⟨idx - 1, by omega⟩
        have hkLt : k < (rest.keys).length := by simp only [List.keys] at hIdxLt ⊢; omega
        have hKeysRest : (rest.keys)[k]'hkLt = x := hKeysIdx
        have hndRest : (rest.keys).Nodup := (List.nodup_cons.mp hnd).2
        have hndRestDel : (List.kerase x rest).keys.Nodup :=
          (List.NodupKeys.kerase x hndRest)
        have hDelEq : List.kerase x (⟨y, Discrete.fin m hm⟩ :: rest : RandomContext) =
            (⟨y, Discrete.fin m hm⟩ :: List.kerase x rest : RandomContext) := by
          rw [List.kerase_cons_ne (h := fun h => hyx h.symm)]
        -- Transported δ' for the deleted context in cons form
        let δ'c := hDelEq ▸ δ'
        -- Transport RHS through hDelEq to get cons form (before unfolding)
        rw [toIndices_eq_of_ctx_eq hDelEq δ' δ'c
            (fun name D h₁ h₂ => env_transport_apply hDelEq δ' name D h₁ h₂)]
        -- Unfold toIndices on both sides
        simp only [toIndices, toIndicesRaw]
        -- Unfold insertIdx for succ case (now RHS is in cons form)
        rw [insertIdx_cons_succ]
        -- Goal: yVal_lhs :: restIdx = yVal_rhs :: delIdx.insertIdx k v.val
        congr 1
        · -- Head: y values agree
          show (Environment.insert x δ' (Discrete.fin n hn) v h hnd
                  y (Discrete.fin m hm) _).val =
               (δ'c y (Discrete.fin m hm) _).val
          unfold Environment.insert
          have hyx_sigma : ¬((⟨y, Discrete.fin m hm⟩ : Σ _ : String, Discrete) = ⟨x, Discrete.fin n hn⟩) :=
            sigma_ne_of_fst_ne hyx
          simp only [dif_neg hyx_sigma]
          congr 1
          exact env_transport_apply hDelEq δ' y (Discrete.fin m hm) _ _
        · -- Tail: restIdx = delIdx.insertIdx k v.val (by IH + ext)
          have ih_applied := ih hfindRest hndRest hndRestDel δ'c.tail k hkLt hKeysRest
          simp only [toIndices] at ih_applied
          rw [← ih_applied]
          exact congrArg Indices.values (toIndices_ext rest _ _ (fun name D h_rest => by
            have hny : name ≠ y :=
              fun heq => (List.nodup_cons.mp hnd).1
                (heq ▸ List.mem_keys.mpr ⟨D, h_rest⟩)
            by_cases hnx : name = x
            · -- name = x: both inserts give v
              have hxn := hnx.symm; subst hxn -- eliminates name, keeps x
              have hDn : D = Discrete.fin n hn :=
                List.NodupKeys.eq_of_mk_mem hndRest h_rest hfindRest
              subst hDn
              show (Environment.insert x δ' (Discrete.fin n hn) v h hnd).tail
                    x (Discrete.fin n hn) h_rest =
                   (Environment.insert x δ'c.tail (Discrete.fin n hn) v hfindRest hndRest)
                    x (Discrete.fin n hn) h_rest
              simp only [Environment.tail]
              unfold Environment.insert
              simp
            · -- name ≠ x: both delegate to underlying δ'
              show (Environment.insert x δ' (Discrete.fin n hn) v h hnd).tail
                    name D h_rest =
                   (Environment.insert x δ'c.tail (Discrete.fin n hn) v hfindRest hndRest)
                    name D h_rest
              simp only [Environment.tail]
              unfold Environment.insert
              simp only [dif_neg (sigma_ne_of_fst_ne (a := D) (b := Discrete.fin n hn) hnx)]
              exact env_transport_apply hDelEq δ' name D _ _))


/-- `toIndices` at position `j` equals `(δ name D hmem).val` where `name = ctx.keys[j]`. -/
private theorem toIndices_getElem :
    (ctx : RandomContext) → (hnd : ctx.keys.Nodup) → (δ : ⟦ctx⟧) →
    (j : Nat) → (hj : j < ctx.keys.length) →
    (name : String) → (n : Nat) → (hn : 0 < n) →
    (hkeys : ctx.keys[j]'hj = name) →
    (hmem : ⟨name, .fin n hn⟩ ∈ ctx) →
    (δ.toIndices).values[j]'(by rw [toIndices_length]; exact hj) =
      (δ name (.fin n hn) hmem).val
  | [], _, _, _, hj, _, _, _, _, _ => absurd hj (by simp [List.keys])
  | ⟨y, .fin m hm⟩ :: rest, hnd, δ, j, hj, name, n, hn, hkeys, hmem => by
      cases j with
      | zero =>
        simp only [List.keys, List.map, List.getElem_cons_zero] at hkeys
        subst hkeys
        have hmn : Discrete.fin m hm = Discrete.fin n hn :=
          List.NodupKeys.eq_of_mk_mem hnd (.head _) hmem
        cases hmn
        simp only [toIndices, toIndicesRaw, List.getElem_cons_zero]
      | succ k =>
        simp only [List.keys, List.map, List.length_cons, List.length_map] at hj hkeys
        have hk : k < rest.keys.length := by
          simp only [List.keys, List.length_map]; omega
        have hkeys' : rest.keys[k]'hk = name := hkeys
        have hne : name ≠ y := by
          intro heq; subst heq
          have : name ∈ rest.keys := hkeys' ▸ List.getElem_mem hk
          exact (List.nodup_cons.mp hnd).1 this
        have hmem_rest : ⟨name, .fin n hn⟩ ∈ rest := Context.mem_cons_ne hne hmem
        -- Unfold to raw list level
        show (δ.toIndices).values[k + 1]'(by rw [toIndices_length]; simp [List.keys] at hj ⊢; exact hj) =
             (δ name (.fin n hn) hmem).val
        simp only [toIndices, toIndicesRaw, List.getElem_cons_succ]
        -- Goal: (toIndicesRaw rest ...)[k] = (δ name ...).val
        -- IH gives: (toIndices rest ...).values[k] = (δ.tail name ...).val
        -- Since .values unfolds to toIndicesRaw, convert:
        have ih := toIndices_getElem rest (List.nodup_cons.mp hnd).2
          δ.tail k hk name n hn hkeys' hmem_rest
        simp only [toIndices] at ih
        rw [ih]; rfl

/-- When all elements of a list produce `some`, `filterMap` preserves length. -/
private theorem filterMap_length_of_forall_isSome' {f : α → Option β}
    {l : List α} (h : ∀ x ∈ l, (f x).isSome) :
    (l.filterMap f).length = l.length := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.filterMap_cons]
    have hx := h x (List.mem_cons_self ..)
    rw [Option.isSome_iff_exists] at hx
    obtain ⟨v, hv⟩ := hx
    simp only [hv, List.length_cons]
    exact congrArg (· + 1) (ih (fun a ha => h a (List.mem_cons_of_mem x ha)))

/-- When all elements produce `some`, `filterMap` at position `i` equals `(f l[i]).get`. -/
private theorem filterMap_getElem_of_forall_isSome {f : α → Option β} {l : List α}
    (hall : ∀ x ∈ l, (f x).isSome)
    (i : Nat) (hi : i < l.length) :
    have hlen : (l.filterMap f).length = l.length :=
      filterMap_length_of_forall_isSome' hall
    (l.filterMap f)[i]'(hlen ▸ hi) = (f l[i]).get (hall l[i] (List.getElem_mem hi)) := by
  induction l generalizing i with
  | nil => simp at hi
  | cons x xs ih =>
    have hx := hall x (List.mem_cons_self ..)
    rw [Option.isSome_iff_exists] at hx
    obtain ⟨v, hv⟩ := hx
    simp only [List.filterMap_cons, hv]
    cases i with
    | zero =>
      simp only [List.getElem_cons_zero]
      show v = (f x).get _
      simp [hv]
    | succ k =>
      simp only [List.getElem_cons_succ, List.length_cons] at hi ⊢
      exact ih (fun a ha => hall a (List.mem_cons_of_mem x ha)) k (by omega)

/-- `computeReturnPerm` at position `i` maps `returns[i]` to its index in `Δ.keys`:
    there exists `idx < Δ.keys.length` with `perm[i] = idx` and `Δ.keys[idx] = returns[i]`. -/
private theorem computeReturnPerm_spec
    (Δ : RandomContext) (returns : List String)
    (hperm : returns.Perm Δ.keys) (_hnd : Δ.keys.Nodup)
    (_hv : Permutation.Valid (Block.computeReturnPerm Δ returns)
      Δ.keys.length)
    (i : Nat) (hi : i < returns.length)
    (hi' : i < (Block.computeReturnPerm Δ returns).length) :
    ∃ (idx : Nat) (hidx : idx < Δ.keys.length),
      (Block.computeReturnPerm Δ returns)[i]'hi' = idx ∧
      Δ.keys[idx]'hidx = returns[i]'hi := by
  have hall : ∀ x ∈ returns, (Δ.keys.findIdx? (· == x)).isSome := by
    intro x hx
    rw [List.findIdx?_isSome]
    exact List.any_eq_true.mpr ⟨x, hperm.subset hx, beq_self_eq_true _⟩
  have hval := filterMap_getElem_of_forall_isSome hall i hi
  simp only [Block.computeReturnPerm] at hval ⊢
  have hany : (Δ.keys.findIdx? (· == returns[i]'hi)).isSome := hall _ (List.getElem_mem hi)
  rw [Option.isSome_iff_exists] at hany
  obtain ⟨idx, hidx⟩ := hany
  have ⟨hidx_lt, hpred, _⟩ := List.findIdx?_eq_some_iff_getElem.mp hidx
  have heq_name : Δ.keys[idx]'hidx_lt = returns[i]'hi := eq_of_beq hpred
  rw [hval]
  simp only [hidx, Option.get_some]
  exact ⟨idx, hidx_lt, rfl, heq_name⟩

/-- For each variable in `Δ`, there exists a membership proof in the permuted context. -/
private theorem mem_rctx_discrete
    (Δ : RandomContext) (hnd : Δ.keys.Nodup)
    (j : Nat) (hj : j < Δ.keys.length) :
    ∃ (n : Nat) (hn : 0 < n), ⟨Δ.keys[j]'hj, Discrete.fin n hn⟩ ∈ Δ := by
  induction Δ generalizing j with
  | nil => simp [List.keys] at hj
  | cons s rest ih =>
    obtain ⟨y, D⟩ := s
    cases D with
    | fin m hm =>
      cases j with
      | zero =>
        simp only [List.keys, List.map, List.getElem_cons_zero]
        exact ⟨m, hm, .head _⟩
      | succ k =>
        have hk : k < rest.keys.length := by
          have : rest.keys.length = rest.length := List.length_map ..
          simp only [List.keys, List.length_map, List.length_cons] at hj
          omega
        simp only [List.keys, List.map, List.getElem_cons_succ]
        obtain ⟨n, hn, hmem⟩ := ih (List.nodup_cons.mp hnd).2 k hk
        exact ⟨n, hn, .tail _ hmem⟩

/-- Indices of an ofPerm'd environment equal inverse-permuted indices of the original environment.
    Given `δ : ⟦Δ⟧` and `hperm : Δ.Perm Δ'`, the indices of `δ.ofPerm hperm : ⟦Δ'⟧`
    are the inverse-permutation of `δ`'s indices under `computeReturnPerm Δ' Δ.keys`. -/
theorem toIndices_ofPerm
    {Δ : RandomContext}
    (δ : ⟦Δ⟧)
    (Δ' : RandomContext)
    (hperm_ctx : Δ.Perm Δ')
    (hnd' : Δ'.NodupKeys)
    (hnd : Δ.keys.Nodup)
    (hv : Permutation.Valid (Block.computeReturnPerm Δ' Δ.keys)
      Δ'.toShape.rank) :
    let δ' := δ.ofPerm hperm_ctx
    let inv := (Block.computeReturnPerm Δ' Δ.keys).invert hv
    (δ'.toIndices).values =
      inv.map (fun p =>
        (δ.toIndices).values.getD p 0) := by
  -- Eliminate let bindings
  simp only
  -- Abbreviations
  let retPerm := Block.computeReturnPerm Δ' Δ.keys
  let inv := retPerm.invert hv
  let δ' := δ.ofPerm hperm_ctx
  let origIndices := δ.toIndices
  let hperm_keys : Δ.keys.Perm Δ'.keys := hperm_ctx.map Sigma.fst
  -- The valid permutation over keys length
  have hv_keys : Permutation.Valid retPerm Δ'.keys.length := by
    rwa [Δ'.toShape_length] at hv
  -- Both sides have the same length
  have lhs_len : (δ'.toIndices).values.length = Δ'.keys.length :=
    toIndices_length δ'
  have inv_len : inv.length = Δ'.keys.length := by
    show (retPerm.invert hv).length = Δ'.keys.length
    rw [Permutation.Valid.invert_length]; rw [Δ'.toShape_length]
  -- original indices length
  have hoi_len : origIndices.values.length = Δ.keys.length :=
    toIndices_length δ
  -- Use List.ext_getElem for element-wise equality
  apply List.ext_getElem
  · rw [lhs_len, List.length_map, inv_len]
  · intro j hj_lhs hj_rhs
    -- j < Δ'.keys.length
    have hj_keys : j < Δ'.keys.length := lhs_len ▸ hj_lhs
    -- Get the variable name and discrete type at position j in Δ'
    obtain ⟨n, hn, hmem_j⟩ := mem_rctx_discrete Δ' hnd' j hj_keys
    -- LHS = (δ' name (.fin n hn) hmem_j).val
    have h_lhs := toIndices_getElem Δ' hnd' δ' j hj_keys
      (Δ'.keys[j]'hj_keys) n hn rfl hmem_j
    -- RHS at position j: (inv.map f)[j] = f (inv[j])
    have hinv_lt : j < inv.length := inv_len ▸ hj_keys
    simp only [List.getElem_map]
    -- inv[j] < toShape.rank (needed for invert_spec)
    have hj_toShape : j < Δ'.toShape.rank := by
      rw [Δ'.toShape_length]; exact hj_keys
    -- By invert_spec: perm[inv[j]] = j
    have h_inv_spec := Permutation.Valid.invert_spec hv j hj_toShape
    -- inv[j] < toShape.rank (from invert_forall_lt)
    have hinv_val_lt_ts : inv[j]'hinv_lt < Δ'.toShape.rank :=
      hv.invert_forall_lt (inv[j]'hinv_lt) (List.getElem_mem hinv_lt)
    -- inv[j] < Δ.keys.length
    have hinv_val_lt : inv[j]'hinv_lt < Δ.keys.length := by
      rw [hperm_keys.length_eq]
      rwa [Δ'.toShape_length] at hinv_val_lt_ts
    -- inv[j] < origIndices.values.length
    have hinv_val_lt_oi : inv[j]'hinv_lt < origIndices.values.length := hoi_len ▸ hinv_val_lt
    -- Unfold getD to getElem using ?-based reasoning
    have h_getD : origIndices.values.getD (inv[j]'hinv_lt) 0 = origIndices.values[inv[j]'hinv_lt]'hinv_val_lt_oi := by
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hinv_val_lt_oi, Option.getD_some]
    rw [h_lhs, h_getD]
    -- inv[j] < retPerm.length
    have hinv_val_lt_perm : inv[j]'hinv_lt < retPerm.length := by
      rw [hv_keys.length_eq]; rwa [Δ'.toShape_length] at hinv_val_lt_ts
    -- Use computeReturnPerm_spec at index inv[j]
    obtain ⟨idx, hidx_lt, h_perm_eq, hkeys_eq⟩ := computeReturnPerm_spec Δ' Δ.keys hperm_keys hnd'
      hv_keys (inv[j]'hinv_lt) hinv_val_lt hinv_val_lt_perm
    have h_perm_getElem : retPerm[inv[j]'hinv_lt]'hinv_val_lt_perm = j := h_inv_spec
    -- Therefore idx = j; eliminate idx
    have : j = idx := (h_perm_eq ▸ h_perm_getElem).symm
    subst this
    -- Now hkeys_eq : Δ'.keys[j] = Δ.keys[inv[j]]
    -- Membership in Δ (output context)
    have hmem_out : ⟨Δ'.keys[j]'hj_keys, Discrete.fin n hn⟩ ∈ Δ :=
      hperm_ctx.mem_iff.mpr hmem_j
    -- inv[j] is valid as output context key index
    have hinv_val_lt_ctx : inv[j]'hinv_lt < Δ.keys.length := hinv_val_lt
    -- output keys at inv[j] = Δ'.keys[j]
    have hkeys_at : Δ.keys[inv[j]'hinv_lt]'hinv_val_lt_ctx =
        Δ'.keys[j]'hj_keys := hkeys_eq.symm
    -- RHS = (δ name (.fin n hn) hmem_out).val
    have h_rhs := toIndices_getElem Δ hnd δ
      (inv[j]'hinv_lt) hinv_val_lt_ctx
      (Δ'.keys[j]'hj_keys) n hn hkeys_at hmem_out
    rw [h_rhs]
    -- Goal: (δ.ofPerm hperm_ctx ... hmem_j).val = (δ ... hmem_out).val
    -- By definition of ofPerm: rfl
    rfl

end Palladium.Util.Environment
