import Palladium.PAL.Operators

namespace Palladium.PAL

open Palladium.Util

section
variable {Θ : StaticContext}

--  Typing judgment for expressions:
--  Θ; Σ ⊢ e : T
--  Θ = StaticContext, Σ = RandomContext
inductive Expr (Δ : RandomContext) : Ty -> Type
  --  i : Fin n
  --  ────────────────────────────────  (FIN)
  --  Θ; Σ ⊢ i : disc (fin n)
  /-- Finite-type literal value. -/
  | fin : {hn : 0 < n} → Fin n → Expr Δ (Ty.disc (Discrete.fin n hn))
  --  ────────────────────────────────  (FLOAT)
  --  Θ; Σ ⊢ r : float
  /-- Floating-point literal. -/
  | float : (r : Float) → Expr Δ Ty.float
  --  Σ(x) = D
  --  ─────────────────────────────────  (VAR)
  --  Θ; Σ ⊢ x : disc D
  /-- Random variable reference. -/
  | var : (x : String) → (D : Discrete) →
      (h : ⟨x, D⟩ ∈ Δ) → Expr Δ (Ty.disc D)
  --  Θ(x) = T
  --  ─────────────────────────────────  (PARAM)
  --  Θ; Σ ⊢ x : T
  /-- Static parameter reference. -/
  | param : (x : String) → (T : Ty) →
      (h : ⟨x, T⟩ ∈ Θ) → Expr Δ T
  --  Θ; Σ ⊢ e : T    UOp.Valid op T Tₒ
  --  ──────────────────────────────────────  (UOP)
  --  Θ; Σ ⊢ op e : Tₒ
  /-- Unary operator application. -/
  | uop : (op : UOp) → (e : Expr Δ T) → (hv : UOp.Valid op T Tₒ) →
      Expr Δ Tₒ
  --  Θ; Σ ⊢ a : T₁    Θ; Σ ⊢ b : T₂
  --  BOp.Valid op T₁ T₂ Tₒ
  --  ──────────────────────────────────────  (BOP)
  --  Θ; Σ ⊢ a op b : Tₒ
  /-- Binary operator application. -/
  | bop : (op : BOp) → (a : Expr Δ T₁) → (b : Expr Δ T₂) →
      (hv : BOp.Valid op T₁ T₂ Tₒ) →
      Expr Δ Tₒ
  --  Θ; Σ ⊢ c : bool    Θ; Σ ⊢ t : T₁    Θ; Σ ⊢ e : T₂
  --  ─────────────────────────────────────────────────────  (ITE)
  --  Θ; Σ ⊢ ite c t e : T₁ ⊔ T₂
  /-- Conditional (if-then-else) expression. -/
  | ite : (c : Expr Δ Ty.bool) → (t : Expr Δ T₁) → (e : Expr Δ T₂) →
      Expr Δ (T₁ ⊔ T₂)
deriving Repr

end

section
variable {Θ : ValidStaticContext}

/-- Derive a block body's internal combined-context well-formedness from the
    externally declared output context and the return permutation. -/
def blockInternalNodup {Δ' Δ_ret_var Δ_ret : WellFormedRandomContext}
    (hperm : Δ_ret_var.Perm ↑Δ_ret)
    (hwf_ext : (↑Δ_ret ++ ↑Δ' : RandomContext).NodupKeys) :
    (↑Δ_ret_var ++ ↑Δ' : RandomContext).NodupKeys :=
  (List.perm_nodupKeys (List.Perm.append_right _ hperm)).mpr hwf_ext

--  Θ; x⃗:D⃗, Σ ⊢ e : float
--  ────────────────────────────────────  (WPP)
--  Θ; Σ; x⃗:D⃗ ⊢ wpp e
/-- Weight program for a joint conditional distribution (rule P-Sample). Wraps a `float`
    expression in the extended context `vars ++ Δ`, whose values are normalized over all
    valuations of `vars` to produce a conditional distribution `Pr(x⃗ | δ)`. -/
inductive Wppc (Δ : WellFormedRandomContext) (vars : WellFormedRandomContext)
    (hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys) : Type
  | wpp : Expr (Θ := ↑Θ) (↑vars ++ ↑Δ) Ty.float → Wppc Δ vars hwf

instance : Repr (Wppc (Θ := Θ) Δ vars hwf) where
  reprPrec w p := match w with
    | .wpp e => Repr.addAppParen ("Wppc.wpp " ++ reprPrec e 1) p

--  Typing judgment for statements:
--  Θ; Σ ⊢ s ⊣ Σ'
--  Θ = ValidStaticContext, Σ = WellFormedRandomContext
--
--  Includes sequencing (nil/seq) and branching (ite) constructors,
--  so there is no separate Stmts type. `Stmt` and `Block` are mutually
--  inductive so that branch bodies in `ite` can be stored as `Block`s.
mutual
inductive Stmt : WellFormedRandomContext -> WellFormedRandomContext -> Type
  --  ───────────────────  (NIL)
  --  Θ; Σ ⊢ [] ⊣ Σ
  /-- Empty statement (identity on the random context). -/
  | nil (Δ : WellFormedRandomContext) : Stmt Δ Δ
  --  Θ; Σ ⊢ s₁ ⊣ Σ'    Θ; Σ' ⊢ s₂ ⊣ Σ''
  --  ──────────────────────────────────────  (SEQ)
  --  Θ; Σ ⊢ s₁; s₂ ⊣ Σ''
  /-- Sequence two statements. -/
  | seq {Δ Δ' Δ'' : WellFormedRandomContext}
      (first : Stmt Δ Δ') (rest : Stmt Δ' Δ'') :
      Stmt Δ Δ''
  --  x⃗ fresh for Σ    Θ; Σ; x⃗:D⃗ ⊢ wppc
  --  ───────────────────────────────────────────────  (SAMPLE)
  --  Θ; Σ ⊢ x⃗ ← sample wppc ⊣ x⃗:D⃗, Σ
  /-- Jointly sample the fresh random variables `vars` using a weight program for their
      conditional distribution. -/
  | sample (vars : WellFormedRandomContext)
      (hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys)
      (wppc : Wppc (Θ := Θ) Δ vars hwf) :
      Stmt Δ ⟨↑vars ++ ↑Δ, hwf⟩
  --  Σ(x) = D
  --  ─────────────────────────────────  (MARGINALIZE)
  --  Θ; Σ ⊢ marginalize x ⊣ Σ \ x
  /-- Marginalize (sum out) a random variable. -/
  | marginalize (x : String) (D : Discrete)
      (h : ⟨x, D⟩ ∈ Δ) :
      Stmt Δ (Δ.kerase x)
  --  Θ; Σ ⊢ c : Bool
  --  ─────────────────────────────  (CONDITION)
  --  Θ; Σ ⊢ condition c ⊣ Σ
  /-- Bayesian conditioning: keep the mass where `c` holds and set it to zero elsewhere. -/
  | condition {rnd : WellFormedRandomContext} (c : Expr (Θ := Θ) ↑rnd Ty.bool) :
      Stmt rnd rnd
  --  P-If:
  --  Θ ∣ Δ ⊢ E : B    Θ ∣ Δ ⊢ B₁ ⊣ Δ' ∣ D⃗    Θ ∣ Δ ⊢ B₂ ⊣ Δ' ∣ D⃗
  --  ──────────────────────────────────────────────────────────────────
  --  Θ ∣ Δ ⊢ x⃗ ← if E then B₁ else B₂ ⊣ x⃗:D⃗ ++ Δ'
  /-- Stochastic branching: split the distribution by condition, execute both
      branches independently, and merge the results. Each branch is a `Block`
      producing the same carry context `Δ'`. The branches may differ in the
      order/names of their declared return contexts (`Δ_ret_then` vs
      `Δ_ret_else`), but `htypes_then`/`htypes_else` enforce that both align
      with the user-declared `lhs` at the type level (same list of types
      after stripping names). After the `ite`, each branch's returned
      variables are renamed to the names in `lhs` and reordered to match
      `lhs`'s declaration order, so the output context is `Δ' ++ lhs`. -/
  | ite {Δ Δ' Δ_ret_then Δ_ret_else: WellFormedRandomContext}
      (lhs : WellFormedRandomContext)
      (cond : Expr (Θ := ↑Θ) ↑Δ Ty.bool)
      (then_block : Block Δ Δ' Δ_ret_then)
      (else_block : Block Δ Δ' Δ_ret_else)
      (htypes_then : (↑lhs : RandomContext).map Sigma.snd =
          (↑Δ_ret_then : RandomContext).map Sigma.snd)
      (htypes_else : (↑lhs : RandomContext).map Sigma.snd =
          (↑Δ_ret_else : RandomContext).map Sigma.snd)
      (hwf : (↑lhs ++ ↑Δ' : RandomContext).NodupKeys) :
      Stmt Δ ⟨↑lhs ++ ↑Δ', hwf⟩
  --  REPEAT (stochastic count):
  --  Θ ∣ Δ ⊢ n : Fin c
  --  Θ ∣ Δ                    ⊢ initBlock ⊣ Δ_loop_start | Δ_init_ret
  --  Θ ∣ vars ++ Δ_loop_start ⊢ loopBlock ⊣ Δ_loop_start | Δ_body_ret
  --  types(vars) = types(Δ_init_ret)
  --  types(vars) = types(Δ_body_ret)
  --  ──────────────────────────────────────────────────────────────────
  --  Θ ∣ Δ ⊢ vars ← repeat n init { initBlock } do { loopBlock } ⊣ vars ++ Δ_loop_start
  --
  --  The count `n` is a stochastic expression of type `Fin c` over the loop's
  --  input context `Δ`. The loop runs a random number of times: the result is the
  --  mixture, over `k = 0..c-1`, of "run the body `k` times on the slice of the
  --  input distribution where `n = k`" (see `Stmt.eval`).
  --
  --  The init block creates the first loop state, renamed to `vars`. Each loop
  --  iteration consumes `vars ++ Δ_loop_start`, runs the body block, and renames
  --  the body return back to `vars` for the next iteration.
  /-- Stochastic bounded iteration with positional type matching for init/body
      outputs. The iteration count `n : Fin c` is evaluated over the input context
      `Δ`; semantics is the mixture of the loop unrolled to counts `0..c-1`. The
      user-facing `vars` names are the loop-state names visible inside the body;
      init/body block return names may differ from `vars` (and from each other) but
      must each match `vars` by type. -/
  | repeat {Δ Δ_loop_start Δ_init_ret Δ_body_ret : WellFormedRandomContext}
      (vars : WellFormedRandomContext)
      (hwf : (↑vars ++ ↑Δ_loop_start : RandomContext).NodupKeys)
      {c : Nat} {hc : 0 < c}
      (n : Expr (Θ := ↑Θ) ↑Δ (Ty.disc (Discrete.fin c hc)))
      (initBlock : Block Δ Δ_loop_start Δ_init_ret)
      (loopBlock : Block ⟨↑vars ++ ↑Δ_loop_start, hwf⟩
        Δ_loop_start Δ_body_ret)
      (htypes_init : (↑vars : RandomContext).map Sigma.snd =
          (↑Δ_init_ret : RandomContext).map Sigma.snd)
      (htypes_body : (↑vars : RandomContext).map Sigma.snd =
          (↑Δ_body_ret : RandomContext).map Sigma.snd) :
      Stmt Δ ⟨↑vars ++ ↑Δ_loop_start, hwf⟩

--  Θ; Δ ⊢ s ⊣ Δ''    Δ' ~ Δ''
--  ──────────────────────────────────────  (BLOCK)
--  block(Θ, s) : ℙ⟦Δ⟧ → ℙ⟦Δ'⟧
/-- Anonymous block: compiles to a JAX expression (no function wrapper).
    Takes an input random context `Δ`, produces a "carry" output context
    `Δ'` (kept alive after the block), and declares a "return" context
    `Δ_ret` (what the caller treats as returned). The body must end in a
    context that splits as `Δ_ret_var ++ Δ'` for some `Δ_ret_var`. The
    actual returned portion `Δ_ret_var` permutes to the declared `Δ_ret`
    (witnessed by `hperm`), and the externally declared output context
    `Δ_ret ++ Δ'` is well formed (witnessed by `hwf_ext`). The body's
    internal well-formedness is derived from these two facts. The
    return-on-the-left order matches `sample`'s prepend convention. `Δ`,
    `Δ'`, and `Δ_ret` are first-class indices of `Block`. -/
inductive Block : WellFormedRandomContext -> WellFormedRandomContext -> WellFormedRandomContext -> Type where
  | mk
      {Δ Δ' Δ_ret_var Δ_ret : WellFormedRandomContext}
      (hperm : Δ_ret_var.Perm ↑Δ_ret)
      (hwf_ext : (↑Δ_ret ++ ↑Δ' : RandomContext).NodupKeys)
      (body : Stmt Δ ⟨↑Δ_ret_var ++ ↑Δ', blockInternalNodup hperm hwf_ext⟩) :
      Block Δ Δ' Δ_ret
end

deriving instance Repr for Stmt, Block

namespace Block

/-- The externally declared output context of a block: returned variables
    followed by the carry context. -/
def outputContext {Δ Δ' Δ_ret : WellFormedRandomContext}
    (blk : Block (Θ := Θ) Δ Δ' Δ_ret) : WellFormedRandomContext :=
  match blk with
  | Block.mk _ hwf_ext _ => ⟨↑Δ_ret ++ ↑Δ', hwf_ext⟩

private theorem wfc_ext_ctx {a b : WellFormedRandomContext}
    (h : (↑a : RandomContext) = ↑b) : a = b := by
  cases a
  cases b
  cases h
  rfl

/-- The prefix of a block body's output context before the carry suffix. -/
def returnPrefix (bodyCtx carryCtx : WellFormedRandomContext) : WellFormedRandomContext :=
  ⟨(↑bodyCtx : RandomContext).take ((↑bodyCtx : RandomContext).length -
      (↑carryCtx : RandomContext).length),
    List.NodupKeys.sublist (List.take_sublist _ _) bodyCtx.wf⟩

private theorem returnPrefix_empty (bodyCtx : WellFormedRandomContext) :
    returnPrefix bodyCtx (∅ : WellFormedRandomContext) = bodyCtx := by
  apply wfc_ext_ctx
  change (↑bodyCtx : RandomContext).take
      ((↑bodyCtx : RandomContext).length - ([] : RandomContext).length) =
    (↑bodyCtx : RandomContext)
  simp

/-- Build a block from a statement body whose output splits as return prefix
    followed by carry context. The return prefix is then permuted to the
    declared return context. -/
def withCarry {Δ Δ_body Δ' Δ_ret : WellFormedRandomContext}
    (body : Stmt (Θ := Θ) Δ Δ_body)
    (hctx : (↑Δ_body : RandomContext) = ↑(returnPrefix Δ_body Δ') ++ ↑Δ')
    (hperm : (returnPrefix Δ_body Δ').Perm ↑Δ_ret) :
    Block (Θ := Θ) Δ Δ' Δ_ret :=
  let retCtx := returnPrefix Δ_body Δ'
  have hwf_int : (↑retCtx ++ ↑Δ' : RandomContext).NodupKeys := by
    rw [← hctx]
    exact Δ_body.wf
  have hwf_ext : (↑Δ_ret ++ ↑Δ' : RandomContext).NodupKeys :=
    (List.perm_nodupKeys (List.Perm.append_right _ hperm)).mp hwf_int
  let bodyCtx : WellFormedRandomContext :=
    ⟨↑retCtx ++ ↑Δ', blockInternalNodup hperm hwf_ext⟩
  have hbody : Δ_body = bodyCtx := by
    apply wfc_ext_ctx
    exact hctx
  let body' : Stmt (Θ := Θ) Δ bodyCtx := hbody ▸ body
  Block.mk (Δ_ret_var := retCtx) hperm hwf_ext body'

/-- Build a block with no carry context from a statement body.
    The body determines the internal return context, which is then permuted
    to the declared return context. -/
def emptyInitialContext {Δ Δ_body Δ_ret : WellFormedRandomContext}
    (body : Stmt (Θ := Θ) Δ Δ_body)
    (hperm : Δ_body.Perm ↑Δ_ret) :
    Block (Θ := Θ) Δ (∅ : WellFormedRandomContext) Δ_ret :=
  withCarry body (by
    rw [returnPrefix_empty]
    change (↑Δ_body : RandomContext) = (↑Δ_body : RandomContext) ++ ([] : RandomContext)
    simp)
    (by
      rw [returnPrefix_empty]
      change (↑Δ_body : RandomContext).Perm ↑Δ_ret
      exact hperm)

/-- Lift a return-context types-equality to the full output context
    `lhs ++ Δ'` vs `Δ_ret ++ Δ'`. Used by `Block.evalAs`/`Block.compileAs`
    and their correctness theorem to share a single proof term — keeping
    the `▸` casts and `renameByTypes` calls on both sides definitionally
    equal under proof irrelevance. -/
def htypes_full {Δ' Δ_ret lhs : WellFormedRandomContext}
    (htypes : (↑lhs : RandomContext).map Sigma.snd =
        (↑Δ_ret : RandomContext).map Sigma.snd) :
    (↑lhs ++ ↑Δ' : RandomContext).map Sigma.snd =
    (↑Δ_ret ++ ↑Δ' : RandomContext).map Sigma.snd := by
  simp only [List.map_append, htypes]

end Block

namespace Stmt

/-- Sample a single fresh random variable `x : D`: `sample` with `vars = [x : D]`. -/
def sample1 {Δ : WellFormedRandomContext} (x : String) (D : Discrete)
    (hfresh : x ∉ Δ.keys)
    (e : Expr (Θ := ↑Θ) (↑(Δ.cons x D hfresh)) Ty.float) :
    Stmt (Θ := Θ) Δ (Δ.cons x D hfresh) :=
  Stmt.sample ⟨[⟨x, D⟩], List.nodupKeys_cons.mpr ⟨List.not_mem_nil, List.nodupKeys_nil⟩⟩
    (Δ.cons x D hfresh).wf (.wpp e)

/-- Surface `lhs <- if ...` preserves the current random context as carry. -/
def iteCurrent {Δ Δ_then Δ_else : WellFormedRandomContext}
    (lhs : WellFormedRandomContext)
    (thenRet : WellFormedRandomContext)
    (elseRet : WellFormedRandomContext)
    (cond : Expr (Θ := ↑Θ) ↑Δ Ty.bool)
    (thenBody : Stmt (Θ := Θ) Δ Δ_then)
    (elseBody : Stmt (Θ := Θ) Δ Δ_else)
    (hctx_then : (↑Δ_then : RandomContext) = ↑(Block.returnPrefix Δ_then Δ) ++ ↑Δ)
    (hctx_else : (↑Δ_else : RandomContext) = ↑(Block.returnPrefix Δ_else Δ) ++ ↑Δ)
    (htypes_then : (↑lhs : RandomContext).map Sigma.snd =
      (↑thenRet : RandomContext).map Sigma.snd)
    (htypes_else : (↑lhs : RandomContext).map Sigma.snd =
      (↑elseRet : RandomContext).map Sigma.snd)
    (hperm_then : (Block.returnPrefix Δ_then Δ).Perm ↑thenRet)
    (hperm_else : (Block.returnPrefix Δ_else Δ).Perm ↑elseRet)
    (hwf : (↑lhs ++ ↑Δ : RandomContext).NodupKeys) :
    Stmt (Θ := Θ) Δ ⟨↑lhs ++ ↑Δ, hwf⟩ :=
  Stmt.ite lhs cond
    (Block.withCarry thenBody hctx_then hperm_then)
    (Block.withCarry elseBody hctx_else hperm_else)
    htypes_then htypes_else hwf

end Stmt

--  Θ; [] ⊢ s* ⊣ Σ'    Σ ~ Σ'
--  ──────────────────────────────────────  (PROGRAM)
--  program(name, Θ, s*) : Θ → ℙ⟦Σ⟧
inductive Program (Δ : WellFormedRandomContext) : Type where
  | mk
      (name : String)
      (body : Block (Θ := Θ) (∅ : WellFormedRandomContext) (∅ : WellFormedRandomContext) Δ) :
      Program Δ
deriving Repr

end

section
variable {Θ : StaticContext} {Δ : RandomContext}

@[reducible] def Expr.neg {T : Ty} (e : Expr (Θ := Θ) Δ T) : Expr (Θ := Θ) Δ .float :=
  .uop .neg e .neg

@[reducible] def Expr.not {T : Ty} (e : Expr (Θ := Θ) Δ T) : Expr (Θ := Θ) Δ Ty.bool :=
  .uop .not e .not

@[reducible] def Expr.add {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .add a b .add

@[reducible] def Expr.sub {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .sub a b .sub

@[reducible] def Expr.mul {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .mul a b .mul

@[reducible] def Expr.div {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .div a b .div

@[reducible] def Expr.and {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .and a b .and

@[reducible] def Expr.or {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .or a b .or

@[reducible] def Expr.eq {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .eq a b .eq

@[reducible] def Expr.ne {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .ne a b .ne

@[reducible] def Expr.lt {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .lt a b .lt

@[reducible] def Expr.le {T₁ T₂ : Ty} (a : Expr (Θ := Θ) Δ T₁) (b : Expr (Θ := Θ) Δ T₂) :=
  Expr.bop .le a b .le

end

/-!
Declaration well-formedness requirement:

The output context `Δ` (in user-declared returns order) must be a permutation
of the body's internal context `Δ'`. This is enforced by `hperm : Δ.Perm Δ'`.
-/

end Palladium.PAL
