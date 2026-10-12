import Palladium.PAL.Syntax
import Palladium.PAL.Denote
import Util.Macros
import Util.Float
import Std

namespace Palladium.PAL

open Palladium.Util

/-! # Unary operator evaluation -/

/-- Evaluate negation given a validity proof. -/
@[macro_inline]
def UOp.Neg.Valid.eval {T Tₒ : Ty}
    (h : UOp.Neg.Valid T Tₒ) (v : ⟦T⟧) : ⟦Tₒ⟧ :=
  match h with
  | .f => -v
  | .n => -(Float.ofNat v)
  | .d => -(Float.ofNat ↑v)

/-- Evaluate logical not given a validity proof. -/
@[macro_inline]
def UOp.Not.Valid.eval {T Tₒ : Ty}
    (h : UOp.Not.Valid T Tₒ) (v : ⟦T⟧) : ⟦Tₒ⟧ :=
  match h with
  | .f => ↑(!(↑v : Bool))
  | .n => ↑(!Ty.toBool _ v)
  | .d => ↑(!Ty.toBool _ v)

/-- Evaluate a unary operator given a validity proof. -/
@[macro_inline]
def UOp.Valid.eval {op : UOp} {T Tₒ : Ty}
    (h : UOp.Valid op T Tₒ) (v : ⟦T⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ofNeg g => g.eval v
  | .ofNot g => g.eval v

/-! # Binary operator evaluation -/

/-- Evaluate addition given a validity proof. -/
@[macro_inline]
def BOp.Add.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Add.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => va + vb
  | .fn => va + Float.ofNat vb
  | .fd => va + Float.ofNat ↑vb
  | .nf => Float.ofNat va + vb
  | .nn => va + vb
  | .nd => va + vb.val
  | .df => Float.ofNat ↑va + vb
  | .dn => va.val + vb
  | .dd => va.val + vb.val

/-- Evaluate subtraction given a validity proof. -/
@[macro_inline]
def BOp.Sub.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Sub.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => va - vb
  | .fn => va - Float.ofNat vb
  | .fd => va - Float.ofNat ↑vb
  | .nf => Float.ofNat va - vb
  | .nn => va - vb
  | .nd => va - vb.val
  | .df => Float.ofNat ↑va - vb
  | .dn => va.val - vb
  | .dd => ⟨va.val - vb.val, by omega⟩

/-- Evaluate multiplication given a validity proof. -/
@[macro_inline]
def BOp.Mul.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Mul.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => va * vb
  | .fn => va * Float.ofNat vb
  | .fd => va * Float.ofNat ↑vb
  | .nf => Float.ofNat va * vb
  | .nn => va * vb
  | .nd => va * vb.val
  | .df => Float.ofNat ↑va * vb
  | .dn => va.val * vb
  | .dd => ⟨va.val * vb.val, Nat.mul_lt_mul_of_lt_of_lt va.isLt vb.isLt⟩

/-- Evaluate division given a validity proof. -/
@[macro_inline]
def BOp.Div.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Div.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => Float.div0 va vb
  | .fn => Float.div0 va (Float.ofNat vb)
  | .fd => Float.div0 va (Float.ofNat ↑vb)
  | .nf => Float.div0 (Float.ofNat va) vb
  | .nn => Float.div0 (Float.ofNat va) (Float.ofNat vb)
  | .nd => Float.div0 (Float.ofNat va) (Float.ofNat vb.val)
  | .df => Float.div0 (Float.ofNat ↑va) vb
  | .dn => Float.div0 (Float.ofNat ↑va) (Float.ofNat vb)
  | .dd => Float.div0 (Float.ofNat ↑va) (Float.ofNat ↑vb)

/-- Evaluate boolean and given a validity proof. -/
@[macro_inline]
def BOp.And.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.And.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => ↑((↑va : Bool) && (↑vb : Bool))
  | .fn => ↑((↑va : Bool) && (vb != 0))
  | .fd => ↑((↑va : Bool) && Ty.toBool _ vb)
  | .nf => ↑((va != 0) && (↑vb : Bool))
  | .nn => ↑((va != 0) && (vb != 0))
  | .nd => ↑((va != 0) && Ty.toBool _ vb)
  | .df => ↑(Ty.toBool _ va && (↑vb : Bool))
  | .dn => ↑(Ty.toBool _ va && (vb != 0))
  | .dd => ↑(Ty.toBool _ va && Ty.toBool _ vb)

/-- Evaluate boolean or given a validity proof. -/
@[macro_inline]
def BOp.Or.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Or.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => ↑((↑va : Bool) || (↑vb : Bool))
  | .fn => ↑((↑va : Bool) || (vb != 0))
  | .fd => ↑((↑va : Bool) || Ty.toBool _ vb)
  | .nf => ↑((va != 0) || (↑vb : Bool))
  | .nn => ↑((va != 0) || (vb != 0))
  | .nd => ↑((va != 0) || Ty.toBool _ vb)
  | .df => ↑(Ty.toBool _ va || (↑vb : Bool))
  | .dn => ↑(Ty.toBool _ va || (vb != 0))
  | .dd => ↑(Ty.toBool _ va || Ty.toBool _ vb)

/-- Evaluate equality given a validity proof. -/
@[macro_inline]
def BOp.Eq.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Eq.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => ↑(va == vb)
  | .fn => ↑(va == Float.ofNat vb)
  | .fd => ↑(va == Float.ofNat ↑vb)
  | .nf => ↑(Float.ofNat va == vb)
  | .nn => ↑(va == vb)
  | .nd => ↑(va == vb.val)
  | .df => ↑(Float.ofNat ↑va == vb)
  | .dn => ↑(va.val == vb)
  | .dd => ↑(va.val == vb.val)

/-- Evaluate inequality given a validity proof. -/
@[macro_inline]
def BOp.Ne.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Ne.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => ↑(va != vb)
  | .fn => ↑(va != Float.ofNat vb)
  | .fd => ↑(va != Float.ofNat ↑vb)
  | .nf => ↑(Float.ofNat va != vb)
  | .nn => ↑(va != vb)
  | .nd => ↑(va != vb.val)
  | .df => ↑(Float.ofNat ↑va != vb)
  | .dn => ↑(va.val != vb)
  | .dd => ↑(va.val != vb.val)

/-- Evaluate less-than given a validity proof. -/
@[macro_inline]
def BOp.Lt.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Lt.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => ↑(decide (va < vb))
  | .fn => ↑(decide (va < Float.ofNat vb))
  | .fd => ↑(decide (va < Float.ofNat ↑vb))
  | .nf => ↑(decide (Float.ofNat va < vb))
  | .nn => ↑(decide (va < vb))
  | .nd => ↑(decide (va < vb.val))
  | .df => ↑(decide (Float.ofNat ↑va < vb))
  | .dn => ↑(decide (va.val < vb))
  | .dd => ↑(decide (va.val < vb.val))

/-- Evaluate less-or-equal given a validity proof. -/
@[macro_inline]
def BOp.Le.Valid.eval {T₁ T₂ Tₒ : Ty}
    (h : BOp.Le.Valid T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ff => ↑(decide (va ≤ vb))
  | .fn => ↑(decide (va ≤ Float.ofNat vb))
  | .fd => ↑(decide (va ≤ Float.ofNat ↑vb))
  | .nf => ↑(decide (Float.ofNat va ≤ vb))
  | .nn => ↑(decide (va ≤ vb))
  | .nd => ↑(decide (va ≤ vb.val))
  | .df => ↑(decide (Float.ofNat ↑va ≤ vb))
  | .dn => ↑(decide (va.val ≤ vb))
  | .dd => ↑(decide (va.val ≤ vb.val))

/-- Evaluate a binary operator given a validity proof. -/
@[macro_inline]
def BOp.Valid.eval {op : BOp} {T₁ T₂ Tₒ : Ty}
    (h : BOp.Valid op T₁ T₂ Tₒ) (va : ⟦T₁⟧) (vb : ⟦T₂⟧) : ⟦Tₒ⟧ :=
  match h with
  | .ofAdd g => g.eval va vb
  | .ofSub g => g.eval va vb
  | .ofMul g => g.eval va vb
  | .ofDiv g => g.eval va vb
  | .ofAnd g => g.eval va vb
  | .ofOr  g => g.eval va vb
  | .ofEq  g => g.eval va vb
  | .ofNe  g => g.eval va vb
  | .ofLt  g => g.eval va vb
  | .ofLe  g => g.eval va vb

section
variable {Θ : StaticContext} {θ : ⟦Θ⟧}

/-
⟦Θ; Δ ⊢ e : T⟧ : (θ : ⟦Θ⟧) → (δ : ⟦Δ⟧) → ⟦T⟧
-/
/-- Evaluate an expression given random and static environments. -/
def Expr.eval (e : Expr (Θ := Θ) Δ T) (δ : ⟦Δ⟧) :
    ⟦T⟧ :=
  match e with
  | Expr.fin i => i
  | Expr.float r => r
  | Expr.var x _D h => δ⁅x⁆h
  | Expr.param x T h => θ⁅x⁆h
  | Expr.uop _ e hv =>
      δ |> e.eval |> hv.eval
  | Expr.bop _ a b hv =>
      let va := δ |> a.eval
      let vb := δ |> b.eval
      hv.eval va vb
  | Expr.ite (T₁ := T₁) (T₂ := T₂) c t e =>
      let vc := δ |> c.eval
      -- vc : Fin 2 (Ty.bool.denote)
      if Ty.toBool .bool vc then
        let vt := δ |> t.eval
        Ty.cast T₁ (T₁ ⊔ T₂) vt (Ty.Castable.promote_left T₁ T₂)
      else
        let ve := δ |> e.eval
        Ty.cast T₂ (T₁ ⊔ T₂) ve (Ty.Castable.promote_right T₁ T₂)

end

section
variable {Θ : ValidStaticContext} {θ : ⟦Θ⟧}

/-- `Σ_{ρ ∈ ⟦vars⟧} f ρ`: the sum of `f` over all valuations of `vars`. The head variable
    is summed innermost, matching the order in which `REDUCE(e, m, 0)` sums out axis 0
    first. -/
def RandomContext.sumAll : (vars : RandomContext) → (⟦vars⟧ → Float) → Float
  | [], f => f default
  | ⟨x, D⟩ :: rest, f =>
      RandomContext.sumAll rest (fun ρ => (D.support.map (fun v => f (ρ.cons x D v))).sum)

-- ⟦Θ; Δ ⊢ wppc⟧ : (θ : ⟦Θ⟧) → ℙ⟦vars | Δ⟧
/-- Evaluate the weight program to a `ℙ⟦vars | Δ⟧`.
    For each valuation `ρ` of `vars`, the unnormalized weight is
    `w(ρ, δ) := e.eval (ρ ++ δ)`.
    The result is `Pr(x⃗ = ρ | δ) = w(ρ, δ) / Σ_{ρ'} w(ρ', δ)`, computed with
    `Float.div0`, so that `0 / 0 = 0` (normalization is total, as in the paper). -/
def Wppc.eval {Δ vars : WellFormedRandomContext} {hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys} :
    Wppc (Θ := Θ) Δ vars hwf → ℙ⟦↑vars | ↑Δ⟧
  | .wpp e => fun δ =>
    let weight : ⟦(↑vars : RandomContext)⟧ → Float := fun ρ =>
      ρ.prepend (↑vars : RandomContext) δ |> e.eval (θ := θ)
    let total := RandomContext.sumAll ↑vars weight
    fun ρ => Float.div0 (weight ρ) total


-- ⟦Θ; Δ ⊢ x⃗ ← sample wppc⟧ : (θ : ⟦Θ⟧) → ℙ⟦Δ⟧ → ℙ⟦x⃗:D⃗, Δ⟧
/-- Update a joint PMF by sampling: `evalSample(vars, wppc, μ)(ρ ++ δ) =
    μ(δ) * Pr(x⃗ = ρ | δ)`, where `Pr(x⃗ | δ)` is the conditional distribution from
    `wppc.eval`. -/
def Stmt.evalSample {Δ : WellFormedRandomContext} (vars : WellFormedRandomContext)
    {hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys} (wppc : Wppc (Θ := Θ) Δ vars hwf)
    (μ : ℙ⟦Δ⟧) : ℙ⟦(⟨↑vars ++ ↑Δ, hwf⟩ : WellFormedRandomContext)⟧ :=
  fun δ' =>
    let ρ : ⟦(↑vars : RandomContext)⟧ := δ'.takePrefix
    let δ : ⟦Δ⟧ := δ'.dropPrefix
    μ δ * (δ |> wppc.eval (θ := θ)) ρ


-- ⟦Θ; Δ ⊢ marginalize x⟧ : (θ : ⟦Θ⟧) → (δ : ⟦Δ⟧) → ℙ⟦Σ \ x⟧
/-- Marginalize a random variable out of the PMF. -/
def Stmt.evalMarginalize (Δ : WellFormedRandomContext) (x : String)
    (D : Discrete) (h : ⟨x, D⟩ ∈ Δ)
    (μ : ℙ⟦Δ⟧) : ℙ⟦Δ.kerase x⟧ :=
  fun env =>
    D.support
    |> List.map (fun v => env.insert x D v h Δ.wf |> μ)
    |> List.sum

-- ⟦Θ; Δ ⊢ condition c⟧ : (θ : ⟦Θ⟧) → ℙ⟦Δ⟧ → ℙ⟦Δ⟧
/-- Keep the mass `μ δ` where `c` holds and set it to zero elsewhere:
    `⟦condition c⟧ θ μ = λδ. if ⟦c⟧ θ δ then μ δ else 0`. -/
def Stmt.evalCondition {rnd : WellFormedRandomContext} {θ : ⟦Θ⟧}
    (c : Expr (Θ := Θ) ↑rnd Ty.bool) (μ : ℙ⟦rnd⟧) : ℙ⟦rnd⟧ :=
  fun δ => if Ty.toBool .bool (c.eval (θ := θ) δ) then μ δ else 0.0

-- Θ; Σ ⊢ s ⊣ Σ'
-- ⟦ s ⟧ : (param : ⟦ Θ ⟧) → (μ : ℙ⟦ Σ ⟧) → (μ' : ℙ⟦ Σ' ⟧)
mutual
/-- Evaluate a PAL statement (including sequencing and branching). -/
def Stmt.eval {Δ Δ' : WellFormedRandomContext} (s : Stmt (Θ := Θ) Δ Δ')
    (μ : ℙ⟦Δ⟧) : ℙ⟦Δ'⟧ :=
  match s with
  | Stmt.nil _ => μ
  | Stmt.seq first rest =>
      μ |> first.eval |> rest.eval
  | Stmt.sample vars _ wppc =>
      Stmt.evalSample (θ := θ) vars wppc μ
  | Stmt.marginalize x D h =>
      Stmt.evalMarginalize Δ x D h μ
  | Stmt.condition c =>
      Stmt.evalCondition (θ := θ) c μ
  | .ite (Δ' := carry) lhs cond then_block else_block htypes_then htypes_else hwf =>
    let μ_true  : ℙ⟦Δ⟧ := fun δ =>
      if Ty.toBool .bool (cond.eval (θ := θ) δ) then μ δ else 0.0
    let μ_false : ℙ⟦Δ⟧ := fun δ =>
      if Ty.toBool .bool (cond.eval (θ := θ) δ) then 0.0 else μ δ
    fun δ =>
      Block.evalAs then_block htypes_then hwf μ_true δ +
      Block.evalAs else_block htypes_else hwf μ_false δ
  | .repeat (Δ_loop_start := Δ_loop_start) (c := c) (hc := hc)
      lhs hwf n initBlock loopBlock htypes_init htypes_body =>
      let loopCtx : WellFormedRandomContext := ⟨↑lhs ++ ↑Δ_loop_start, hwf⟩
      let f : ℙ⟦loopCtx⟧ → ℙ⟦loopCtx⟧ := fun μ_state =>
        Block.evalAs loopBlock htypes_body hwf μ_state
      let rec loop : Nat → ℙ⟦loopCtx⟧ → ℙ⟦loopCtx⟧
        | 0, acc => acc
        | k + 1, acc => loop k (f acc)
      -- Mixture over the possible iteration counts `k = 0..c-1`: run the loop
      -- `k` times on the slice of the input distribution where `n = k`, then sum.
      fun env =>
        (List.finRange c).foldr
          (fun k acc =>
            let μ_k : ℙ⟦Δ⟧ :=
              Stmt.evalCondition (θ := θ) (Expr.eq n (Expr.fin (hn := hc) k)) μ
            loop k.val (Block.evalAs initBlock htypes_init hwf μ_k) env + acc)
          0.0

/-- Evaluate a block and rename its declared returns to caller-supplied names
    `lhs` of matching types. Used by `Stmt.ite` so that both branches' returns
    align to the user-declared `lhs ++ Δ'` output context, regardless of how
    the per-branch `Δ_ret` were named internally. -/
def Block.evalAs {Δ Δ' Δ_ret lhs : WellFormedRandomContext}
    (blk : Block (Θ := Θ) Δ Δ' Δ_ret)
    (htypes : (↑lhs : RandomContext).map Sigma.snd =
        (↑Δ_ret : RandomContext).map Sigma.snd)
    (hwf : (↑lhs ++ ↑Δ' : RandomContext).NodupKeys)
    (μ : ℙ⟦Δ⟧) :
    ℙ⟦(⟨↑lhs ++ ↑Δ', hwf⟩ : WellFormedRandomContext)⟧ :=
  match blk with
  | Block.mk hperm _ body =>
      fun δ =>
        body.eval μ
          ((δ.renameByTypes (Block.htypes_full htypes)).ofPerm
            (List.Perm.append_right _ hperm.symm))
end

/-- Evaluate a block at its declared return context. -/
def Block.eval {Δ Δ' Δ_ret : WellFormedRandomContext}
    {θ : ⟦Θ⟧}
    (blk : Block (Θ := Θ) Δ Δ' Δ_ret)
    (μ : ℙ⟦Δ⟧) :
    ℙ⟦blk.outputContext⟧ :=
  match blk with
  | Block.mk _ hwf_ext _ =>
      @Block.evalAs Θ θ Δ Δ' Δ_ret Δ_ret blk rfl hwf_ext μ

-- ⟦decl⟧ : (θ : ⟦Θ⟧) → ℙ⟦Δ⟧
/-- Evaluate a PAL declaration starting from the uniform PMF.
    Returns a PMF over the output context `Δ` (in returns order).
    The body is a `Block ∅ ∅ Δ` (empty carry, return = Δ); since `[] ++ ↑Δ`
    reduces definitionally to `↑Δ`, the output context `⟨[] ++ ↑Δ, hwf⟩`
    is the same WFRC as `Δ` (proof irrelevance handles the witness). -/
def Program.eval {Δ : WellFormedRandomContext} (decl : Program (Θ := Θ) Δ) : ℙ⟦Δ⟧ :=
  match decl with
  | .mk _ body =>
      let extCtx : WellFormedRandomContext := body.outputContext
      have htypes : (↑extCtx : RandomContext).map Sigma.snd =
          (↑Δ : RandomContext).map Sigma.snd := by
        cases body
        change (((↑Δ : RandomContext) ++ ([] : RandomContext)).map Sigma.snd =
            (↑Δ : RandomContext).map Sigma.snd)
        simp
      fun δ => (Block.eval (θ := θ) body (fun _ => 1.0))
        (δ.renameByTypes htypes.symm)

end

/-- `⟦e⟧ : (θ : ⟦Θ⟧) → ⟦Δ⟧ → ⟦T⟧`. An expression denotes its semantic function. -/
@[reducible] instance {Θ : StaticContext} : Util.Denote (Expr (Θ := Θ) Δ T) where
  Sem := (θ : ⟦Θ⟧) → ⟦Δ⟧ → ⟦T⟧
  denote e := fun θ => e.eval (θ := θ)

/-- `⟦w⟧ : (θ : ⟦Θ⟧) → ℙ⟦vars | Δ⟧`. A weight program denotes its conditional PMF. -/
@[reducible] instance {Θ : ValidStaticContext} {Δ vars : WellFormedRandomContext}
    {hwf : (↑vars ++ ↑Δ : RandomContext).NodupKeys} :
    Util.Denote (Wppc (Θ := Θ) Δ vars hwf) where
  Sem := (θ : ⟦Θ⟧) → ℙ⟦↑vars | ↑Δ⟧
  denote wppc := fun θ => wppc.eval (θ := θ)

/-- `⟦s⟧ : (θ : ⟦Θ⟧) → ℙ⟦Δ⟧ → ℙ⟦Δ'⟧`. A statement denotes a PMF transformer. -/
@[reducible] instance {Θ : ValidStaticContext} {Δ Δ' : WellFormedRandomContext} :
    Util.Denote (Stmt (Θ := Θ) Δ Δ') where
  Sem := (θ : ⟦Θ⟧) → ℙ⟦Δ⟧ → ℙ⟦Δ'⟧
  denote s := fun θ => s.eval (θ := θ)

/-- `⟦prog⟧ : (θ : ⟦Θ⟧) → ℙ⟦Δ⟧`. A program denotes a PMF over its output context. -/
@[reducible] instance {Θ : ValidStaticContext} {Δ : WellFormedRandomContext} :
    Util.Denote (Program (Θ := Θ) Δ) where
  Sem := (θ : ⟦Θ⟧) → ℙ⟦Δ⟧
  denote prog := fun θ => prog.eval (θ := θ)

end Palladium.PAL
