import Palladium.Compiler.Correctness.Sample
import Palladium.Compiler.Correctness.Marginalize
import Palladium.Compiler.Correctness.Condition
import Palladium.Compiler.Compile.ReturnPerm
import Palladium.Compiler.Compile.Stmt
import Palladium.Compiler.Compile.Block

/-! # Statement & Block-As Correctness -/

namespace Palladium.Compiler

open Palladium.Util
open Palladium.TensorLib
open Palladium.PAL

/-- Pointwise congruence for the zero-seeded right fold used by the `repeat` mixture. -/
theorem foldr_add_congr {c : Nat} (g1 g2 : Fin c → Float) (h : ∀ k, g1 k = g2 k) :
    (List.finRange c).foldr (fun k acc => g1 k + acc) (0.0 : Float)
      = (List.finRange c).foldr (fun k acc => g2 k + acc) (0.0 : Float) := by
  induction (List.finRange c) with
  | nil => rfl
  | cons k l ih => simp only [List.foldr_cons, h k, ih]

section

variable {L : Tensor.Layout} [Tensor.Operations L]

/-- The denotation of a single same-shape `bop .add` (the cell used in `mixtureAdd` and in
    the `ite` case): `⟦a + acc⟧ γ [i] = ⟦a⟧ γ [i] + ⟦acc⟧ γ [i]`. -/
theorem bop_add_denote {Γ : JAX.Context} {σ : Shape}
    (a accE : JAX.Expr (L := L) Γ (JAX.Ty.tensor σ .float))
    (γ : JAX.Valuation Γ) (idx : Indices σ)
    (hbc : Shape.BroadcastCompatible σ σ) (hshape : Shape.broadcast2 σ σ hbc = σ) :
    (hshape ▸ JAX.Expr.bop JAX.BOp.add a accE JAX.BOp.Valid.add_ff hbc).denote γ idx
      = a.denote γ idx + accE.denote γ idx := by
  rw [denote_shape_cast]
  simp only [JAX.Expr.denote]
  rw [Indices.proj_of_eq _ hshape.symm]
  rfl

/-- The denotation of the stochastic-`repeat` mixture: the right fold of the per-term
    denotations, with `0.0` seed. -/
theorem mixtureAdd_denote [Tensor.LawfulOperations L] [Tensor.OfTabulated L]
    {Γ : JAX.Context} (σ : Shape) (c : Nat)
    (term : Fin c → JAX.Expr (L := L) Γ (JAX.Ty.tensor σ .float))
    (γ : JAX.Valuation Γ) (idx : Indices σ) :
    (PAL.Stmt.mixtureAdd σ c term).denote γ idx
      = (List.finRange c).foldr (fun k acc => (term k).denote γ idx + acc) 0.0 := by
  unfold PAL.Stmt.mixtureAdd
  induction (List.finRange c) with
  | nil =>
    simp only [List.foldr_nil, JAX.Expr.denote]
    exact Tensor.index_fill (0.0 : Float) σ idx
  | cons k l ih =>
    rw [List.foldr_cons, List.foldr_cons, bop_add_denote, ih]

mutual

/-- Dispatches to sample/marginalize/condition correctness, and handles nil/seq.
    The `ite` and `repeat` cases call `Block.correctnessAs` on the blocks (mutual). -/
theorem Stmt.correctness
    [Tensor.LawfulOperations L] [Tensor.OfTabulated L] [Tensor.LawfulOfTabulated L]
    {Θ : ValidStaticContext} {Δ Δ' : WellFormedRandomContext}
    {Γ : JAX.Context}
    (s : PAL.Stmt (Θ := Θ) Δ Δ')
    {θ : ⟦Θ⟧}
    (μ : ℙ⟦↑Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float))
    (hpt : probabilityTensor ∉ (Γ ++ 𝒞⟦Θ⟧).keys) :
    Stmt.Correct (θ := θ) s μ γ ptExpr := by
  unfold Stmt.Correct
  simp only [Util.Compile.compile, Util.Denote.denote]
  intro hpc hμ
  match s with
  | .nil _ => exact hμ
  | .seq first rest =>
    let firstExpr := first.compile (Γ := Γ) ptExpr
    have h_first := Stmt.correctness first (θ := θ) μ γ ptExpr hpt
    unfold Stmt.Correct at h_first
    simp only [Util.Compile.compile, Util.Denote.denote] at h_first
    have h_mid_μ := h_first hpc hμ
    exact (Stmt.correctness rest (μ |> ⟦first⟧ (θ := θ)) γ firstExpr hpt) hpc h_mid_μ
  | .sample vars hwf wppc =>
    exact Sample.correctness vars hwf wppc μ γ ptExpr hμ hpc
  | .marginalize x D h =>
    exact Marginalize.correctness x D h μ γ ptExpr hμ
  | .condition c =>
    exact Condition.correctness c μ γ ptExpr hμ hpc hpt
  | .ite (Δ' := carry) lhs cond
      (PAL.Block.mk (Δ_ret_var := Δ_ret_var_t) hperm_t hwf_then body_t)
      (PAL.Block.mk (Δ_ret_var := Δ_ret_var_e) hperm_e hwf_else body_e)
      htypes_then htypes_else hwf =>
    let outCtx : WellFormedRandomContext := ⟨↑lhs ++ ↑carry, hwf⟩
    let then_block : PAL.Block (Θ := Θ) Δ carry _ :=
      .mk hperm_t hwf_then body_t
    let else_block : PAL.Block (Θ := Θ) Δ carry _ :=
      .mk hperm_e hwf_else body_e
    -- Step 1: Condition correctness for both branches.
    have h_cond_true := Condition.correctness cond μ γ ptExpr hμ hpc hpt
    have h_cond_false :=
      Condition.correctness (PAL.Expr.not cond) μ γ ptExpr hμ hpc hpt
    let maskedTrue := PAL.Stmt.compileCondition (L := L) cond ptExpr
    let maskedFalse := PAL.Stmt.compileCondition (L := L) (PAL.Expr.not cond) ptExpr
    -- Step 2: Per-branch correctnessAs (encapsulated transpose + shape cast).
    have h_then_blk : (PAL.Block.compileAs then_block htypes_then hwf maskedTrue).denote γ ~
        PAL.Block.evalAs (θ := θ) then_block htypes_then hwf
          (PAL.Stmt.evalCondition (θ := θ) cond μ) :=
      Block.correctnessAs then_block htypes_then hwf
        (PAL.Stmt.evalCondition (θ := θ) cond μ) γ maskedTrue hpt hpc h_cond_true
    have h_else_blk : (PAL.Block.compileAs else_block htypes_else hwf maskedFalse).denote γ ~
        PAL.Block.evalAs (θ := θ) else_block htypes_else hwf
          (PAL.Stmt.evalCondition (θ := θ) (PAL.Expr.not cond) μ) :=
      Block.correctnessAs else_block htypes_else hwf
        (PAL.Stmt.evalCondition (θ := θ) (PAL.Expr.not cond) μ) γ maskedFalse hpt hpc
        h_cond_false
    -- Step 3: Reduce evalCondition to if-form (so the PMFs match Stmt.eval ite).
    have h_eq_true : PAL.Stmt.evalCondition (θ := θ) cond μ =
        (fun δ' => if Ty.toBool .bool (cond.eval (θ := θ) δ') then μ δ' else 0.0) :=
      funext fun δ' => by
        simp only [PAL.Stmt.evalCondition]
    have h_eq_false : PAL.Stmt.evalCondition (θ := θ) (PAL.Expr.not cond) μ =
        (fun δ' => if Ty.toBool .bool (cond.eval (θ := θ) δ') then 0.0 else μ δ') :=
      funext fun δ' => by
        simp only [PAL.Stmt.evalCondition, PAL.Expr.eval, PAL.UOp.Valid.eval,
          PAL.UOp.Not.Valid.eval, PAL.Ty.toBool_not_eq]
        cases Ty.toBool Ty.bool (cond.eval (θ := θ) δ') <;> rfl
    rw [h_eq_true] at h_then_blk
    rw [h_eq_false] at h_else_blk
    -- Step 4: Add the two branches.
    intro δ
    have hbc := (Shape.Broadcastable.refl 𝒞⟦outCtx⟧).toBroadcastCompatible
    have hshape : Shape.broadcast2 𝒞⟦outCtx⟧ 𝒞⟦outCtx⟧ hbc = 𝒞⟦outCtx⟧ :=
      (Shape.Broadcastable.refl 𝒞⟦outCtx⟧).broadcast2_eq_right hbc
    calc
      _ = (PAL.Block.compileAs then_block htypes_then hwf maskedTrue).denote γ
            δ.toIndices +
          (PAL.Block.compileAs else_block htypes_else hwf maskedFalse).denote γ
            δ.toIndices := by
        simp only [PAL.Stmt.compile]
        exact bop_add_denote _ _ γ δ.toIndices hbc hshape
      _ = (PAL.Block.evalAs (θ := θ) then_block htypes_then hwf
              (fun δ' => if Ty.toBool .bool (cond.eval (θ := θ) δ') then μ δ' else 0.0)) δ +
          (PAL.Block.evalAs (θ := θ) else_block htypes_else hwf
              (fun δ' => if Ty.toBool .bool (cond.eval (θ := θ) δ') then 0.0 else μ δ')) δ := by
        rw [h_then_blk δ, h_else_blk δ]
      _ = _ := by
        -- Stmt.eval ite is defined to call Block.evalAs on each branch.
        simp only [PAL.Stmt.eval, then_block, else_block]
  | .repeat (Δ_loop_start := Δ_loop_start) (c := c) (hc := hc)
      lhs hwf n initBlock loopBlock htypes_init htypes_body =>
    let loopCtx : WellFormedRandomContext := ⟨↑lhs ++ ↑Δ_loop_start, hwf⟩
    let zName := probabilityTensor ++ "_scan"
    let zType := JAX.Ty.tensor 𝒞⟦loopCtx⟧ .float
    let zBind : Σ _ : String, JAX.Ty := ⟨zName, zType⟩
    let zVar : JAX.Expr (L := L) ((zBind :: Γ) ++ 𝒞⟦Θ⟧) zType :=
      JAX.Expr.var zName zType (.head _)
    let bodyExpr : JAX.Expr (L := L) ((zBind :: Γ) ++ 𝒞⟦Θ⟧)
        (JAX.Ty.tensor 𝒞⟦loopCtx⟧ .float) :=
      PAL.Block.compileAs loopBlock htypes_body hwf zVar
    let f_t : zType.Sem → zType.Sem :=
      fun t => bodyExpr.denote (γ.cons zName zType t)
    let f_μ : ℙ⟦loopCtx⟧ → ℙ⟦loopCtx⟧ :=
      fun μ_state => PAL.Block.evalAs (θ := θ) loopBlock htypes_body hwf μ_state
    have hReserved : isReserved (probabilityTensor ++ "_scan") = true := by decide
    have hpt_ne_scan : probabilityTensor ≠ probabilityTensor ++ "_scan" := by decide
    have hpt_ext : probabilityTensor ∉ ((zBind :: Γ) ++ 𝒞⟦Θ⟧).keys := by
      intro hmem
      exact (List.mem_cons.mp hmem).elim (fun h => absurd h hpt_ne_scan) (fun h => hpt h)
    have h_body_step : ∀ (t_k : zType.Sem) (μ_k : ℙ⟦loopCtx⟧),
        t_k ~ μ_k → f_t t_k ~ f_μ μ_k := by
      intro t_k μ_k h_tk
      have hpc' := hpc.cons_reserved (probabilityTensor ++ "_scan")
        zType t_k hReserved
      have h_input : zVar.denote (γ.cons zName zType t_k) = t_k :=
        Environment.get_cons_eq γ zName zType t_k
      have h_tk' : zVar.denote (γ.cons zName zType t_k) ~ μ_k := by
        rw [h_input]; exact h_tk
      exact Block.correctnessAs loopBlock htypes_body hwf μ_k
        (γ.cons zName zType t_k) zVar hpt_ext hpc' h_tk'
    have h_iter : ∀ (k : Nat) (t₀ : zType.Sem) (μ₀ : ℙ⟦loopCtx⟧),
        t₀ ~ μ₀ →
        Nat.iterate f_t k t₀ ~ PAL.Stmt.eval.loop loopCtx f_μ k μ₀ := by
      intro k
      induction k with
      | zero =>
        intro t₀ μ₀ h
        simpa only [Function.iterate_zero, id, PAL.Stmt.eval.loop.eq_1] using h
      | succ k ih =>
        intro t₀ μ₀ h_tm
        simp only [Function.iterate_succ, Function.comp, PAL.Stmt.eval.loop.eq_2]
        exact ih _ _ (h_body_step t₀ μ₀ h_tm)
    -- Per-slice correctness: the `k`-th masked scan term represents the loop run
    -- `k` times on the `n = k` slice of the input distribution.
    have h_perk : ∀ (k : Fin c),
        (JAX.Expr.scan zName
            (PAL.Block.compileAs initBlock htypes_init hwf
              (PAL.Stmt.compileCondition (L := L)
                (PAL.Expr.eq n (PAL.Expr.fin (hn := hc) k)) ptExpr))
            k.val bodyExpr).denote γ ~
          PAL.Stmt.eval.loop loopCtx f_μ k.val
            (PAL.Block.evalAs (θ := θ) initBlock htypes_init hwf
              (PAL.Stmt.evalCondition (θ := θ)
                (PAL.Expr.eq n (PAL.Expr.fin (hn := hc) k)) μ)) := by
      intro k
      have h_cond_k :=
        Condition.correctness (PAL.Expr.eq n (PAL.Expr.fin (hn := hc) k))
          μ γ ptExpr hμ hpc hpt
      have h_init_k :=
        Block.correctnessAs initBlock htypes_init hwf
          (PAL.Stmt.evalCondition (θ := θ)
            (PAL.Expr.eq n (PAL.Expr.fin (hn := hc) k)) μ)
          γ (PAL.Stmt.compileCondition (L := L)
            (PAL.Expr.eq n (PAL.Expr.fin (hn := hc) k)) ptExpr) hpt hpc h_cond_k
      have hstep := h_iter k.val _ _ h_init_k
      simpa only [JAX.Expr.denote] using hstep
    -- Assemble the mixture termwise.
    intro env
    simp only [PAL.Stmt.compile]
    rw [mixtureAdd_denote]
    simp only [PAL.Stmt.eval]
    apply foldr_add_congr
    intro k
    exact h_perk k env

/-- Per-branch correctness: `Block.compileAs` represents `Block.evalAs`. -/
theorem Block.correctnessAs
    [Tensor.LawfulOperations L] [Tensor.OfTabulated L] [Tensor.LawfulOfTabulated L]
    {Θ : ValidStaticContext} {Δ Δ' Δ_ret lhs : WellFormedRandomContext}
    {Γ : JAX.Context} (blk : PAL.Block (Θ := Θ) Δ Δ' Δ_ret)
    (htypes : (↑lhs : RandomContext).map Sigma.snd =
        (↑Δ_ret : RandomContext).map Sigma.snd)
    (hwf : (↑lhs ++ ↑Δ' : RandomContext).NodupKeys) {θ : ⟦Θ⟧} (μ : ℙ⟦Δ⟧)
    (γ : JAX.Valuation (Γ ++ 𝒞⟦Θ⟧))
    (ptExpr : JAX.Expr (L := L) (Γ ++ 𝒞⟦Θ⟧) (JAX.Ty.tensor 𝒞⟦Δ⟧ .float))
    (hpt : probabilityTensor ∉ (Γ ++ 𝒞⟦Θ⟧).keys)
    (hpc : JAX.Valuation.RepresentsParams γ θ) (hμ : ptExpr.denote γ ~ μ) :
    (PAL.Block.compileAs blk htypes hwf ptExpr).denote γ ~
      PAL.Block.evalAs (θ := θ) blk htypes hwf μ := by
  match blk with
  | PAL.Block.mk (Δ_ret_var := Δ_ret_var) hperm hwf_ext body =>
  let bodyCtx : WellFormedRandomContext :=
    ⟨↑Δ_ret_var ++ ↑Δ', PAL.blockInternalNodup hperm hwf_ext⟩
  let extCtx : WellFormedRandomContext := ⟨↑Δ_ret ++ ↑Δ', hwf_ext⟩
  have hperm_full : (↑extCtx : RandomContext).Perm ↑bodyCtx :=
    List.Perm.append_right _ hperm.symm
  let returnPerm := PAL.Block.computeReturnPerm ↑bodyCtx extCtx.keys
  have hPermValid : Permutation.Valid returnPerm 𝒞⟦bodyCtx⟧.rank := by
    rw [bodyCtx.toShape_length]
    exact PAL.Block.computeReturnPerm_valid ↑bodyCtx extCtx.keys bodyCtx.wf
      (hperm_full.map Sigma.fst)
  have hshape_perm : 𝒞⟦bodyCtx⟧.permute returnPerm hPermValid = 𝒞⟦extCtx⟧ :=
    PAL.Block.toShape_permute_computeReturnPerm ↑bodyCtx (↑extCtx) hperm_full
      extCtx.keys_nodup hPermValid
  let extCtx_lhs : WellFormedRandomContext := ⟨↑lhs ++ ↑Δ', hwf⟩
  have hshape_lhs : (𝒞⟦extCtx⟧ : Shape) = 𝒞⟦extCtx_lhs⟧ :=
    RandomContext.toShape_eq_of_map_snd_eq (PAL.Block.htypes_full htypes).symm
  have h_body := (Stmt.correctness (Γ := Γ) body μ γ ptExpr hpt) hpc hμ
  by_cases hctx : (↑Δ_ret_var : RandomContext) = (↑Δ_ret : RandomContext)
  · have hctx_wfc : Δ_ret_var = Δ_ret := by
      cases Δ_ret_var
      cases Δ_ret
      cases hctx
      rfl
    subst hctx_wfc
    have hshape_body_ext : 𝒞⟦bodyCtx⟧ = 𝒞⟦extCtx⟧ := by
      change RandomContext.toShape (↑bodyCtx : RandomContext) =
        RandomContext.toShape (↑extCtx : RandomContext)
      rfl
    have htypes_direct : (↑bodyCtx : RandomContext).map Sigma.snd =
        (↑extCtx_lhs : RandomContext).map Sigma.snd := by
      exact (PAL.Block.htypes_full htypes).symm
    have h_direct := h_body.of_shape_cast htypes_direct bodyCtx.wf
      (hshape := hshape_body_ext.trans hshape_lhs)
    have h_env : ∀ σ : ⟦(↑extCtx_lhs : RandomContext)⟧,
        σ.renameByTypes htypes_direct.symm =
          (σ.renameByTypes (PAL.Block.htypes_full htypes)).ofPerm hperm_full := by
      intro σ
      funext x D hx
      simp [Environment.ofPerm]
    intro σ
    have h := h_direct σ
    change _ = PAL.Stmt.eval (θ := θ) body μ
      (σ.renameByTypes htypes_direct.symm) at h
    rw [h_env σ] at h
    simp only [PAL.Block.compileAs, PAL.Block.evalAs, Util.Compile.compile]
    change (((hshape_body_ext.trans hshape_lhs ▸ PAL.Stmt.compile body ptExpr).denote γ
        σ.toIndices) =
      PAL.Stmt.eval (θ := θ) body μ
        ((σ.renameByTypes (PAL.Block.htypes_full htypes)).ofPerm hperm_full))
    rw [denote_shape_cast]
    exact h
  · have h_perm :=
      h_body.transpose
        (hnd_keys := bodyCtx.wf) bodyCtx.wf hperm_full returnPerm rfl hPermValid
        extCtx.keys_nodup hshape_perm
    have h_cast := h_perm.of_shape_cast (PAL.Block.htypes_full htypes).symm extCtx.wf
    intro σ
    have h := h_cast σ
    simp only [PAL.Block.compileAs, PAL.Block.evalAs, hctx, Util.Compile.compile, dite_false]
    rw [denote_shape_cast]
    exact h

end

end

end Palladium.Compiler
