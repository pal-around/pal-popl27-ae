import TensorLib.Operations.Instances
import TensorLib.Operations.Lemmas
import TensorLib.Operations.TabulatedInterop
import Std

namespace Palladium.TensorLib

namespace Instances

/-! ### Tabulated helper lemmas for LawfulOperations -/

/-- `Vector.mapM id` in the `Id` monad is the identity. -/
private theorem Vector_mapM_id_eq {α : Type} {n : Nat} (v : Vector α n) :
    Vector.mapM (m := Id) id v = v := by
  show Vector.mapM (m := Id) (fun x => @pure Id _ _ (id x)) v = v
  rw [Vector.mapM_pure (m := Id), Vector.map_id]
  rfl

/-! ### broadcast_map helpers for Tabulated -/

/-- `wrapSingletons` commutes with `map`. -/
private theorem wrapSingletons_map
    {α β : Type} {σ : Shape} (data : Tabulated σ α) (f : α → β) (n : Nat) :
    Tabulated.wrapSingletons σ (Tabulated.map data f) n =
    Tabulated.map (Tabulated.wrapSingletons σ data n) f := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show Vector.singleton (Tabulated.wrapSingletons σ (Tabulated.map data f) n) =
      Vector.map (fun x => Tabulated.map x f)
        (Vector.singleton (Tabulated.wrapSingletons σ data n))
    rw [ih]
    ext i hi
    simp only [Vector.getElem_map]
    have hi0 : i = 0 := by omega
    subst hi0; simp

/-- `broadcastAligned` commutes with `map`. -/
private theorem broadcastAligned_map
    {α β : Type} (σₛ σₜ : Shape)
    (data : Tabulated σₛ α) (f : α → β)
    (h : Shape.BroadcastableAligned σₛ σₜ) :
    Tabulated.broadcastAligned σₛ σₜ (Tabulated.map data f) h =
    Tabulated.map (Tabulated.broadcastAligned σₛ σₜ data h) f := by
  match σₛ, σₜ, data, h with
  | [], [], _, _ => rfl
  | [], _ :: _, _, h => exact nomatch h
  | _ :: _, [], _, h => exact nomatch h
  | df :: dfs, dt :: dts, xs, h =>
    simp only [Tabulated.broadcastAligned]
    by_cases heq : df = dt
    · subst heq; simp only [↓reduceDIte, Tabulated.map]
      ext i hi; simp only [Vector.getElem_map]
      exact broadcastAligned_map dfs dts xs[i] f h.tail
    · simp only [heq, ↓reduceDIte, Tabulated.map]
      have h0 : 0 < df := by have := h.one_of_ne heq; omega
      simp only [Vector.getElem_map]
      rw [broadcastAligned_map dfs dts (xs[0]'h0) f h.tail]
      ext i hi; simp [Vector.getElem_replicate]

/-! ### broadcast_index helpers for Tabulated -/

/-- In the equal-dimensions case, `get` distributes over `broadcastAligned`. -/
private theorem broadcastAligned_get_eq
    {α : Type} {dfs dts : Shape} {df : Nat}
    (xs : Tabulated (df :: dfs) α)
    (h : Shape.BroadcastableAligned (df :: dfs) (df :: dts))
    (i : Nat) (hi : i < df) :
    Tensor.Operations.get
      (Tabulated.broadcastAligned (df :: dfs) (df :: dts) xs h) ⟨i, hi⟩ =
    Tabulated.broadcastAligned dfs dts (xs[i]'hi) h.tail := by
  simp only [Tensor.Operations.get, Tabulated.get,
             Tabulated.broadcastAligned, ↓reduceDIte,
             Fin.getElem_fin, Vector.getElem_map]

/-- In the one-dimension case, `get` distributes over `broadcastAligned`. -/
private theorem broadcastAligned_get_one
    {α : Type} {df : Nat} {dfs dts : Shape} {dt : Nat}
    (xs : Tabulated (df :: dfs) α)
    (h : Shape.BroadcastableAligned (df :: dfs) (dt :: dts))
    (heq : ¬(df = dt))
    (i : Nat) (hi : i < dt) :
    Tensor.Operations.get
      (Tabulated.broadcastAligned (df :: dfs) (dt :: dts) xs h) ⟨i, hi⟩ =
    Tabulated.broadcastAligned dfs dts
      (xs[0]'(by have := h.one_of_ne heq; omega)) h.tail := by
  simp only [Tensor.Operations.get, Tabulated.get,
             Tabulated.broadcastAligned, heq, ↓reduceDIte,
             Fin.getElem_fin, Vector.getElem_replicate]

/-- `broadcastAligned` preserves `index` correspondence: indexing the result
    at any valid target index yields a value equal to indexing the source
    at some projected index. -/
private theorem broadcastAligned_index
    {α : Type} {σₛ σₜ : Shape} (data : Tabulated σₛ α)
    (h : Shape.BroadcastableAligned σₛ σₜ) (indices : Indices σₜ) :
    ∃ indices' : Indices σₛ,
      Tensor.Operations.index σₜ
        (Tabulated.broadcastAligned σₛ σₜ data h) indices =
      Tensor.Operations.index σₛ data indices' := by
  induction h with
  | nil => exact ⟨indices, rfl⟩
  | @eq σₛ' σₜ' d h_tail ih =>
    obtain ⟨indices'_tail, h_eq⟩ := ih data[indices.head] indices.tail
    refine ⟨Indices.cons indices.head.val indices.head.isLt indices'_tail, ?_⟩
    unfold Tensor.Operations.index
    simp only [Indices.cons_head, Indices.cons_tail]
    have h_get : Tensor.Operations.get
        (Tabulated.broadcastAligned (d :: σₛ') (d :: σₜ') data (.eq h_tail)) indices.head =
      Tabulated.broadcastAligned σₛ' σₜ' data[indices.head] h_tail := by
      show Tabulated.get _ indices.head = _
      simp only [Tabulated.broadcastAligned, ↓reduceDIte, Tabulated.get,
                 Vector.getElem_map, Fin.getElem_fin]
    rw [h_get]; exact h_eq
  | @one σₛ' σₜ' d h_tail ih =>
    have h0 : (0 : Nat) < 1 := by omega
    obtain ⟨indices'_tail, h_eq⟩ := ih (data[0]'h0) indices.tail
    refine ⟨Indices.cons 0 h0 indices'_tail, ?_⟩
    unfold Tensor.Operations.index
    simp only [Indices.cons_head, Indices.cons_tail]
    have h_get : Tensor.Operations.get
        (Tabulated.broadcastAligned (1 :: σₛ') (d :: σₜ') data (.one h_tail)) indices.head =
      Tabulated.broadcastAligned σₛ' σₜ' (data[0]'h0) h_tail := by
      show Tabulated.get _ indices.head = _
      simp only [Tabulated.broadcastAligned]
      by_cases hd : (1 : Nat) = d
      · subst hd; simp only [↓reduceDIte, Tabulated.get, Vector.getElem_map, Fin.getElem_fin]
        have hi0 : indices.head.val = 0 := by have := indices.head.isLt; omega
        simp only [hi0]
      · simp only [hd, ↓reduceDIte, Tabulated.get, Vector.getElem_replicate, Fin.getElem_fin]
    rw [h_get]; exact h_eq

/-- `wrapSingletons` preserves `index` correspondence: indexing into
    singleton-padded data yields the same value as indexing the original. -/
private theorem wrapSingletons_index
    {α : Type} (σ : Shape) (data : Tabulated σ α)
    (n : Nat) (indices : Indices (List.replicate n 1 ++ σ)) :
    ∃ indices' : Indices σ,
      Tensor.Operations.index (List.replicate n 1 ++ σ)
        (Tabulated.wrapSingletons σ data n) indices =
      Tensor.Operations.index σ data indices' := by
  induction n with
  | zero => exact ⟨indices, rfl⟩
  | succ n ih =>
    -- index (replicate (n+1) 1 ++ σ) (wrapSingletons data (n+1)) indices
    -- reduces by def to index (replicate n 1 ++ σ) (get (...) indices.head) indices.tail
    have h_reduce : Tensor.Operations.index (List.replicate (n + 1) 1 ++ σ)
        (Tabulated.wrapSingletons σ data (n + 1)) indices =
      Tensor.Operations.index (List.replicate n 1 ++ σ)
        (Tensor.Operations.get (Tabulated.wrapSingletons σ data (n + 1)) indices.head)
        indices.tail := rfl
    have h_get : Tensor.Operations.get
        (Tabulated.wrapSingletons σ data (n + 1)) indices.head =
        Tabulated.wrapSingletons σ data n := by
      show Tabulated.get (Tabulated.wrapSingletons σ data (n + 1)) indices.head = _
      simp only [Tabulated.wrapSingletons, Tabulated.get]
      have hi0 : indices.head.val = 0 := by omega
      simp
    rw [h_reduce, h_get]
    exact ih indices.tail

/-! ### Broadcast law helpers -/

-- These lemmas capture the properties of `Tabulated.broadcast` used to prove the
-- broadcast laws of `Tensor.LawfulOperations`.

/- Broadcasting a tensor to its own shape is the identity. -/
private theorem broadcastAligned_refl {α : Type} {σ : Shape}
    (t : Tabulated σ α) (h : Shape.BroadcastableAligned σ σ) :
    Tabulated.broadcastAligned σ σ t h = t := by
  match σ, t with
  | [], _ => rfl
  | d :: ds, xs =>
    ext i hi
    show Tensor.Operations.get
      (Tabulated.broadcastAligned (d :: ds) (d :: ds) xs h) ⟨i, hi⟩ = xs[i]
    rw [broadcastAligned_get_eq xs h i hi]
    exact broadcastAligned_refl xs[i] h.tail

private theorem broadcastAligned_trans {α : Type} {σ₁ σ₂ σ₃ : Shape}
    (t : Tabulated σ₁ α)
    (h12 : Shape.BroadcastableAligned σ₁ σ₂)
    (h23 : Shape.BroadcastableAligned σ₂ σ₃)
    (h13 : Shape.BroadcastableAligned σ₁ σ₃) :
    Tabulated.broadcastAligned σ₂ σ₃
      (Tabulated.broadcastAligned σ₁ σ₂ t h12) h23 =
    Tabulated.broadcastAligned σ₁ σ₃ t h13 := by
  match σ₁, σ₂, σ₃, t, h12, h23, h13 with
  | [], [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, h, _, _ => exact nomatch h
  | _ :: _, [], _, _, h, _, _ => exact nomatch h
  | d1 :: ds1, d2 :: ds2, d3 :: ds3, xs, h12, h23, h13 =>
    by_cases h12_eq : d1 = d2 <;> by_cases h23_eq : d2 = d3
    · -- d1 = d2, d2 = d3, so d1 = d3
      subst h12_eq; subst h23_eq
      simp only [Tabulated.broadcastAligned, dif_pos trivial]
      refine Vector.ext fun i hi => ?_
      simp only [Vector.getElem_map]
      exact broadcastAligned_trans xs[i] h12.tail h23.tail h13.tail
    · -- d1 = d2, d2 ≠ d3
      subst h12_eq
      have h1_one := h23.one_of_ne h23_eq
      simp only [Tabulated.broadcastAligned, dif_pos trivial, dif_neg h23_eq]
      have h0 : 0 < d1 := by omega
      simp only [Vector.getElem_map]
      congr 1
      exact broadcastAligned_trans (xs[0]'h0) h12.tail h23.tail h13.tail
    · -- d1 ≠ d2, d2 = d3
      subst h23_eq
      have h1_one := h12.one_of_ne h12_eq
      simp only [Tabulated.broadcastAligned, dif_neg h12_eq, dif_pos trivial]
      have h0 : 0 < d1 := by omega
      refine Vector.ext fun i hi => ?_
      simp only [Vector.getElem_map, Vector.getElem_replicate]
      exact broadcastAligned_trans (xs[0]'h0) h12.tail h23.tail h13.tail
    · -- d1 ≠ d2, d2 ≠ d3 → d1 = 1, d2 = 1 → d1 ≠ d3
      have h1_one := h12.one_of_ne h12_eq
      have h2_one := h23.one_of_ne h23_eq
      have h13_ne : ¬(d1 = d3) := fun h => h23_eq (by omega)
      simp only [Tabulated.broadcastAligned, dif_neg h12_eq, dif_neg h23_eq, dif_neg h13_ne]
      have h0_d1 : 0 < d1 := by omega
      congr 1
      simp only [Vector.getElem_replicate]
      exact broadcastAligned_trans (xs[0]'h0_d1) h12.tail h23.tail h13.tail

/- Vector.singleton preserves HEq. -/
private theorem Vector.singleton_heq' {α β : Type} {a : α} {b : β}
    (h : HEq a b) : HEq (Vector.singleton a) (Vector.singleton b) := by
  cases h; rfl

/- broadcastAligned is compatible with shape equality + data HEq. -/
private theorem broadcastAligned_shape_heq {α : Type} {S1 S2 σₜ : Shape}
    (hS : S1 = S2) {D1 : Tabulated S1 α} {D2 : Tabulated S2 α}
    (hD : HEq D1 D2) (h1 : Shape.BroadcastableAligned S1 σₜ)
    (h2 : Shape.BroadcastableAligned S2 σₜ) :
    Tabulated.broadcastAligned S1 σₜ D1 h1 =
    Tabulated.broadcastAligned S2 σₜ D2 h2 := by
  subst hS; cases hD; rfl

/- Nested wrapSingletons collapses: wrapping b then a is the same as wrapping (b+a).
   Uses HEq because the types differ propositionally.
   Uses b+a (not a+b) because Nat.add matches on the second argument,
   so b+0=b and b+(a+1)=(b+a)+1 are definitional. -/
private theorem wrap_wrap_collapse_heq {α : Type} {σ : Shape}
    (data : Tabulated σ α) (a b : Nat) :
    HEq (Tabulated.wrapSingletons (List.replicate b 1 ++ σ)
           (Tabulated.wrapSingletons σ data b) a)
        (Tabulated.wrapSingletons σ data (b + a)) := by
  induction a with
  | zero => exact HEq.rfl
  | succ a ih =>
    simp only [Tabulated.wrapSingletons, List.replicate_succ, List.cons_append] at ih ⊢
    exact Vector.singleton_heq' ih

/- wrapSingletons with propositionally equal nat arguments are HEq. -/
private theorem wrapSingletons_nat_heq {α : Type} {σ : Shape} (data : Tabulated σ α)
    {a b : Nat} (h : a = b) :
    HEq (Tabulated.wrapSingletons σ data a) (Tabulated.wrapSingletons σ data b) := by
  subst h; rfl

/-- `wrapSingletons` with `n = 0` is the identity on data. -/
private theorem wrapSingletons_zero {α : Type} {σ : Shape} (data : Tabulated σ α) :
    Tabulated.wrapSingletons σ data 0 = data := by
  cases σ with
  | nil => rfl
  | cons d ds =>
    rfl

/- Wrapping singletons around a broadcastAligned result, then broadcasting again,
   is the same as wrapping the original data and broadcasting directly.
   Combines wrapSingletons_broadcastAligned + broadcastAligned_trans + wrap_wrap_collapse. -/
private theorem wrapBroadcast_trans {α : Type} {σ₁ σ₂ : Shape}
    (data : Tabulated σ₁ α) (h12 : Shape.BroadcastableAligned σ₁ σ₂)
    (n : Nat) {σ₃ : Shape}
    (h23 : Shape.BroadcastableAligned (List.replicate n 1 ++ σ₂) σ₃)
    (h13 : Shape.BroadcastableAligned (List.replicate n 1 ++ σ₁) σ₃) :
    Tabulated.broadcastAligned (List.replicate n 1 ++ σ₂) σ₃
      (Tabulated.wrapSingletons σ₂ (Tabulated.broadcastAligned σ₁ σ₂ data h12) n) h23 =
    Tabulated.broadcastAligned (List.replicate n 1 ++ σ₁) σ₃
      (Tabulated.wrapSingletons σ₁ data n) h13 := by
  induction n generalizing σ₃ with
  | zero => exact broadcastAligned_trans data h12 h23 h13
  | succ n ih =>
    match σ₃, h23, h13 with
    | [], h23, _ => exact nomatch h23
    | d :: σ₃', h23, h13 =>
      -- Step 1: Normalize shapes to cons form
      simp only [List.replicate_succ, List.cons_append] at h23 h13 ⊢
      -- Step 2: Unfold broadcastAligned at the outermost cons/cons case
      simp only [Tabulated.broadcastAligned]
      by_cases hd : (1 : Nat) = d
      · subst hd
        simp only [dite_true]
        refine Vector.ext fun i hi => ?_
        have hi0 : i = 0 := by omega
        subst hi0
        simp only [Vector.getElem_map, Tabulated.wrapSingletons]
        exact ih h23.tail h13.tail
      · simp only [dif_neg hd, Tabulated.wrapSingletons]
        congr 1
        exact ih h23.tail h13.tail

private theorem wrap_zipWith {α β γ : Type} {σ : Shape}
    (t1 : Tabulated σ α) (t2 : Tabulated σ β) (f : α → β → γ) (n : Nat) :
    Tabulated.wrapSingletons σ (Id.run (Tabulated.zipWithM t1 t2
      (fun a b => pure (f a b)))) n =
    Id.run (Tabulated.zipWithM
      (Tabulated.wrapSingletons σ t1 n)
      (Tabulated.wrapSingletons σ t2 n)
      (fun a b => pure (f a b))) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    refine Vector.ext fun i hi => ?_
    have hi0 : i = 0 := by omega
    subst hi0
    simp only [Tabulated.zipWithM, Id.run, Vector_mapM_id_eq, Vector.getElem_zipWith]
    exact ih

private theorem broadcastAligned_zipWith {α β γ : Type} {σₛ σₜ : Shape}
    (t1 : Tabulated σₛ α) (t2 : Tabulated σₛ β) (f : α → β → γ)
    (h : Shape.BroadcastableAligned σₛ σₜ) :
    Tabulated.broadcastAligned σₛ σₜ
      (Id.run (Tabulated.zipWithM t1 t2 (fun a b => pure (f a b)))) h =
    Id.run (Tabulated.zipWithM
      (Tabulated.broadcastAligned σₛ σₜ t1 h)
      (Tabulated.broadcastAligned σₛ σₜ t2 h)
      (fun a b => pure (f a b))) := by
  match σₛ, σₜ, t1, t2, h with
  | [], [], _, _, _ => rfl
  | [], _ :: _, _, _, h => exact nomatch h
  | _ :: _, [], _, _, h => exact nomatch h
  | df :: dfs, dt :: dts, xs, ys, h =>
    simp only [Tabulated.broadcastAligned]
    by_cases heq : df = dt
    · subst heq
      simp only [↓reduceDIte, Tabulated.zipWithM, Id.run, Vector_mapM_id_eq]
      refine Vector.ext fun i hi => ?_
      simp only [Vector.getElem_map, Vector.getElem_zipWith]
      exact broadcastAligned_zipWith xs[i] ys[i] f h.tail
    · simp only [heq, ↓reduceDIte, Tabulated.zipWithM, Id.run, Vector_mapM_id_eq]
      have h0 : 0 < df := by have := h.one_of_ne heq; omega
      refine Vector.ext fun i hi => ?_
      simp only [Vector.getElem_replicate, Vector.getElem_zipWith]
      exact broadcastAligned_zipWith (xs[0]'h0) (ys[0]'h0) f h.tail

private theorem wrap_zipWith3 {α β γ δ : Type} {σ : Shape}
    (t1 : Tabulated σ α) (t2 : Tabulated σ β) (t3 : Tabulated σ γ)
    (f : α → β → γ → δ) (n : Nat) :
    Tabulated.wrapSingletons σ (Id.run (Tabulated.zipWith3M t1 t2 t3
      (fun a b c => pure (f a b c)))) n =
    Id.run (Tabulated.zipWith3M
      (Tabulated.wrapSingletons σ t1 n)
      (Tabulated.wrapSingletons σ t2 n)
      (Tabulated.wrapSingletons σ t3 n)
      (fun a b c => pure (f a b c))) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    refine Vector.ext fun i hi => ?_
    have hi0 : i = 0 := by omega
    subst hi0
    simp only [Tabulated.zipWith3M, Id.run, Vector_mapM_id_eq, Vector.getElem_zipWith, Vector.getElem_zip]
    exact ih

private theorem broadcastAligned_zipWith3 {α β γ δ : Type} {σₛ σₜ : Shape}
    (t1 : Tabulated σₛ α) (t2 : Tabulated σₛ β) (t3 : Tabulated σₛ γ)
    (f : α → β → γ → δ)
    (h : Shape.BroadcastableAligned σₛ σₜ) :
    Tabulated.broadcastAligned σₛ σₜ
      (Id.run (Tabulated.zipWith3M t1 t2 t3 (fun a b c => pure (f a b c)))) h =
    Id.run (Tabulated.zipWith3M
      (Tabulated.broadcastAligned σₛ σₜ t1 h)
      (Tabulated.broadcastAligned σₛ σₜ t2 h)
      (Tabulated.broadcastAligned σₛ σₜ t3 h)
      (fun a b c => pure (f a b c))) := by
  match σₛ, σₜ, t1, t2, t3, h with
  | [], [], _, _, _, _ => rfl
  | [], _ :: _, _, _, _, h => exact nomatch h
  | _ :: _, [], _, _, _, h => exact nomatch h
  | df :: dfs, dt :: dts, xs, ys, zs, h =>
    simp only [Tabulated.broadcastAligned]
    by_cases heq : df = dt
    · subst heq
      simp only [↓reduceDIte, Tabulated.zipWith3M, Id.run, Vector_mapM_id_eq]
      refine Vector.ext fun i hi => ?_
      simp only [Vector.getElem_map, Vector.getElem_zipWith, Vector.getElem_zip]
      exact broadcastAligned_zipWith3 xs[i] ys[i] zs[i] f h.tail
    · simp only [heq, ↓reduceDIte, Tabulated.zipWith3M, Id.run, Vector_mapM_id_eq]
      have h0 : 0 < df := by have := h.one_of_ne heq; omega
      refine Vector.ext fun i hi => ?_
      simp only [Vector.getElem_replicate, Vector.getElem_zipWith, Vector.getElem_zip]
      exact broadcastAligned_zipWith3 (xs[0]'h0) (ys[0]'h0) (zs[0]'h0) f h.tail

/-! ### index_transpose helpers -/

/-- Flattening the result of `unflatten` recovers the original list. -/
private theorem flatten_of_unflatten {α : Type} (σ : Shape) (xs : List α)
    (hlen : xs.length = σ.volume) :
    Tabulated.flatten (Tabulated.unflatten σ xs hlen) = xs :=
  Tabulated.flatten_unflatten_eq σ xs hlen

/-- `ravelIndex` at all-zeros indices is always 0. -/
private theorem ravelIndex_replicate_zeros (k : Nat) :
    Shape.ravelIndex (Shape.computeStrides (List.replicate k 1))
        (List.replicate k 0) = 0 := by
  induction k with
  | zero => rfl
  | succ k' ih =>
    let ones : Shape := List.replicate k' 1
    change Shape.ravelIndex
      (ones.volume :: Shape.computeStrides ones)
      (0 :: List.replicate k' 0) = 0
    rw [Shape.ravelIndex_cons, Nat.mul_zero, Nat.zero_add]
    exact ih

/-- For Tabulated tensors, `index` equals indexing into the flat list at `ravelIndex`. -/
private theorem index_eq_flatten_ravelIndex {α : Type} (σ : Shape) (t : Tabulated σ α)
    (indices : Indices σ) :
    Tensor.Operations.index σ t indices =
    (Tabulated.flatten t)[Shape.ravelIndex (Shape.computeStrides σ) indices.values]'(by
      rw [Tabulated.flatten_length]
      exact Shape.ravelIndex_lt_volume σ indices.values indices.length_eq indices.valid) := by
  induction σ with
  | nil =>
    unfold Tensor.Operations.index
    show Tabulated.scalar t = _
    rw [show Tabulated.scalar t = t from rfl]; congr 1
  | cons d ds ih =>
    unfold Tensor.Operations.index
    let ds : Shape := ds
    rw [ih (Tensor.Operations.get t indices.head) indices.tail]
    -- Goal: (flatten (get t indices.head))[ravelIndex(strides ds, tail.values)]'_ =
    --       (flatten t)[ravelIndex(strides(d::ds), indices.values)]'_
    -- Bridge to getElem? to avoid dependent bound proof issues
    have hne : indices.values ≠ [] := by intro h; have := indices.length_eq; simp [h] at this
    have hvals : indices.values = indices.head.val :: indices.tail.values := by
      have := (List.cons_head_tail hne).symm
      simp only [Indices.head, Indices.tail, List.head_eq_getElem] at this ⊢
      exact this
    have hr_lt := Shape.ravelIndex_lt_volume ds indices.tail.values indices.tail.length_eq indices.tail.valid
    apply Option.some.inj
    rw [← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem, hvals,
      show Shape.computeStrides (d :: ds) = ds.volume :: Shape.computeStrides ds from rfl,
      Shape.ravelIndex_cons]
    exact (Tabulated.flatten_getElem_slice? t indices.head.val indices.head.isLt _ hr_lt).symm

/-- Key property of transpose: flat indexing of transposed tensor equals flat indexing of
    original tensor at permuted indices. -/
private theorem transpose_flat_spec {α : Type} {σ : Shape}
    (t : Tabulated σ α) (perm : Permutation) (h : Permutation.Valid perm σ.rank)
    (indices : List Nat)
    (hlen : indices.length = (σ.permute perm h).rank)
    (hvalid : ∀ i (hi : i < (σ.permute perm h).rank),
      indices[i]'(hlen ▸ hi) < (σ.permute perm h)[i]) :
    (Tabulated.flatten (Tabulated.transpose t perm h))[Shape.ravelIndex
      (Shape.computeStrides (σ.permute perm h)) indices]? =
    (Tabulated.flatten t)[Shape.ravelIndex (Shape.computeStrides σ)
      ((perm.invert h).map (fun p => indices.getD p 0))]? := by
  have hflatLen := Tabulated.flatten_length t
  have hSizeEq := Shape.volume_permute σ perm h
  have hrav_lt := Shape.ravelIndex_lt_volume (σ.permute perm h) indices hlen hvalid
  have hsize_pos : 0 < σ.volume := by omega
  have hpos : 0 < (Tabulated.flatten t).length := by omega
  -- Unfold transpose = transposeByFlattening
  show (Tabulated.flatten (Tensor.Operations.transposeByFlattening
    Tabulated.flatten Tabulated.flatten_length Tabulated.unflatten
    t perm h))[Shape.ravelIndex
    (Shape.computeStrides (σ.permute perm h)) indices]? =
    (Tabulated.flatten t)[Shape.ravelIndex (Shape.computeStrides σ)
    ((perm.invert h).map (fun p => indices.getD p 0))]?
  unfold Tensor.Operations.transposeByFlattening
  simp only [hpos, ↓reduceDIte]
  rw [flatten_of_unflatten (σ.permute perm h) _ (List.length_ofFn ..)]
  rw [List.getElem?_ofFn (i := Shape.ravelIndex
    (Shape.computeStrides (σ.permute perm h)) indices)]
  simp only [hrav_lt, ↓reduceDIte]
  simp only [Shape.unravelIndex_ravelIndex (σ.permute perm h) indices hlen hvalid]
  -- LHS is `some flat[...]` with `h.invert`, RHS is `flat[...]?` with `perm.invert h`
  -- Prove inv-mapped indices are in bounds for σ
  have hlen_inv := h.invert_length
  have hinv_len : ((perm.invert h).map (fun p => indices.getD p 0)).length =
      σ.rank := by rw [List.length_map, hlen_inv]
  have hlen_out : (σ.permute perm h).rank = σ.rank := by
    simp [Shape.permute, List.length_pmap, h.length_eq]
  have hlen_idx : indices.length = σ.rank := by omega
  have hinv_valid : ∀ i (hi : i < σ.rank),
      ((perm.invert h).map (fun p => indices.getD p 0))[i]'(by
        rw [List.length_map, hlen_inv]; exact hi) <
      σ[i] := by
    intro i hi
    have hi_inv : i < (perm.invert h).length := by rw [hlen_inv]; exact hi
    have hinv_lt : (perm.invert h)[i]'hi_inv < σ.rank :=
      h.invert_forall_lt _ (List.getElem_mem hi_inv)
    have hinv_lt_idx : (perm.invert h)[i]'hi_inv < indices.length := by
      rw [hlen_idx]; exact hinv_lt
    have h_idx := hvalid ((perm.invert h)[i]'hi_inv) (by rw [hlen_out]; exact hinv_lt)
    rw [Shape.permute_getElem_invert σ perm h i hi] at h_idx
    -- h_idx : indices[inv[i]] < σ[i]
    simp only [List.getElem_map]
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hinv_lt_idx, Option.getD_some]
    exact h_idx
  have hInBound : Shape.ravelIndex (Shape.computeStrides σ)
      ((perm.invert h).map (fun p => indices.getD p 0)) <
      (Tabulated.flatten t).length := by
    rw [hflatLen]; exact Shape.ravelIndex_lt_volume σ _ hinv_len hinv_valid
  rw [List.getElem?_eq_getElem hInBound]; rfl

/-- `Tensor.LawfulOperations` instance for `Tabulated`. -/
instance : Tensor.LawfulOperations Tabulated where
  index_eq_getElem_flatten := fun t indices => index_eq_flatten_ravelIndex _ t indices
  flatten_unflatten := fun σ xs h => Tabulated.flatten_unflatten_eq σ xs h
  broadcast_map := by
    intro α β σₛ σₜ t f h
    show Tabulated.broadcast (Tabulated.map t f) σₜ h =
      Tabulated.map (Tabulated.broadcast t σₜ h) f
    simp only [Tabulated.broadcast]
    rw [wrapSingletons_map, broadcastAligned_map]
  scalar_map := by intros; rfl
  get_map := by
    intro d ds α β t f i
    simp only [Tensor.Operations.get, Tensor.Operations.map, Tabulated.get, Tabulated.map]
    simp [Vector.getElem_map]
  scalar_fill := by intros; rfl
  get_fill := by
    intro α v d ds i
    simp only [Tensor.Operations.get, Tensor.Operations.fill, Tabulated.get, Tabulated.fill]
    simp [Vector.getElem_replicate]
  scalar_zipWith := by intros; rfl
  get_zipWith := by
    intro d ds α β γ t1 t2 f i
    show Tabulated.get (Id.run (Tabulated.zipWithM t1 t2
      (fun a b => pure (f a b)))) i = _
    simp only [Tabulated.zipWithM, Id.run, Tabulated.get, Vector_mapM_id_eq]
    exact Vector.getElem_zipWith i.isLt
  scalar_zipWith3 := by intros; rfl
  get_zipWith3 := by
    intro d ds α β γ δ t1 t2 t3 f i
    show Tabulated.get (Id.run (Tabulated.zipWith3M t1 t2 t3
      (fun a b c => pure (f a b c)))) i =
      Id.run (Tabulated.zipWith3M (Tabulated.get t1 i) (Tabulated.get t2 i)
        (Tabulated.get t3 i) (fun a b c => pure (f a b c)))
    simp only [Tabulated.zipWith3M, Id.run, Tabulated.get, Vector_mapM_id_eq]
    simp [Vector.getElem_zipWith, Vector.getElem_zip]
  broadcast_scalar := by
    intro α t σ h
    show Tabulated.broadcast t σ h = Tabulated.fill (Tabulated.scalar t) σ
    simp only [Tabulated.broadcast, Tabulated.scalar]
    induction σ with
    | nil => rfl
    | cons d ds ih =>
      simp only [Tabulated.fill, Shape.pad, List.length_nil, Nat.sub_zero, List.length_cons]
      dsimp only [Tabulated.wrapSingletons]
      change (Tabulated.broadcastAligned (1 :: (List.replicate ds.length 1 ++ [])) (d :: ds)
        (Vector.singleton (Tabulated.wrapSingletons [] t ds.length)) h) =
        Vector.replicate d (Tabulated.fill t ds)
      simp only [Tabulated.broadcastAligned]
      split
      · rename_i heq; subst heq
        ext i hi
        simp only [Vector.getElem_map, Vector.getElem_replicate]
        have hi0 : i = 0 := by omega
        subst hi0; simp; exact ih _
      · congr 1; simp; exact ih _
  broadcast_index := by
    intro α σₛ σₜ data h indices
    -- Decompose broadcast = broadcastAligned ∘ wrapSingletons
    show ∃ indices', Tensor.Operations.index σₜ
        (Tabulated.broadcast data σₜ h) indices =
      Tensor.Operations.index σₛ data indices'
    simp only [Tabulated.broadcast]
    obtain ⟨padIdx, hpad⟩ := broadcastAligned_index
      (Tabulated.wrapSingletons σₛ data (σₜ.rank - σₛ.rank))
      h indices
    obtain ⟨origIdx, hwrap⟩ := wrapSingletons_index σₛ data
      (σₜ.rank - σₛ.rank) padIdx
    exact ⟨origIdx, hpad.trans hwrap⟩
  broadcast_broadcastable_refl := by
    intro α σ t
    show Tabulated.broadcast t σ (Shape.Broadcastable.refl σ) = t
    have hpad : Shape.pad σ σ.length = σ := Shape.pad_eq_of_rank_eq σ σ.length rfl
    have hwrap_eq : σ.length - σ.length = 0 := Nat.sub_self σ.length
    have bc_aligned : Shape.BroadcastableAligned σ σ := Shape.BroadcastableAligned.refl σ
    have hbc : Tabulated.broadcast t σ (Shape.Broadcastable.refl σ) =
        Tabulated.broadcastAligned σ σ (Tabulated.wrapSingletons σ t 0) bc_aligned :=
      broadcastAligned_shape_heq hpad (wrapSingletons_nat_heq t hwrap_eq)
        (Shape.Broadcastable.refl σ) bc_aligned
    rw [hbc, wrapSingletons_zero t]
    exact broadcastAligned_refl t bc_aligned
  get_broadcast_cons_left := by
    intro α σ d t h i
    show Tabulated.get (Tabulated.broadcast t (d :: σ) h) i = t
    have hpad : Shape.pad σ (d :: σ).length = 1 :: σ := by simp [Shape.pad]
    have hlen_eq : (d :: σ).length - σ.length = 1 := by simp
    have h' : Shape.BroadcastableAligned (1 :: σ) (d :: σ) := by rw [← hpad]; exact h
    have hbc_eq : Tabulated.broadcast t (d :: σ) h =
        Tabulated.broadcastAligned (1 :: σ) (d :: σ)
          (Tabulated.wrapSingletons σ t 1) h' :=
      broadcastAligned_shape_heq hpad (wrapSingletons_nat_heq t hlen_eq) h h'
    simp only [hbc_eq]
    show Tensor.Operations.get
      (Tabulated.broadcastAligned (1 :: σ) (d :: σ)
        (Tabulated.wrapSingletons σ t 1) h') i = t
    by_cases hd : (1 : Nat) = d
    · subst hd
      rw [broadcastAligned_get_eq _ h' i i.isLt, broadcastAligned_refl]
      have hi0 : i = ⟨0, by omega⟩ := Fin.ext (by omega)
      subst hi0; rfl
    · rw [broadcastAligned_get_one _ h' hd i i.isLt, broadcastAligned_refl]
      rfl
  broadcast_broadcast := by
    intro α σ₁ σ₂ σ₃ t h12 h23 h13
    show Tabulated.broadcast (Tabulated.broadcast t σ₂ h12) σ₃ h23 =
      Tabulated.broadcast t σ₃ h13
    simp only [Tabulated.broadcast, Shape.pad]
    -- Get length inequalities from broadcastability
    have hlen12 : σ₁.rank ≤ σ₂.rank := by
      have := Shape.BroadcastableAligned.rank_eq h12
      simp [Shape.pad_rank] at this; omega
    have hlen23 : σ₂.rank ≤ σ₃.rank := by
      have := Shape.BroadcastableAligned.rank_eq h23
      simp [Shape.pad_rank] at this; omega
    -- The h13 alignment proof for wrapBroadcast_trans, transported across shape arithmetic
    have h13' : Shape.BroadcastableAligned
        (List.replicate (σ₃.rank - σ₂.rank) 1 ++
          (List.replicate (σ₂.rank - σ₁.rank) 1 ++ σ₁)) σ₃ := by
      rw [← List.append_assoc, List.replicate_append_replicate,
        show σ₃.rank - σ₂.rank + (σ₂.rank - σ₁.rank) = σ₃.rank - σ₁.rank from by omega]
      exact h13
    rw [wrapBroadcast_trans _ _ _ _ h13']
    -- Goal: broadcastAligned (rep a ++ (rep b ++ σ₁)) σ₃ D_nested h13'
    --     = broadcastAligned (rep c ++ σ₁) σ₃ D_single h13
    -- Use broadcastAligned_shape_heq with shape equality + data HEq
    exact broadcastAligned_shape_heq
      (by rw [← List.append_assoc, List.replicate_append_replicate,
            show σ₃.rank - σ₂.rank + (σ₂.rank - σ₁.rank) =
              σ₃.rank - σ₁.rank from by omega])
      (wrap_wrap_collapse_heq t (σ₃.rank - σ₂.rank) (σ₂.rank - σ₁.rank) |>.trans
        (wrapSingletons_nat_heq t (by omega)))
      h13' h13
  broadcast_zipWith := by
    intro α β γ σₛ σₜ t1 t2 f h
    show Tabulated.broadcast (Id.run (Tabulated.zipWithM t1 t2
      (fun a b => pure (f a b)))) σₜ h =
      Id.run (Tabulated.zipWithM
        (Tabulated.broadcast t1 σₜ h)
        (Tabulated.broadcast t2 σₜ h)
        (fun a b => pure (f a b)))
    simp only [Tabulated.broadcast]
    rw [wrap_zipWith, broadcastAligned_zipWith]
  broadcast_zipWith3 := by
    intro α β γ δ σₛ σₜ t1 t2 t3 f h
    show Tabulated.broadcast (Id.run (Tabulated.zipWith3M t1 t2 t3
      (fun a b c => pure (f a b c)))) σₜ h =
      Id.run (Tabulated.zipWith3M
        (Tabulated.broadcast t1 σₜ h)
        (Tabulated.broadcast t2 σₜ h)
        (Tabulated.broadcast t3 σₜ h)
        (fun a b c => pure (f a b c)))
    simp only [Tabulated.broadcast]
    rw [wrap_zipWith3, broadcastAligned_zipWith3]
  get_mapSlices := by
    intro d ds ds' α β xs f i
    show (Id.run (Vector.mapM (m := Id) (pure <| f ·) xs))[i.val] = f xs[i.val]
    rw [Vector.mapM_pure (m := Id) f]
    show (Vector.map f xs)[↑i] = f xs[↑i]
    simp [Vector.getElem_map]
  squeeze_zero := by
    intro α ds t _
    rfl
  get_squeeze := by
    intro d ds α t axis hone i
    show Tabulated.get (Tabulated.squeeze t ⟨axis.val + 1, Nat.succ_lt_succ axis.isLt⟩
        (by simpa [List.getElem_cons_succ] using hone)) i =
      Tabulated.squeeze (Tabulated.get t i) axis hone
    have key : Tabulated.squeeze t ⟨axis.val + 1, Nat.succ_lt_succ axis.isLt⟩
               (by simpa [List.getElem_cons_succ] using hone) =
             Vector.map (fun elem => Tabulated.squeeze elem axis hone) t := rfl
    rw [key]
    simp [Tabulated.get, Vector.getElem_map]
  get_broadcast_eq := by
    intro α d ds σₜ t bc bc' h_len i
    show Tabulated.get (Tabulated.broadcast t (d :: σₜ) bc) i =
      Tabulated.broadcast (Tabulated.get t i) σₜ bc'
    have hpad : Shape.pad (d :: ds) (d :: σₜ).length = d :: ds :=
      Shape.pad_eq_of_rank_eq _ _ (by simp; omega)
    have hpad' : Shape.pad ds σₜ.rank = ds :=
      Shape.pad_eq_of_rank_eq _ _ h_len
    have hwrap_eq : (d :: σₜ).length - (d :: ds).length = 0 := by simp [Shape.rank] at h_len ⊢; omega
    have hwrap_eq' : σₜ.rank - ds.rank = 0 := by omega
    have bc_aligned : Shape.BroadcastableAligned (d :: ds) (d :: σₜ) := by
      rw [← hpad]; exact bc
    have bc'_aligned : Shape.BroadcastableAligned ds σₜ := by
      rw [← hpad']; exact bc'
    have hbc : Tabulated.broadcast t (d :: σₜ) bc =
        Tabulated.broadcastAligned (d :: ds) (d :: σₜ)
          (Tabulated.wrapSingletons (d :: ds) t 0) bc_aligned :=
      broadcastAligned_shape_heq hpad (wrapSingletons_nat_heq t hwrap_eq) bc bc_aligned
    have hbc' : Tabulated.broadcast (Tabulated.get t i) σₜ bc' =
        Tabulated.broadcastAligned ds σₜ
          (Tabulated.wrapSingletons ds (Tabulated.get t i) 0) bc'_aligned :=
      broadcastAligned_shape_heq hpad' (wrapSingletons_nat_heq (Tabulated.get t i) hwrap_eq') bc' bc'_aligned
    simp only [hbc, hbc']
    show Tensor.Operations.get
      (Tabulated.broadcastAligned (d :: ds) (d :: σₜ)
        (Tabulated.wrapSingletons (d :: ds) t 0) bc_aligned) i =
      Tabulated.broadcastAligned ds σₜ
        (Tabulated.wrapSingletons ds (Tabulated.get t i) 0) bc'_aligned
    rw [broadcastAligned_get_eq _ bc_aligned i.val i.isLt]; rfl
  get_broadcast_one := by
    intro α ds σₜ d t bc bc' h_len i
    show Tabulated.get (Tabulated.broadcast t (d :: σₜ) bc) i =
      Tabulated.broadcast (Tabulated.get t ⟨0, by omega⟩) σₜ bc'
    have hpad : Shape.pad (1 :: ds) (d :: σₜ).length = 1 :: ds :=
      Shape.pad_eq_of_rank_eq _ _ (by simp; omega)
    have hpad' : Shape.pad ds σₜ.rank = ds :=
      Shape.pad_eq_of_rank_eq _ _ h_len
    have hwrap_eq : (d :: σₜ).length - (1 :: ds).length = 0 := by simp [Shape.rank] at h_len ⊢; omega
    have hwrap_eq' : σₜ.rank - ds.rank = 0 := by omega
    have bc_aligned : Shape.BroadcastableAligned (1 :: ds) (d :: σₜ) := by
      rw [← hpad]; exact bc
    have bc'_aligned : Shape.BroadcastableAligned ds σₜ := by
      rw [← hpad']; exact bc'
    have hbc : Tabulated.broadcast t (d :: σₜ) bc =
        Tabulated.broadcastAligned (1 :: ds) (d :: σₜ)
          (Tabulated.wrapSingletons (1 :: ds) t 0) bc_aligned :=
      broadcastAligned_shape_heq hpad (wrapSingletons_nat_heq t hwrap_eq) bc bc_aligned
    have hbc' : Tabulated.broadcast (Tabulated.get t ⟨0, by omega⟩) σₜ bc' =
        Tabulated.broadcastAligned ds σₜ
          (Tabulated.wrapSingletons ds (Tabulated.get t ⟨0, by omega⟩) 0) bc'_aligned :=
      broadcastAligned_shape_heq hpad' (wrapSingletons_nat_heq (Tabulated.get t ⟨0, by omega⟩) hwrap_eq') bc' bc'_aligned
    simp only [hbc, hbc']
    show Tensor.Operations.get
      (Tabulated.broadcastAligned (1 :: ds) (d :: σₜ)
        (Tabulated.wrapSingletons (1 :: ds) t 0) bc_aligned) i =
      Tabulated.broadcastAligned ds σₜ
        (Tabulated.wrapSingletons ds (Tabulated.get t ⟨0, by omega⟩) 0) bc'_aligned
    by_cases hd : (1 : Nat) = d
    · subst hd
      have hi : i = ⟨0, by omega⟩ := Fin.ext (by omega)
      subst hi
      rw [broadcastAligned_get_eq _ bc_aligned 0 (by omega)]; rfl
    · rw [broadcastAligned_get_one _ bc_aligned hd i.val i.isLt]; rfl
  index_reshape_replicate_ones := by
    intro α n k t h i hi
    -- Both sides reduce to flat list indexing via index_eq_flatten_ravelIndex
    rw [index_eq_flatten_ravelIndex (List.replicate k 1) _ _,
        index_eq_flatten_ravelIndex [n] t _]
    -- Bridge to getElem? to avoid dependent bound issues
    apply Option.some.inj
    rw [← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem]
    -- Simplify index values
    have h_zeros : (Indices.zeros (List.replicate k 1)
        (fun j hj => by rw [List.getElem_replicate]; omega)).values = List.replicate k 0 := by
      simp [Indices.zeros]
    have h_rhs_vals : (Indices.cons i hi Indices.nil).values = [i] := rfl
    rw [h_zeros, h_rhs_vals]
    -- ravelIndex simplifications
    rw [ravelIndex_replicate_zeros k]
    have h_rav_i : Shape.ravelIndex (Shape.computeStrides [n]) [i] = i := by
      simp [Shape.computeStrides, Shape.ravelIndex, Shape.volume]
    rw [h_rav_i]
    -- Goal: (flatten (get (reshape ...) ⟨i, hi⟩))[0]? = (flatten t)[i]?
    -- Step 1: Use flatten_getElem_slice? to relate flat indexing to slice indexing
    have h_slice := Tabulated.flatten_getElem_slice?
      (@Tensor.reshape Tabulated _ [n] α t (n :: List.replicate k 1) h) i hi 0
      (by rw [Shape.volume_replicate_one]; omega)
    rw [Shape.volume_replicate_one, Nat.one_mul, Nat.add_zero] at h_slice
    -- Normalize Tensor.Operations.get to Tabulated.get for rw matching
    show (Tabulated.flatten (Tabulated.get
      (@Tensor.reshape Tabulated _ [n] α t (n :: List.replicate k 1) h) ⟨i, hi⟩))[0]? =
      (Tabulated.flatten t)[i]?
    rw [← h_slice]
    -- Step 2: flatten (reshape ...) = flatten t (reshape preserves flat list)
    have h_flat : Tabulated.flatten
      (@Tensor.reshape Tabulated _ [n] α t (n :: List.replicate k 1) h) =
      Tabulated.flatten t :=
      flatten_of_unflatten _ _ ((Tabulated.flatten_length t).trans h)
    rw [h_flat]
  index_transpose := by
    intro α σ t perm h indices
    let outShape := σ.permute perm h
    let vals := indices.values
    have hlen : vals.length = outShape.rank := indices.length_eq
    have hvalid_elem : ∀ i (hi : i < outShape.rank),
        vals[i]'(by rw [hlen]; exact hi) < outShape[i] := indices.valid
    -- Both sides reduce to flat indexing via index_eq_flatten_ravelIndex
    rw [index_eq_flatten_ravelIndex outShape (Tensor.Operations.transpose t perm h) indices,
        index_eq_flatten_ravelIndex σ t indices.invPermute]
    -- Convert total getElem to getElem? for transpose_flat_spec
    apply Option.some.inj
    rw [← List.getElem?_eq_getElem, ← List.getElem?_eq_getElem]
    exact transpose_flat_spec t perm h vals hlen hvalid_elem

/-- `LawfulOfTabulated` instance for `Tabulated`. -/
instance : Tensor.Operations.LawfulOfTabulated Tabulated where
  scalar_ofTabulated := by intros; rfl
  get_ofTabulated := by intros; rfl

end Instances

end Palladium.TensorLib
