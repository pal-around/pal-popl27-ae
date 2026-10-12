import Palladium.PAL.Syntax

namespace Palladium.Compiler

open Palladium.Util
open Palladium.PAL

/-! ### Naming Constants -/

/-- Prefix for unnormalized weight variables. -/
def weightPrefix : String := reserved ++ "_weight_"
/-- Prefix for normalized probability variables. -/
def probPrefix : String := reserved ++ "_prob_"
/-- Global name for the posterior tensor variable. -/
def probabilityTensor : String := reserved ++ "_probability_tensor"

/-- Construct the weight variable name for a random variable. -/
def weightName (x : String) : String := weightPrefix ++ x
/-- Construct the probability variable name for a random variable. -/
def probName (x : String) : String := probPrefix ++ x
/-- Name suffix for the weight and probability variables of a joint `sample`. -/
def sampleName (names : List String) : String := String.intercalate "_" names

/-!
### Axis Management

The compiler maintains a joint probability tensor (`__pal_probability_tensor`) representing
the distribution over all sampled variables.

**Layout Convention (head = newest)**:
New variables are added at the *head* of the random context via `.cons`.
`Δ.toShape` maps `.cons x (.fin n) rest` to `n :: rest.toShape`,
so the head variable is at axis 0.

- The most recently sampled variable is at axis 0 (head of keys).
- The i-th variable in `Δ.keys` (0-indexed, newest-first) is at axis `i`.

Example:
After `x ← sample …; y ← sample …; z ← sample …`:
- `RandomContext` is `.cons z D3 (.cons y D2 (.cons x D1 .nil))`
- `keys = [z, y, x]`, shape = `[z_size, y_size, x_size]`
- `z` (idx 0) ↦ axis 0, `y` (idx 1) ↦ axis 1, `x` (idx 2) ↦ axis 2.

This convention allows JAX broadcasting to work naturally when multiplying the
current probability tensor by a new variable's probability tensor.
-/

/-- Locate a random variable in context and compute its tensor axis.
    With the head-newest convention, the context index equals the tensor axis.
    Total: requires a membership proof. -/
def randomAxis (ctx : RandomContext) (name : String) (D : Discrete)
    (h : ⟨name, D⟩ ∈ ctx) : Nat :=
  (ctx.keys.findIdx? (· == name)).get (by
    have hmem : name ∈ ctx.keys := by rw [List.mem_keys]; exact ⟨D, h⟩
    have hany : ctx.keys.any (· == name) = true :=
      List.any_eq_true.mpr ⟨name, hmem, beq_self_eq_true name⟩
    rw [List.findIdx?_isSome]
    exact hany)

/-- `randomAxis` agrees with `findIdx?`. -/
theorem randomAxis_eq_findIdx? (ctx : RandomContext) (name : String) (D : Discrete)
    (h : ⟨name, D⟩ ∈ ctx) :
    ctx.keys.findIdx? (· == name) = some (randomAxis ctx name D h) := by
  simp only [randomAxis, Option.some_get]

/-- The axis returned by `randomAxis` is in bounds. -/
theorem randomAxis_lt (ctx : RandomContext) (name : String) (D : Discrete)
    (h : ⟨name, D⟩ ∈ ctx) : randomAxis ctx name D h < ctx.keys.length :=
  (List.findIdx?_eq_some_iff_getElem.mp (randomAxis_eq_findIdx? ctx name D h)).1

/-- At the axis returned by `randomAxis`, the key is `name`. -/
theorem randomAxis_keysGetElem (ctx : RandomContext) (name : String) (D : Discrete)
    (h : ⟨name, D⟩ ∈ ctx) :
    ctx.keys[randomAxis ctx name D h]'(randomAxis_lt ctx name D h) = name := by
  exact eq_of_beq (List.findIdx?_eq_some_iff_getElem.mp (randomAxis_eq_findIdx? ctx name D h)).2.1

/-! ### String Inequality Infrastructure -/

/-! #### Reserved-Prefix Theorems -/

theorem isReserved_probabilityTensor : isReserved probabilityTensor = true := by decide

/-! #### Cross-Prefix Distinctness -/

theorem ne_of_isReserved {a b : String}
    (ha : isReserved a = true) (hb : isReserved b = false) : a ≠ b := by
  intro heq; subst heq; simp [ha] at hb

/- A prefixed string cannot equal a concrete string when they differ at a given index. -/
private theorem ne_prefix_of_concrete (pa : String) (b : String) (idx : Nat)
    (hidx_a : idx < pa.toList.length)
    (hne : pa.toList[idx]? ≠ b.toList[idx]?)
    {sa : String} : pa ++ sa ≠ b := by
  intro heq
  have h : (pa ++ sa).toList[idx]? = b.toList[idx]? :=
    congrArg (fun s => s.toList[idx]?) heq
  rw [String.toList_append, List.getElem?_append_left hidx_a] at h
  exact absurd h hne

theorem weightName_ne_probabilityTensor {x : String} : weightName x ≠ probabilityTensor :=
  ne_prefix_of_concrete weightPrefix probabilityTensor 6
    (by decide) (by decide)

theorem probName_ne_probabilityTensor {x : String} : probName x ≠ probabilityTensor :=
  ne_prefix_of_concrete probPrefix probabilityTensor 10
    (by decide) (by decide)

end Palladium.Compiler
