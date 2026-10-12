import Util.Context
import Util.Denote

namespace Palladium.Util

/-!
Probability mass functions live in the shared environment module because they are
just weights assigned to coherent variable bindings.
 -/

/-- `PMF ctx` maps each assignment `σ` for `ctx` to a weight.
We write `p(σ) : PMF(Σ)`. -/
abbrev PMF {α : Type} (denote : α → Type) (ctx : Context α) : Type :=
  Environment denote ctx -> Float

/-- `ConditionalPMF cond rand` maps each conditioning assignment `δ`
to a PMF over `rand` assignments `σ`.
We write `p(σ | δ) : PMF(Σ | Δ)`. -/
abbrev ConditionalPMF {α : Type} (denote : α → Type)
    (cond : Context α) (rand : Context α) : Type :=
  Environment denote cond -> PMF denote rand

instance {α : Type} {denote : α → Type} {ctx : Context α} :
    GetElem (PMF denote ctx) (Environment denote ctx) Float (fun _ _ => True) where
  getElem μ env _ := μ env

end Palladium.Util
