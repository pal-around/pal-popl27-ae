import TensorLib.Operations.Basics
import TensorLib.Operations.Tabulated
import TensorLib.Tensor
import Std

/-! # Tabulated Interop

Construction of tensors in any layout from `Tabulated` data, used to embed literal
arrays in Jay programs. -/

namespace Palladium.TensorLib

namespace Tensor.Operations

/-- Conversion from `Tabulated` to another layout. -/
class OfTabulated (L : Tensor.Layout) where
  /-- Convert from Tabulated form (for embedding literal data from AST). -/
  ofTabulated : {σ : Shape} → Tabulated σ α → L σ α

/-- Convert from Tabulated form (for embedding literal data from AST). -/
def ofTabulated {L : Tensor.Layout} {σ : Shape} {α : Type}
    [Tensor.Operations.OfTabulated L]
    (data : Tabulated σ α) : L σ α :=
  Tensor.Operations.OfTabulated.ofTabulated data

/-- Laws for `OfTabulated` interaction with `Operations`. -/
class LawfulOfTabulated (L : Tensor.Layout)
    [Tensor.Operations L] [Tensor.Operations.OfTabulated L] : Prop where
  /-- Extracting the scalar from a rank-0 `ofTabulated` tensor recovers the original value. -/
  scalar_ofTabulated : ∀ {α : Type} (data : Tabulated [] α),
    Tensor.Operations.scalar (Tensor.Operations.ofTabulated (L := L) data) = data
  /-- Indexing an `ofTabulated` tensor distributes through `get`. -/
  get_ofTabulated : ∀ {d : Nat} {ds : Shape} {α : Type}
    (data : Tabulated (d :: ds) α) (i : Fin d),
    Tensor.Operations.get (Tensor.Operations.ofTabulated (L := L) data) i =
      Tensor.Operations.ofTabulated (data[i])

end Tensor.Operations

namespace Tensor

/-- Construct a tensor in any layout from tabulated data. -/
def ofTabulated
    {L : Tensor.Layout} {σ : Shape} {α : Type}
    [Tensor.Operations.OfTabulated L]
    (data : Tabulated σ α) :
    Tensor (L := L) σ α :=
  Tensor.Operations.ofTabulated data

/-- Build the rank-1 tensor `[0, 1, ..., n - 1]`. -/
def arange {L : Tensor.Layout}
    [Tensor.Operations.OfTabulated L]
    (n : Nat) :
    Tensor (L := L) [n] (Fin n) :=
  Tensor.ofTabulated (L := L) (Vector.ofFn (fun i : Fin n => i))

end Tensor

namespace Instances

/-- `OfTabulated` instance for the `Tabulated` layout. -/
instance : Tensor.Operations.OfTabulated Tabulated where
  ofTabulated := id

end Instances

end Palladium.TensorLib
