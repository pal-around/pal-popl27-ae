import TensorLib.Operations.Basics
import TensorLib.Operations.Tabulated

/-! # Layout Instance for Tabulated -/

namespace Palladium.TensorLib

namespace Instances

/-- `Tensor.Operations` instance making `Tabulated` a Layout. -/
instance : Tensor.Operations Tabulated where
  instBEq := Tabulated.instBEq
  instInhabited := Tabulated.instInhabited
  scalar := Tabulated.scalar
  fill := Tabulated.fill
  get := Tabulated.get
  broadcast := Tabulated.broadcast
  flatten := Tabulated.flatten
  unflatten := Tabulated.unflatten
  squeeze := Tabulated.squeeze
  transpose := Tabulated.transpose
  map := Tabulated.map
  mapSlicesM := Tabulated.mapSlicesM
  zipWithM := Tabulated.zipWithM
  zipWith3M := Tabulated.zipWith3M
  flatten_length := Tabulated.flatten_length

end Instances

end Palladium.TensorLib
