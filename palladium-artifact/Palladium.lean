-- Root of the project
-- See `Results.lean` for the main theorems and their axiom footprint.

import Util.Context
import Util.Denote
import Util.Discrete
import Util.Macros
import Util.PMF
import Util.Float

import TensorLib.DType
import TensorLib.Shape
import TensorLib.Permutation
import TensorLib.Tensor
import TensorLib.Operations.Basics
import TensorLib.Operations.Tabulated
import TensorLib.Operations.Instances
import TensorLib.Operations.LawfulOperations
import TensorLib.Operations.Lemmas
import TensorLib.Operations.LawfulInstances
import TensorLib.Operations.TabulatedInterop
import TensorLib.IndexMaps
import TensorLib.IndexLemmas

import Palladium.PAL.Operators
import Palladium.PAL.Syntax
import Palladium.PAL.Denote
import Palladium.PAL.Semantics

import JAX.Operators
import JAX.Syntax
import JAX.Denote
import JAX.Semantics
import JAX.Adequacy

import Palladium.Compiler.Compile
import Palladium.Compiler.Encoding
import Palladium.Compiler.Correctness
