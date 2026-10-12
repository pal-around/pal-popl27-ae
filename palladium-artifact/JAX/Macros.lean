import JAX.Syntax
import TensorLib.Operations.TabulatedInterop

/-! # JAX Macros -/

namespace Palladium.JAX

declare_syntax_cat jaxExpr
declare_syntax_cat jaxArrayData
declare_syntax_cat jaxShape
declare_syntax_cat jaxAxis
declare_syntax_cat jaxSumAxes
declare_syntax_cat jaxPerm
declare_syntax_cat jaxArrayTy

syntax num : jaxAxis
syntax "~(" term ")" : jaxAxis

syntax num : jaxSumAxes
syntax "[" num,* "]" : jaxSumAxes

syntax "(" ")" : jaxShape
syntax "(" num "," num,* ")" : jaxShape
syntax "~(" term ")" : jaxShape

syntax "perm" "(" num,* ")" : jaxPerm
syntax "~(" term ")" : jaxPerm

syntax num : jaxArrayData
syntax scientific : jaxArrayData
syntax "~(" term ")" : jaxArrayData
syntax "[" jaxArrayData,* "]" : jaxArrayData

syntax "float[" num,* "]" : jaxArrayTy
syntax "bool[" num,* "]" : jaxArrayTy
syntax "nat[" num,* "]" : jaxArrayTy
syntax "disc(" num ")" "[" num,* "]" : jaxArrayTy

syntax num : jaxExpr
syntax scientific : jaxExpr
syntax ident : jaxExpr
syntax "-" jaxExpr : jaxExpr
syntax "logical_not" "(" jaxExpr ")" : jaxExpr
syntax "(" jaxExpr ")" : jaxExpr
syntax "~(" term ")" : jaxExpr
syntax "array" "(" jaxArrayData ")" : jaxExpr
syntax "array" "(" "~(" term ")" ")" : jaxExpr
syntax "arange" "(" num ")" : jaxExpr
syntax "arange" "(" "~(" term ")" ")" : jaxExpr
syntax "ones" "(" jaxShape ")" : jaxExpr
syntax "ones_like" "(" jaxExpr ")" : jaxExpr
syntax "broadcast_to" "(" jaxExpr "," jaxShape ")" : jaxExpr
syntax "sum" "(" jaxExpr "," "axis" "=" jaxSumAxes ")" : jaxExpr
syntax "select" "(" jaxExpr "," jaxExpr "," jaxExpr ")" : jaxExpr
syntax "reshape" "(" jaxExpr "," jaxShape ")" : jaxExpr
syntax "squeeze" "(" jaxExpr "," "axis" "=" jaxAxis ")" : jaxExpr
syntax "transpose" "(" jaxExpr "," jaxPerm ")" : jaxExpr
syntax "let" ident "=" jaxExpr "in" jaxExpr : jaxExpr
syntax "let" ident ":" jaxArrayTy "=" jaxExpr "in" jaxExpr : jaxExpr
syntax "comment" str jaxExpr : jaxExpr
syntax "scan" ident "<-" jaxExpr "for" num "in" jaxExpr : jaxExpr
syntax:60 jaxExpr " + " jaxExpr : jaxExpr
syntax:60 jaxExpr " - " jaxExpr : jaxExpr
syntax:70 jaxExpr " * " jaxExpr : jaxExpr
syntax:70 jaxExpr " / " jaxExpr : jaxExpr
syntax:60 jaxExpr " == " jaxExpr : jaxExpr
syntax:60 jaxExpr " != " jaxExpr : jaxExpr
syntax:60 jaxExpr " < " jaxExpr : jaxExpr
syntax:60 jaxExpr " <= " jaxExpr : jaxExpr
syntax:60 jaxExpr " > " jaxExpr : jaxExpr
syntax:60 jaxExpr " >= " jaxExpr : jaxExpr
syntax "jax{" jaxExpr "}" : term

open Lean
open Palladium.Util

abbrev JaxTerm := Lean.TSyntax `term

private def mkList (xs : Array JaxTerm) : Lean.MacroM JaxTerm := do
  let mut acc ← `(List.nil)
  for x in xs.reverse do
    acc ← `(List.cons $x $acc)
  return acc

/-- Build a Vector literal from an array of terms. -/
private def mkVector (xs : Array JaxTerm) : Lean.MacroM JaxTerm := do
  let lst ← mkList xs
  `(($lst).toArray.toVector)

private def mkNatList (xs : List Nat) : Lean.MacroM JaxTerm := do
  let mut acc ← `(List.nil)
  for x in xs.reverse do
    let term : Lean.TSyntax `term := Lean.quote x
    acc ← `(List.cons $term $acc)
  return acc

private def mkTensor (dimsList : List Nat) (data : JaxTerm) : Lean.MacroM JaxTerm := do
  let dimsTerm ← mkNatList dimsList
  `(TensorLib.Tensor.ofTabulated (σ := $dimsTerm) $data)

/-- Proof fallback for shape-indexed constructors:
    use `rfl` when definitional reduction is enough,
    `native_decide` for decidable propositions like `BroadcastCompatible`. -/
private def mkShapeProof : Lean.MacroM JaxTerm :=
  `(by first | exact rfl | native_decide)

/-- Expand a `jaxAxis` syntax node to a `Nat` term (squeeze / single-axis uses). -/
private def axisNatTerm (ax : Lean.Syntax) : Lean.MacroM JaxTerm := do
  match ax with
  | `(jaxAxis| $n:num) => pure (Lean.quote n.getNat)
  | `(jaxAxis| ~( $t:term )) => pure t
  | _ => Lean.Macro.throwErrorAt ax "axis expects a Nat term (use n or ~(t))"

/-- Build `List (Fin σ.rank)` for `sum` from `jaxSumAxes` (`axis = n` or `axis = [n, ...]`). -/
private def mkSumAxesList (ax : Lean.Syntax) : Lean.MacroM JaxTerm := do
  let pf ← `(by native_decide)
  match ax with
  | `(jaxSumAxes| $n:num) =>
      `(List.cons (⟨$(Lean.quote n.getNat), $pf⟩) List.nil)
  | `(jaxSumAxes| [ $ns:num,* ]) => do
      let elems := ns.getElems.map (fun x => (x : Lean.TSyntax `num))
      let mut acc ← `(List.nil)
      for n in elems.reverse do
        acc ← `(List.cons (⟨$(Lean.quote n.getNat), $pf⟩) $acc)
      return acc
  | _ =>
      Lean.Macro.throwErrorAt ax "sum axis expects n or [n, ...] (axis indices)"

private def expandShape (stx : Lean.Syntax) : Lean.MacroM JaxTerm := do
  match stx with
  | `(jaxShape| ( )) => `(List.nil)
  | `(jaxShape| ( $x:num , $xs:num,* )) => do
      let items := #[x] ++ xs.getElems
      let mut dimsRev : List (Lean.TSyntax `term) := []
      for x in items do
        dimsRev := (Lean.quote x.getNat) :: dimsRev
      let mut acc ← `(List.nil)
      for d in dimsRev do
        acc ← `(List.cons $d $acc)
      pure acc
  | `(jaxShape| ~( $t:term )) => pure t
  | _ =>
      Lean.Macro.throwErrorAt stx
        "shape expects nat literals (shape(...)) or ~(shapeTerm)"

private def expandPerm (stx : Lean.Syntax) : Lean.MacroM JaxTerm := do
  match stx with
  | `(jaxPerm| perm ( $xs:num,* )) => do
      let items := xs.getElems.map (fun x => (x : Lean.TSyntax `num))
      let mut acc ← `(List.nil)
      for x in items.reverse do
        let natTerm : Lean.TSyntax `term := Lean.quote x.getNat
        acc ← `(List.cons $natTerm $acc)
      pure acc
  | `(jaxPerm| ~( $t:term )) => pure t
  | _ =>
      Lean.Macro.throwErrorAt stx
        "perm expects nat literals (perm(...)) or ~(permTerm)"

private partial def expandArrayData : Lean.Syntax -> Lean.MacroM (JaxTerm × List Nat)
  | `(jaxArrayData| $n:num) => do
      let term ← `(Float.ofNat $n)
      pure (term, [])
  | `(jaxArrayData| $s:scientific) => do
      let term ← `(($s : Float))
      pure (term, [])
  | `(jaxArrayData| ~( $t:term )) => do
      pure (t, [])
  | `(jaxArrayData| [ $xs:jaxArrayData,* ]) => do
      let items := xs.getElems.map (fun x => (x : Lean.TSyntax `jaxArrayData))
      let mut terms : Array JaxTerm := #[]
      let mut shapeAcc : List Nat := []
      let mut isFirst := true
      for x in items do
        let (term, elemShape) ← expandArrayData x
        if isFirst then
          shapeAcc := elemShape
          isFirst := false
        else if elemShape != shapeAcc then
          Lean.Macro.throwErrorAt x "array literal has inconsistent element shapes"
        terms := terms.push term
      let vec ← mkVector terms
      pure (vec, items.size :: shapeAcc)
  | e => Lean.Macro.throwErrorAt e "unsupported jax array literal syntax"

private def expandArrayTy : Lean.Syntax -> Lean.MacroM (JaxTerm × JaxTerm)
  | `(jaxArrayTy| float[ $dims,* ]) => do
      let dimTerms : Array JaxTerm := dims.getElems.map (fun d => ⟨d.raw⟩)
      let dimList ← mkList dimTerms
      pure (dimList, ← `(Palladium.TensorLib.DType.float))
  | `(jaxArrayTy| bool[ $dims,* ]) => do
      let dimTerms : Array JaxTerm := dims.getElems.map (fun d => ⟨d.raw⟩)
      let dimList ← mkList dimTerms
      pure (dimList, ← `(Palladium.TensorLib.DType.bool))
  | `(jaxArrayTy| nat[ $dims,* ]) => do
      let dimTerms : Array JaxTerm := dims.getElems.map (fun d => ⟨d.raw⟩)
      let dimList ← mkList dimTerms
      pure (dimList, ← `(Palladium.TensorLib.DType.nat))
  | `(jaxArrayTy| disc( $n )[ $dims,* ]) => do
      let dimTerms : Array JaxTerm := dims.getElems.map (fun d => ⟨d.raw⟩)
      let dimList ← mkList dimTerms
      pure (dimList, ← `(Palladium.TensorLib.DType.disc (Discrete.fin $n)))
  | t => Lean.Macro.throwErrorAt t "unsupported array type (expected float[dims], nat[dims], bool[dims], or disc(n)[dims])"

private partial def expandExpr : Lean.Syntax -> Lean.MacroM JaxTerm
  | `(jaxExpr| $n:num) => do
      let data ← `(Float.ofNat $n)
      let t ← mkTensor [] data
      `(Expr.array Palladium.TensorLib.DType.float $t)
  | `(jaxExpr| $s:scientific) => do
      let data ← `(($s : Float))
      let t ← mkTensor [] data
      `(Expr.array Palladium.TensorLib.DType.float $t)
  | `(jaxExpr| $x:ident) =>
      let name : Lean.TSyntax `term := ⟨Lean.Syntax.mkStrLit x.getId.toString⟩
      `(Expr.ref _ $name)
  | `(jaxExpr| - $e:jaxExpr) => do
      let e' ← expandExpr e
      `(Expr.neg $e')
  | `(jaxExpr| logical_not ( $e:jaxExpr )) => do
      let e' ← expandExpr e
      `(Expr.uop .not $e' .not)
  | `(jaxExpr| ( $e:jaxExpr )) => expandExpr e
  | `(jaxExpr| ~( $t:term )) => `($t)
  | `(jaxExpr| array ( $x:jaxArrayData )) => do
      let (data, shapeAcc) ← expandArrayData x
      let t ← mkTensor shapeAcc data
      `(Expr.array Palladium.TensorLib.DType.float $t)
  | `(jaxExpr| array ( ~( $t:term ) )) =>
      `(Expr.array _ $t)
  | `(jaxExpr| arange ( $n:num )) => do
      let nTerm : Lean.TSyntax `term := Lean.quote n.getNat
      `(Expr.array (Palladium.TensorLib.DType.disc (Palladium.Util.Discrete.fin $nTerm))
        (TensorLib.Tensor.arange $nTerm))
  | `(jaxExpr| arange ( ~( $t:term ) )) =>
      `(Expr.array (Palladium.TensorLib.DType.disc (Palladium.Util.Discrete.fin $t))
        (TensorLib.Tensor.arange $t))
  | `(jaxExpr| ones ( $x:jaxShape )) => do
      let shapeTerm ← expandShape x
      `(Expr.ones $shapeTerm)
  | `(jaxExpr| ones_like ( $x:jaxExpr )) => do
      let term ← expandExpr x
      `(Expr.onesLike $term)
  | `(jaxExpr| broadcast_to ( $arg:jaxExpr, $shp:jaxShape )) => do
      let arg' ← expandExpr arg
      let shapeTerm ← expandShape shp
      let pf ← mkShapeProof
      `(Expr.broadcastTo $arg' $shapeTerm $pf)
  | `(jaxExpr| sum ( $arg:jaxExpr, axis = $ax:jaxSumAxes )) => do
      let arg' ← expandExpr arg
      let axesList ← mkSumAxesList ax
      `(Expr.sum $arg' $axesList (by native_decide))
  | `(jaxExpr| select ( $cond:jaxExpr, $x:jaxExpr, $y:jaxExpr )) => do
      let cond' ← expandExpr cond
      let x' ← expandExpr x
      let y' ← expandExpr y
      let pf ← mkShapeProof
      `(Expr.mkSelect $cond' $x' $y' $pf)
  | `(jaxExpr| reshape ( $arr:jaxExpr, $shp:jaxShape )) => do
      let arr' ← expandExpr arr
      let shapeTerm ← expandShape shp
      let pf ← mkShapeProof
      `(Expr.reshape $arr' $shapeTerm $pf)
  | `(jaxExpr| squeeze ( $arg:jaxExpr, axis = $ax:jaxAxis )) => do
      let arg' ← expandExpr arg
      let axis' ← axisNatTerm ax
      let pf ← mkShapeProof
      `(Expr.squeeze $arg' ⟨$axis', $pf⟩ $pf)
  | `(jaxExpr| transpose ( $arg:jaxExpr, $p:jaxPerm )) => do
      let arg' ← expandExpr arg
      let permTerm ← expandPerm p
      let pf ← mkShapeProof
      `(Expr.transpose $arg' $permTerm $pf)
  | `(jaxExpr| let $x:ident = $val:jaxExpr in $body:jaxExpr) => do
      let val' ← expandExpr val
      let body' ← expandExpr body
      let name : Lean.TSyntax `term := ⟨Lean.Syntax.mkStrLit x.getId.toString⟩
      `(Expr.letIn $name $val' $body')
  | `(jaxExpr| let $x:ident : $ty:jaxArrayTy = $val:jaxExpr in $body:jaxExpr) => do
      let val' ← expandExpr val
      let body' ← expandExpr body
      let name : Lean.TSyntax `term := ⟨Lean.Syntax.mkStrLit x.getId.toString⟩
      let (shapeTerm, dtypeTerm) ← expandArrayTy ty
      `(Expr.letIn (T₁ := Ty.tensor $shapeTerm $dtypeTerm) $name $val' $body')
  | `(jaxExpr| comment $s:str $body:jaxExpr) => do
      let body' ← expandExpr body
      `(Expr.comment $s $body')
  | `(jaxExpr| scan $x:ident <- $init:jaxExpr for $n:num in $body:jaxExpr) => do
      let init' ← expandExpr init
      let body' ← expandExpr body
      let name : Lean.TSyntax `term := ⟨Lean.Syntax.mkStrLit x.getId.toString⟩
      `(Expr.scan $name $init' $n $body')
  | `(jaxExpr| $a:jaxExpr * $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.mul $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr / $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.div $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr + $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.add $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr - $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.sub $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr == $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.eq $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr != $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.ne $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr < $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.lt $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr <= $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.le $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr > $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.gt $a' $b' $pf)
  | `(jaxExpr| $a:jaxExpr >= $b:jaxExpr) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      let pf ← mkShapeProof
      `(Expr.ge $a' $b' $pf)
  | e => Lean.Macro.throwErrorAt e "unsupported jax expr syntax"

macro "jax{" e:jaxExpr "}" : term => do
  expandExpr e

declare_syntax_cat jaxStmt
syntax ident ":=" jaxExpr : jaxStmt
syntax "~(" term ")" ":=" jaxExpr : jaxStmt
syntax "comment" str : jaxStmt
syntax "comment" term : jaxStmt
syntax "delete" ident : jaxStmt
syntax "delete" term : jaxStmt
syntax "jax{" jaxStmt "}" : term

private def expandStmt : Lean.Syntax -> Lean.MacroM JaxTerm
  | `(jaxStmt| $x:ident := $e:jaxExpr) => do
      let ex ← expandExpr e
      let name : Lean.TSyntax `term := ⟨Lean.Syntax.mkStrLit x.getId.toString⟩
      `(Stmt.assign $name $ex)
  | `(jaxStmt| ~( $x:term ) := $e:jaxExpr) => do
      let ex ← expandExpr e
      `(Stmt.assign $x $ex)
  | `(jaxStmt| comment $s:str) =>
      `(Stmt.comment $s)
  | `(jaxStmt| comment $s:term) =>
      `(Stmt.comment $s)
  | `(jaxStmt| delete $x:ident) =>
      let name : Lean.TSyntax `term := ⟨Lean.Syntax.mkStrLit x.getId.toString⟩
      `(Stmt.delete $name)
  | `(jaxStmt| delete $x:term) =>
      `(Stmt.delete $x)
  | s => Lean.Macro.throwErrorAt s "unsupported jax stmt syntax"

macro "jax{" s:jaxStmt "}" : term => do
  expandStmt s

declare_syntax_cat jaxStmts
syntax jaxStmt ";" jaxStmts : jaxStmts
syntax jaxStmt ";" : jaxStmts
syntax "jax{" jaxStmts "}" : term

private partial def expandStmts : Lean.Syntax -> Lean.MacroM JaxTerm
  | `(jaxStmts| $st:jaxStmt; $ss:jaxStmts) => do
      let st' ← expandStmt st
      let ss' ← expandStmts ss
      `(Stmts.cons $st' $ss')
  | `(jaxStmts| $st:jaxStmt ;) => do
      let st' ← expandStmt st
      `(Stmts.cons $st' Stmts.nil)
  | s => Lean.Macro.throwErrorAt s "unsupported jax stmts syntax"

macro "jax{" s:jaxStmts "}" : term => do
  expandStmts s

declare_syntax_cat jaxProg
declare_syntax_cat jaxParam

syntax ident ":" jaxArrayTy : jaxParam

syntax ident "(" jaxParam,* ")" ":" jaxStmts "return" ident : jaxProg

private def expandParam : Lean.Syntax -> Lean.MacroM JaxTerm
  | `(jaxParam| $name:ident : $T:jaxArrayTy) => do
      let (shapeTerm, dtypeTerm) ← expandArrayTy T
      let nameStr : Lean.TSyntax `term := Lean.quote name.getId.toString
      `(⟨$nameStr, Ty.tensor $shapeTerm $dtypeTerm⟩)
  | p => Lean.Macro.throwErrorAt p "unsupported jax param syntax (expected name : float[dims], name : nat[dims], or name : bool[dims])"

private def expandProg : Lean.Syntax -> Lean.MacroM JaxTerm
  | `(jaxProg| $f:ident ( $params:jaxParam,* ) : $body:jaxStmts return $ret:ident) => do
      let paramStx := params.getElems
      let mut paramPairs : Array JaxTerm := #[]
      for p in paramStx do
        paramPairs := paramPairs.push (← expandParam p)
      let jctxTerm ← if paramPairs.isEmpty then
        `(([] : Util.Context JAX.Ty))
      else
        mkList paramPairs
      let retName : Lean.TSyntax `term := Lean.quote ret.getId.toString
      let bodyTerm ← expandStmts body
      `(Program.mk (argTy := $jctxTerm)
        $(Lean.quote f.getId.toString)
        $bodyTerm
        $retName
        (by native_decide))
  | d => Lean.Macro.throwErrorAt d "unsupported jax decl syntax"

macro "jax{" d:jaxProg "}" : term => do
  expandProg d

end Palladium.JAX
