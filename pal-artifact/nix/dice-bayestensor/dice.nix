{ lib, fetchFromGitHub, pkgs, callPackage, ocaml-ng, stdenv }:

let
  ocamlPackages = ocaml-ng.ocamlPackages_4_14;
  cudd = callPackage ../mlcuddidl.nix { inherit ocamlPackages; };
  rsdd = callPackage ../rsdd/rsdd.nix { };
in
ocamlPackages.buildDunePackage rec {
  pname = "dice";
  version = "git-1d8a277";

  src = fetchFromGitHub {
    owner = "SHoltzen";
    repo = "dice";
    rev = "1d8a277636a428eb46164ba4639fab12a310f30c";
    hash = "sha256-xG5yxjEqEf002vSp1GNiLtKJoYc9q9VOnAjAUnBbTKQ=";
  };

  nativeBuildInputs = with ocamlPackages; [
    menhir
  ] ++ [
    pkgs.makeWrapper
  ];

  propagatedBuildInputs = with ocamlPackages; [
    menhir
    core
    core_unix
    ounit2
    ppx_sexp_conv
    sexplib
    core_bench
    ppx_deriving
    yojson
    ctypes
    bignum
    menhirLib
    ppx_jane
  ] ++ [
    ocamlPackages."ctypes-foreign"
    cudd
  ];

  # The benchmark driver is not needed.
  postPatch = ''
    rm bench/dune
    cat > lib/dune <<EOF
(menhir
 (modules Parser)
 (flags "--table"))

(ocamllex lexer)

(rule
 (targets librsdd.a dllrsdd.so)
 (action
  (no-infer
    (progn
      (copy ${rsdd}/lib/librsdd${stdenv.hostPlatform.extensions.sharedLibrary} dllrsdd.so)
      (copy ${rsdd}/lib/librsdd.a librsdd.a)
      ))))

(library
 (libraries core core_unix.sys_unix cudd sexplib menhirLib ounit2 ctypes ctypes.foreign bignum)
 (foreign_archives rsdd)
 (library_flags (-linkall))
 (name diceLib)
 (preprocess (pps ppx_jane)))
EOF
  '';

  doCheck = false;

  postInstall = ''
    wrapProgram $out/bin/dice --run "ulimit -s ${builtins.toString (64 * 1024 * 1024 * 1024)}"
  '';

  meta = {
    platforms = lib.platforms.linux;
  };
}
