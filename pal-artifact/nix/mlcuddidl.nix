{ lib, gnum4, stdenv, fetchurl, ocamlPackages }:

# The OCaml interface to the CUDD BDD library, built as opam builds it.
stdenv.mkDerivation rec {
  pname = "mlcuddidl";
  version = "3.0.8";

  src = fetchurl {
    url = "https://framagit.org/nberth/mlcuddidl/-/archive/${version}/mlcuddidl-${version}.tar.gz";
    hash = "sha256-JrIrHx6tWhnqdCNIK1L1tcSYo9G54TzgVmIKd6VJir0=";
  };

  createFindlibDestdir = true;

  # opam's exn_init_val_unit.diff: the generated stubs initialize an OCaml value.
  postPatch = ''
    echo 's/(\*_c2)\.exn = NULL;/(*_c2).exn = Val_unit;/g;' >> sedscript_c
  '';

  configurePhase = ''
    runHook preConfigure
    ./configure --prefix "$out"
    runHook postConfigure
  '';

  nativeBuildInputs = [
    ocamlPackages.ocaml
    ocamlPackages.camlidl
    ocamlPackages.findlib
    ocamlPackages.ocamlbuild
    gnum4
  ];
}
