# bngen from https://github.com/SHoltzen/bngen: Bayesian networks in .bif to Dice
# (`sym`), psi and simppl programs. Installed as `bngen-upstream`.
{ stdenv, ocaml-ng, fetchFromGitHub }:

let
  ocamlPackages = ocaml-ng.ocamlPackages_4_09;
in
stdenv.mkDerivation rec {
  pname = "bngen-upstream";
  version = "bee38266eaf53ab5977cf09d328bed0f9667c0d6";

  src = fetchFromGitHub {
    owner = "SHoltzen";
    repo = "bngen";
    rev = version;
    hash = "sha256-XNJek2qxF4TesTrNYXPtbck8FmLbYGlp5BJirx8WXVg=";
  };

  nativeBuildInputs = with ocamlPackages; [ ocaml findlib ocamlbuild ];
  buildInputs = with ocamlPackages; [ core sexplib ];

  # oUnit is only for tests, which the repository does not include.
  postPatch = ''
    substituteInPlace _tags --replace-fail "package(oUnit), " ""
  '';

  buildPhase = ''
    runHook preBuild
    ocamlbuild -use-ocamlfind Main.native
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 _build/src/Main.native $out/bin/bngen-upstream
    runHook postInstall
  '';
}
