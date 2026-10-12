# [POPL'27 Artifact] Tensorizing Probabilistic Programs for Accelerated Exact Inference

- [`palladium-artifact/`](palladium-artifact/README.md): Lean 4 formalization and proofs.
- [`pal-artifact/`](pal-artifact/popl-ae/README.md): PAL implementation and the paper's experiments, with Nix setup instructions.

A Dockerfile is also included in `pal-artifact` for users unfamiliar with Nix:

- Install Docker by following the [official instructions](https://docs.docker.com/get-docker/).

- Install and configure the [NVIDIA Container Toolkit for Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker).
  NVIDIA driver (≥ 525) and CUDA (≥ 12.2) are required. 

- Choose your GPU:

  The Nix environment bundles CUDA 12.2 by default.

  | Architecture | Compute capability | Example GPUs |
  |---|---|---|
  | Pascal | 6.0 / 6.1 | P100 / GTX 1080, GTX 1080 Ti |
  | Volta | 7.0 | V100, Titan V |
  | Turing | 7.5 | RTX 2080, RTX 2080 Ti, T4 |
  | Ampere | 8.0 / 8.6 | A100 / RTX 3090, RTX 3080 |
  | Ada | 8.9 | RTX 4090, L40, L4 |
  | Hopper | 9.0 | H100 |
  | Blackwell | 10.0 / 12.0 | B200 / RTX 5090 |

  Look up your card's compute capability at [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus).
  For example, if your card is an RTX 3090 (`8.6`), change `cudaCapabilities` in [`flake.nix`](pal-artifact/flake.nix) accordingly:

  ```diff
  - cudaCapabilities = [ "7.5" ];
  + cudaCapabilities = [ "8.6" ];
  ```

- Build the image from the `pal-artifact/` directory:

  ```shell
  > cd pal-artifact
  > docker build -t pal-artifact .
  ```

  Condensed timings from an earlier build, before reordering:

  ```text
  [+] Building 1828.4s (17/17) FINISHED                     docker:default 
  => [internal] load build definition from Dockerfile                0.0s 
  => [internal] load metadata for docker.io/nixos/nix:latest         0.6s 
  => [internal] load .dockerignore                                   0.0s 
  => [ 1/12] FROM docker.io/nixos/nix:latest                        19.8s 
  => [ 2/12] RUN echo "experimental-features = nix-command flakes"   2.6s
  => [ 3/12] COPY . /pal-artifact                                    0.0s
  => [ 4/12] WORKDIR /pal-artifact                                   0.0s
  => [ 5/12] RUN nix build .#bngen --cores 4 --print-build-logs    244.6s
  => [ 6/12] RUN nix build .#jax --cores 4 --print-build-logs      722.5s
  => [ 7/12] RUN nix build .#dice --print-build-logs               152.0s
  => [ 8/12] RUN nix build .#memo --print-build-logs                 7.1s
  => [ 9/12] RUN nix build .#python --print-build-logs              80.2s
  => [10/12] RUN nix build .#alea-jl --print-build-logs             12.5s
  => [11/12] RUN nix build .#tex --print-build-logs                 16.4s
  => [12/12] RUN nix build .#artifact-env --print-build-logs         6.0s
  => exporting to image                                            563.6s
  => => exporting layers                                           414.8s
  => => unpacking to docker.io/library/pal-artifact:latest         148.7s
  ```

  This build took about **30 min**.


- (Optional) Save the clean image before running the experiments:

  ```shell
  > docker save -o pal-artifact.tar pal-artifact
  ```

- From the same `pal-artifact/` directory, run the image and save results in `../pal-artifact-results/`:

  ```shell
  > docker run -it --name pal-artifact-run --gpus all \
      -v "$PWD/../pal-artifact-results:/pal-artifact/popl-ae/results" pal-artifact
  ```

  `--gpus all` exposes the host GPUs to the container.

- Inside the container, verify that JAX can use the GPU:

  ```shell
  > python -c "import jax; print(jax.devices('gpu'))"
  ```

  This must list CUDA devices (e.g. `[CudaDevice(id=0)]`).

- Inside the container, follow [`pal-artifact/popl-ae/README.md`](pal-artifact/popl-ae/README.md).
  In **Installation**, skip steps 1 and start at step 2.
  Use `make smoke GPU=0` or `make full GPU=0` for GPU experiments; replace `0` with your GPU's index.

- (Optional) After exiting, save the container's initialized environment:

  ```shell
  > docker commit pal-artifact-run pal-artifact-tested
  > docker save -o pal-artifact-tested.tar pal-artifact-tested
  ```

  Mounted results remain on the host and are not included in the saved image.
