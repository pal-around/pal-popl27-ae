FROM nixos/nix

RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

COPY . /pal-artifact
WORKDIR /pal-artifact

RUN nix build .#bngen --cores 4 --print-build-logs
RUN nix build .#jax --cores 4 --print-build-logs
RUN nix build .#dice --print-build-logs
RUN nix build .#memo --print-build-logs
RUN nix build .#python --print-build-logs
RUN nix build .#alea-jl --print-build-logs
RUN nix build .#tex --print-build-logs
RUN nix build .#artifact-env --print-build-logs

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en

CMD ["nix", "develop", ".#artifact"]
