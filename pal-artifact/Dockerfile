FROM nixos/nix

RUN echo "experimental-features = nix-command flakes" >> /etc/nix/nix.conf

COPY . /pal-dev
WORKDIR /pal-dev

RUN nix build .#jax
RUN nix build .#memo
RUN nix build .#python
RUN nix build .#dice
RUN nix build .#bngen
RUN nix build .#alea-jl
RUN nix build .#tex
RUN nix build .#artifact-env

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en

CMD ["nix", "develop", ".#artifact"]
