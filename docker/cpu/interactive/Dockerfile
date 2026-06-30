FROM gitlab-registry.in2p3.fr/cta-lapp/cours/gray_scott_revolutions/grayscottbasecontainer/gray_scott_container_interactive:2.0.0@sha256:45c7d62310480242bc5ca30a0a214b83f6a76c2730db8fb26b51bafdcb4d5c8b

USER root

ADD --checksum=267ebc25fc5c8b96bb321f34d78f97b1c30f8830 https://github.com/kokkos/kokkos.git#5.1.1 /src/kokkos
ADD --checksum=3d33d0a783c5321d6981b3b4e43d3c157d2ca887 https://github.com/kokkos/kokkos-tools.git#3d33d0a783c5321d6981b3b4e43d3c157d2ca887 /src/kokkos-tools

RUN date -I > /version.txt

USER $MAMBA_USER

COPY --chown=${MAMBA_USER} exercises /home/$MAMBA_USER/Examples/exercises
COPY --chown=${MAMBA_USER} README.md /home/$MAMBA_USER/Examples/

RUN pixi global install --environment lecture binutils=2.45 cli11=2.6 cmake=3.31 gcc=14.3 gxx=14.3 hdf5=1.14 make=4.4 && pixi clean cache --yes
