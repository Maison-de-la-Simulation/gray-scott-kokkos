FROM gitlab-registry.in2p3.fr/cta-lapp/cours/gray_scott_revolutions/grayscottbasecontainer/gray_scott_container_interactive:1.1.2@sha256:50e8f926ded069786a7bea91b937436edf129535a2b4ccfc54ae4ac5bb6e26bb

USER root

RUN mkdir -p /src \
 && chown -R $MAMBA_USER:$MAMBA_USER /src

USER $MAMBA_USER

RUN git clone --branch 5.1.1 https://github.com/kokkos/kokkos.git /src/kokkos
RUN git clone https://github.com/kokkos/kokkos-tools.git /src/kokkos-tools

RUN pixi global install --environment lecture binutils=2.45 cli11=2.6 cmake=3.31 gcc=14.3 gxx=14.3 hdf5=1.14 make=4.4 && pixi clean cache --yes
