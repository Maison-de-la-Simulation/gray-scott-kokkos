FROM gitlab-registry.in2p3.fr/cta-lapp/cours/gray_scott_revolutions/grayscottbasecontainer/gray_scott_interactive_vscode:1.1.3@sha256:06a820ca0999ab4f46e28208711f25b738c523b7d19584f6081f6883c87d6fe0

USER root

RUN mkdir -p /src \
 && chown -R $MAMBA_USER:$MAMBA_USER /src

USER $MAMBA_USER

RUN git clone --branch 5.1.1 https://github.com/kokkos/kokkos.git /src/kokkos
RUN git clone https://github.com/kokkos/kokkos-tools.git /src/kokkos-tools

RUN pixi global install --environment lecture binutils=2.45 cli11=2.6 cmake=3.31 cuda-nvcc=12.8 gcc=14.3 gxx=14.3 hdf5=1.14 make=4.4 && pixi clean cache --yes
