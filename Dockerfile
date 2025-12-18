FROM jupyter/base-notebook:latest

# Install Deno
RUN curl -fsSL https://deno.land/install.sh | sh && \
    ln -s /root/.deno/bin/deno /usr/local/bin/deno

# Install Deno Jupyter kernel
RUN deno jupyter --install --unstable

# Upgrade JupyterLab
RUN pip install --no-cache-dir "jupyterlab>=4.0"

WORKDIR /home/jovyan

RUN mkdir -p /home/jovyan/notebooks
COPY wiki3-kg-extraction.ipynb /home/jovyan/notebooks/

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token="]
