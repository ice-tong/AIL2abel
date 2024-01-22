cd /tmp && \
git clone https://github.com/angr/angr.git -b feat/decompiler_varname_from_symbols && \
cd angr && \
sed -i "s/.dev0//g" setup.cfg && \
sed -i "s/.dev0//g" pyproject.toml && \
sed -i "s/.post1//g" pyproject.toml && \
pip install . && \
cd /tmp && \
git clone https://github.com/angr/cle.git -b feat/decompiler_varname_from_symbols && \
cd cle && \
sed -i "s/.dev0//g" setup.cfg && \
pip install . && \
pip install --no-cache-dir func_timeout paramiko tqdm tokenizers && \
cd /tmp && \
rm -rf /tmp/*