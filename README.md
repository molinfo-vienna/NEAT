Let's get started!

Create an evironment:

```bash
mamba create -n molgen python==3.11.13
mamba activate molgen
```

Install the GPU version of pytorch

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```
Install the rest

```bash
pip install -e .
```