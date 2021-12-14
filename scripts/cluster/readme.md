## GPU Cluster

This directory contains some scripts to support training execution on a SLURM GPU cluster.

### Setup

One time setup at the GPU cluster:

1. Clone the project and switch into directory: `git clone git@github.com:DFKI-NLP/sherlock.git && cd sherlock`
2. Change permission for setup scripts:
   1. `chmod +x scripts/cluster/batch.sh`
   2. `chmod +x scripts/cluster/wrapper.sh`
   3. `chmod +x scripts/cluster/binary_relation_clf.sh`
3. (Optional) Verify setup by successfully executing a command, e.g.:

```bash
srun -K --ntasks=1 --gpus-per-task=0 --cpus-per-task=1 -p batch \
--container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.07-py3.sqsh \
--container-workdir="`pwd`" ./scripts/cluster/wrapper.sh pip list
```

Run:
```bash
./scripts/cluster/batch.sh -p <PROFILE> --gpus=1 ./scripts/cluster/wrapper.sh ./scripts/cluster/binary_relation_clf.sh
```


