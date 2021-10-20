# Derek's DiffBragg MLX5 Reproducer

This is a Python + mpi4py reproducer for the mlx5 error reported
in Case 293224.

The MLX5 error occurs (pretty consistently) on Perlmutter's GPU nodes. What I
have found the bug is reliably tripped when:
1. Using 70 (or more) GPU nodes
2. All sources and dependencies (including the Conda environtment) are on $CFS,
   which is our large GPFS filesystem

Substituting the DVS mount (`/dvs_ro/`) stops this reproducer from tripping the
error (possibly completely, or at least supressing the probability). Installing
to $SCRATCH (our Lustre filesystem) also seems to prevent the error.

We note that this app also sometimes hangs. This may be a separate issue.

## Installation

```bash
ssh perlmutter
#create dir at $CFS/nvendor/<your-username>
cd $CFS/nvendor/<your-username>
git clone https://github.com/JBlaschke/diffBragg-Reproducer
cd diffBragg-Reproducer
./setup.sh
```

This will create a conda environment called `diffbragg` 
at `$(pwd)/diffbragg`.

If you'd like to test on different filesystems, you can
istall this repo in another place and follow the same
instructions.

## Running

```bash
salloc -C gpu -n 308 -c 2 -t 60 --gpus-per-task 1 -A nvendor_g -q early_science
source env.sh
srun -u -N77 -n308 -c2 python -u -X faulthandler pmUsrIssue5.py
```

## Sample errror

```
mlx5: nid003244: got completion with error:
00000000 00000000 00000000 00000000
00000000 00000000 00000000 00000000
00000000 20009232 00000000 00000300
00003c40 92083204 000180b8 0085a0e0
MPICH ERROR [Rank 256] [job id 126699.1] [Wed Oct 20 12:32:36 2021] [nid003244] - Abort(70891919) (rank 256 in comm 0): Fatal error in PMPI_Gatherv: Other MPI error, error stack:
PMPI_Gatherv(415)..............: MPI_Gatherv failed(sbuf=0x55ee4a4ebea0, scount=88, MPI_BYTE, rbuf=(nil), rcnts=(nil), displs=(nil), datatype=MPI_BYTE, root=0, comm=MPI_COMM_WORLD) failed
MPIR_CRAY_Gatherv(353).........: 
MPIC_Recv(197).................: 
MPIC_Wait(71)..................: 
MPIR_Wait_impl(41).............: 
MPID_Progress_wait(186)........: 
MPIDI_Progress_test(80)........: 
MPIDI_OFI_handle_cq_error(1059): OFI poll failed (ofi_events.c:1061:MPIDI_OFI_handle_cq_error:Input/output error - local protection error)
```

You can view the full output in `example-error.out`

We note that this app also sometimes hangs for reasons we don't understand. It may
be related to a different problem. 
