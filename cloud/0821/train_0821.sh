
#!/bin/bash
#PBS -N train_0822_0940
#PBS -o master:$PBS_O_WORKDIR/$PBS_JOBID.out
#PBS -e master:$PBS_O_WORKDIR/$PBS_JOBID.err
#PBS -q q24cores
#PBS -l nodes=1:ppn=1
cd $PBS_O_WORKDIR
ulimit -a
echo '======================================================='
echo Working directory is $PBS_O_WORKDIR
echo "Starting on `hostname` at `date`"
if [ -n "$PBS_NODEFILE" ]; then
 if [ -f $PBS_NODEFILE ]; then
 echo "Nodes used for this job:"
 cat ${PBS_NODEFILE}
 NPROCS=`wc -l < $PBS_NODEFILE`
 fi
fi

source /lustre/lwork/ttang001/myproject/venv/bin/activate

export file='/lustre/lwork/ttang001/myproject/train_0821.py'

mpirun -hostfile $PBS_NODEFILE -n $NPROCS python3 $file > log_0822_0940


echo "Job Ended at `date`"
echo '======================================================='

