
#!/bin/bash
#PBS -N train_0722_0125
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

module load python-3.12.3
source venv/bin/activate

export file='../myproject/train_0720.py'

mpirun -hostfile $PBS_NODEFILE -n $NPROCS python3 $file > log_0722_0125


echo "Job Ended at `date`"
echo '======================================================='


