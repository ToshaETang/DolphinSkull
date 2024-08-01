
#!/bin/bash
#PBS -N train_0731_1230
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

export file='../myproject/train_0731.py'

mpirun -hostfile $PBS_NODEFILE -n $NPROCS python3 $file > log_0731_1230


echo "Job Ended at `date`"
echo '======================================================='


#discriminator_optimizer 1e-4 -> 2e-4
