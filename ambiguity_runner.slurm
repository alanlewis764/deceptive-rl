#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Fri Nov 19 2021 12:59:09 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=physical

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="Ambiguity runner"

# The project ID which this job should run under:
#SBATCH --account="punim1607"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12800MB



# Use this email address:
#SBATCH --mail-user=lewisa1@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-36:0:00
#SBATCH --array=2,3,4,5,6,7,8

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load pytorch/1.4.0-python-3.7.4

# The job command(s):
python3 ambiguity_runner.py --hyperparam "pruning_decay" --map_num ${SLURM_ARRAY_TASK_ID}
