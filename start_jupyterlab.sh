#!/bin/sh
#SBATCH -J stabilize2p-jupyterlab
#SBATCH -p p5
# #SBATCH --nodelist=node102
# #SBATCH -N 1
# #SBATCH -c 16
# #SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output="slurm/jupyterlab-%j.log"
#SBATCH --chdir=""

# you may want to change `SBATCH -p` to the partition where you
# want to execute JupyterLab

# running as a slurm job
if [ -n "$SLURM_JOB_NAME" ]; then
    echo "DATE: `date`"
    echo "starting job '$SLURM_JOB_NAME' in '$(pwd)'.."

    [ $(squeue --name=stabilize2p-jupyterlab | wc -l) -gt 2 ] && {
        echo "jupyterlab already running. You can stop it by running: scancel $(squeue --name=stabilize2p-jupyterlab -o %A | sort -n | head -n2 | tail -n1)"
        exit 1
    }

    echo 'Please create an ssh tunnel to the following IP from your local machine:'
    grep "$(squeue --name=stabilize2p-jupyterlab -o %N | tail -n1)" /etc/hosts

    echo "Starting Jupyter-Lab.." \
        && jupyter-lab --ContentsManager.allow_hidden=True --no-browser --ip=0.0.0.0 --port=8080
    
    echo "FINISH DATE: `date`"
    exit 0
fi

# running as a tmux session
if tmux ls 2>/dev/null | grep 'stabilize2p-jupyterlab' >/dev/null; then
    echo "JupyterLab already started. You can stop it by running: tmux kill-session -t 'stabilize2p-jupyterlab'"
else
    commands='
echo "INFO: PRESS ctrl+b d TO EXIT THE TMUX SESSION" && 
export LD_LIBRARY_PATH=/home/adrian/miniconda3/lib &&
echo "Starting Jupyter-Lab.." &&
jupyter-lab --ContentsManager.allow_hidden=True --no-browser --port=8080
'
    
    tmux new-session -s stabilize2p-jupyterlab "$commands"
    
    echo "tmux sessions:"
    tmux ls
fi
