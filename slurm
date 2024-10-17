#!/bin/sh
#SBATCH --job-name=vss-pid-ddpg
#SBATCH --mem 32G
#SBATCH -c 20
#SBATCH -o vss-pid-ddpg.log
#SBATCH --output=/home/CIN/mcaa/robocin/vss-rl/vss-pid-ddpg-output.txt
#SBATCH --error=/home/CIN/mcaa/robocin/vss-rl/vss-pid-ddpg-error.txt

module load Python/3.10
module load Xvfb/21.1.6

source /home/CIN/mcaa/vss-rl-agents/bin/activate

cd /home/CIN/mcaa/robocin/vss-with-rl

python train.py --capture-video --env Pid --setup PURE --track
