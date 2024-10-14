salloc --ntasks=1 --cpus-per-task=4 --mem=32G --time=24:00:00 --partition=htzhulab --gres=gpu:1 --qos=gpu_access
# It should show something like this: salloc: Nodes g180702 are ready for job
ssh g180702  # modify according to information
cd /work/users/y/u/yuukias/Annotation/Segment_Anything
./gui.sh
