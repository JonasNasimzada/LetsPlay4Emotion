#!/usr/bin/env bash
#SBATCH --time 2-1
#SBATCH --partition GPUampere
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
srun python create_video_threading.py \
--input_directory /groups/constantin_students/jnasimzada/output_mesh \
--output_dir /groups/constantin_students/jnasimzada/video_head \
--thread_num 100 \
--input_vids '120514_w_56-PA2-047' '102008_w_22-BL1-088' '112009_w_43-PA3-072' '101015_w_43-PA4-071' '101015_w_43-PA3-073' '102316_w_50-PA1-024' '110909_m_29-PA2-060' '091809_w_43-PA2-010' '082109_m_53-PA2-036' '102316_w_50-PA4-018' '082809_m_26-BL1-094' '112009_w_43-PA3-041' '112310_m_20-PA3-044' '120614_w_61-PA1-037' '102008_w_22-PA1-062' '073114_m_25-PA4-062' '082809_m_26-PA2-080' '072514_m_27-PA2-049' '071709_w_23-PA1-037' '083013_w_47-PA2-029' '102316_w_50-BL1-088' '082014_w_24-PA4-062' '112914_w_51-PA2-026' '080709_m_24-PA3-064' '072714_m_23-PA1-061' '082014_w_24-PA3-044'


