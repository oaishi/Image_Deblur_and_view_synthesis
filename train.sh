# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

conda activate pytorch3d

export DEBUG=0
export USE_SLURM=0

# How to run on RealEstate10K
python train.py --batch-size 1 --folder 'temp' \
         --resume --dataset 'REDS' --use_inv_z --accumulation 'alphacomposite' \
         --model_type 'zbuffer_pts' --refine_model_type 'resnet_256W8UpDown64'  \
         --norm_G 'sync:spectral_batch' --gpu_ids 0 --render_ids 1 \
         --suffix '' --normalize_image --lr 0.0001

# How to run on KITTI
# python train.py --batch-size 32 --folder 'temp' --num_workers 4  \
#         --resume --dataset 'kitti' --use_inv_z --use_inverse_depth --accumulation 'alphacomposite' \
#         --model_type 'zbuffer_pts' --refine_model_type 'resnet_256W8UpDown64'  \
#         --norm_G 'sync:spectral_batch' --gpu_ids 0,1 --render_ids 1 \
#         --suffix '' --normalize_image --lr 0.0001

# # How to run on Matterport3D
# python train.py --batch-size 32 --folder 'temp' --num_workers 0  \
#        --resume --accumulation 'alphacomposite' \
#        --model_type 'zbuffer_pts' --refine_model_type 'resnet_256W8UpDown64'  \
#        --norm_G 'sync:spectral_batch' --gpu_ids 0 --render_ids 1 \
#        --suffix '' --normalize_image --lr 0.0001

#python train.py --batch-size 1 --resume --dataset "REDS" --use_inv_z --accumulation "alphacomposite" --model_type "zbuffer_pts"  --max_epoch 5000 --refine_model_type "resnet_256W8UpDown64" --norm_G "sync:spectral_batch" --gpu_ids 0 --render_ids 1 --suffix '' --normalize_image --lr 0.0001

#python train.py --batch-size 1 --resume --dataset "REDS" --use_inv_z --accumulation "alphacomposite" --model_type "zbuffer_pts"  --max_epoch 5000 --refine_model_type "resnet_256W8UpDown64" --norm_G "sync:spectral_batch" --gpu_ids 0 --render_ids 1 --suffix '' --normalize_image --lr 0.0001 --concat --log_dir "/checkpoint_concat/ow045820/logging/viewsynthesis3d/%s/"