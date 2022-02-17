
python train.py --name KAIST_NTIR2DC --dataroot ./KAIST_datasets/ --n_domains 2 --niter 60 --niter_decay 60 --loadSize 288 --fineSize 256 --resize_or_crop crop --IR_edge_path ./KAIST_IR_edge_map/ --Vis_edge_path ./KAIST_Vis_edge_map/ --grad_th_vis 0.44 --grad_th_IR 0.44 --SSIM_start_epoch 30 --SSIM_fullload_epoch 30 --ACCS_start_epoch 30 --gpu_ids 0
