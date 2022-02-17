
python train.py --name FLIR_NTIR2DC --dataroot ./FLIR_datasets/ --n_domains 2 --niter 40 --niter_decay 40 --loadSize 288 --fineSize 256 --resize_or_crop crop --IR_edge_path ./FLIR_IR_edge_map/ --Vis_edge_path ./FLIR_Vis_edge_map/ --grad_th_vis 0.8 --grad_th_IR 0.8 --SSIM_start_epoch 10 --SSIM_fullload_epoch 10 --ACCS_start_epoch 10 --gpu_ids 0
