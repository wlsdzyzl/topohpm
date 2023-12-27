#!/bin/bash
echo "WITHOUT TL"
echo "UNET"
python /home/wlsdzyzl/project/topohpm/topohpm/scripts/evaluate3d.py -p /home/wlsdzyzl/project/topohpm/generated_files/predictions/3d_asoca/UNET -g /media/wlsdzyzl/DATA2/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo "SKELETON"
python /home/wlsdzyzl/project/topohpm/topohpm/scripts/evaluate3d.py -p /home/wlsdzyzl/project/topohpm/generated_files/predictions/3d_asoca/SKELETON_LOSS_0.6_0.5 -g /media/wlsdzyzl/DATA2/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo "WITH DTL"
echo "UNET"
python /home/wlsdzyzl/project/topohpm/topohpm/scripts/evaluate3d.py -p /home/wlsdzyzl/project/topohpm/generated_files/predictions/3d_asoca/UNET_DTL -g /media/wlsdzyzl/DATA2/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo "SKELETON"
python /home/wlsdzyzl/project/topohpm/topohpm/scripts/evaluate3d.py -p /home/wlsdzyzl/project/topohpm/generated_files/predictions/3d_asoca/SKELETON_LOSS_0.6_0.5_DTL -g /media/wlsdzyzl/DATA2/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo "WITH WTL"
echo "UNET"
python /home/wlsdzyzl/project/topohpm/topohpm/scripts/evaluate3d.py -p /home/wlsdzyzl/project/topohpm/generated_files/predictions/3d_asoca/UNET_WTL -g /media/wlsdzyzl/DATA2/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label
echo "SKELETON"
python /home/wlsdzyzl/project/topohpm/topohpm/scripts/evaluate3d.py -p /home/wlsdzyzl/project/topohpm/generated_files/predictions/3d_asoca/SKELETON_LOSS_0.6_0.5_WTL -g /media/wlsdzyzl/DATA2/datasets/ASOCA/nii/cropped_middle_zoomed_72_128_128/data/label