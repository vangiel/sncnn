# sncnn
Using a CNN for training on the SocNav2 dataset

python3 train.py --dataroot ./images_dataset --dataset_mode map --name map__pix2pix --model pix2pix --direction AtoB --input_nc 9 --output_nc 1

python3 test.py --dataroot ./images_dataset --dataset_mode map --direction AtoB --model pix2pix --name map__pix2pix --input_nc 9 --output_nc 1