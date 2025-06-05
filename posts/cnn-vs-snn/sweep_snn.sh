for seed_everything in 1 2 3;
do
    # for time_window in 1000 5000 10000;
    # do
    python train_snn.py fit --config config.yaml --data data_config_raster.yaml --seed_everything=$seed_everything #--data.post_slicing_transform.init_args.time_window=
    # done
done