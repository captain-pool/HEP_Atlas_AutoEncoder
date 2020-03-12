# HEP Atlas AutoEncoder
HEP task for Variable compression using AutoEncoders

Training Script
----------------

```bash
$ python3 main.py -d cpu --lr 0.01 --configs configs/*.yaml --type vanilla \
                  --num_steps 50000 --train_pickle_path dataset/all_jets_train_4D_100_percent.pkl \
                  --test_pickle_path dataset/all_jets_test_4D_100_percent.pkl --logdir logdir/ 
                  --export_config --export_saved_model final_saved_model
```
or if you already have a config file exported once using `--export_config` feel free to reuse that exported config file
to pass the arguments too.
```bash
$ python3 main.py --configs biggie_config.yaml
```

Check `$ python3 main.py -h` for more options

Plots for the task can be generated using:

```bash
$ python3 task.py --saved_model_path final_saved_model/ \
                  --infer_pickle_path dataset/all_jets_test_4D_100_percent.pkl \
                  --logdir logdir/inference --configs configs/*.yaml --nbins 10
```
One can also export the plots to JPEGs by passing the directory to export to using `--export_jpegs` tag
Please check `$ python3 task.py -h` for more options.
