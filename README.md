# Object-Centric Learning with Slot Mixture Module
This repository is the code for the paper "Object-Centric Learning with Slot Mixture Module".

## Download datasets
Download CLEVR, ShapeStacks and CLEVRTex datasets with
```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

Remember to change the ``DATA_ROOT`` in ``downloads_data.sh`` and in ``path.json`` to your own paths.


## CLEVR property prediction
We use set prediction task to measure the performance of our approach.

## Evaluate pretrained model
To evaluate pretrained model, run this command
```bash
python eval.py
```
`e725.pth` contains default pretrained weights

## Train model from scratch
To train set prediction model, run this command:
```bash
python train_set_predictor.py 
```



## Train object discovery model
To train object discovery model, run this command with specified parameters:

- SMM
```bash
python train_smm_object_discovery.py 
```

- Base slot-attention:
```bash
python train_base_sa_object_discovery.py 
```

You can see available parameters by providing ``-h`` flag at the end.

# Citation
If you find our paper and/or code helpful, please consider citing:

```
@inproceedings{airi2024smm,
  title={Object-Centric Learning with Slot Mixture Module},
  author={Daniil Kirilenko, Vitaliy Vorobyov, Alexey K. Kovalev and Aleksandr I. Panov},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}
```

# Acknowledgement
The code uses resources from [Slot-Attention](https://github.com/google-research/google-research/tree/master/slot_attention), [slot_attention.pytorch](https://github.com/untitled-ai/slot_attention), [shapestacks](https://github.com/ogroth/shapestacks) and [BO-QSA](https://github.com/YuLiu-LY/BO-QSA). We thank authors of these wonderful projects for open-sourcing their work.
