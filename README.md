# 【IEEE TMM 2024】SMC-NCA: Semantic-guided Multi-level Contrast for Semi-supervised Temporal Action Segmentation

## This repository contains the code for SMC-NCA: Semantic-guided Multi-level Contrast for Semi-supervised Temporal Action Segmentation [paper](https://arxiv.org/abs/2312.12347), which has been accepted for IEEE Transactions on Multimedia. 

### Requirements
* PyTorch 1.12.1

* Torchvision 0.13.1



### Dataset:

The I3D features, ground-truth and test split files are similar used to [ICC](https://github.com/dipika-singhania/ICC-Semi-Supervised-TAS). To test our approach, we have provided a checkpoint for unsupervised representation learning on 50Salads.

The data directory is arranged in following structure:

- mstcn_data
   - mapping.csv
   - dataset_name
   - groundTruth
   - splits
   - semi_supervised 
   - results
        - SMC
            - unsupervised experiments
            - semi-supervised experiments

### Run Scripts

#### Train Unsupervised Contrastive Representation Learning SMC
    ##### python unsupervised_traineval.py --base_dir mstcn_data/<dataset_name>/ --output_dir <output_directory_to_dump_modelcheckpoint_logs>
    Example:
    python unsupervised_traineval.py --output_dir mstcn_data/50salads/results/SMC/ --base_dir mstcn_data/50salads/


#### Evaluate Unsupervised Contrastive Representation Learning SMC
    ##### python unsupervised_traineval.py --base_dir mstcn_data/<dataset_name>/ --output_dir <output_directory_to_dump_modelcheckpoint_logs> --eval
    Example:
    python unsupervised_traineval.py --output_dir mstcn_data/50salads/results/SMC/ --base_dir mstcn_data/50salads/ --eval

#### Train SMC-NCA for particular split
    ##### python semi_supervised_train.py --split_number <split_number> --semi_per <semi_supervised_percentage> --base_dir mstcn_data/<datasetname>/ --output_dir <output_directory_to_dump_model_checkpoints_logs> --model_wt <unsupervised_representation_modelwt> 
    Example:
    python semi_supervised_train.py --split_number 1 --semi_per 0.05 --output_dir mstcn_data/50salads/results/SMC/ --base_dir mstcn_data/50salads/ --model_wt mstcn_data/50salads/results/SMC/unsupervised_C2FTCN_splitfull/best_50salads_c2f_tcn.wt 


### Acknowledgements
This code and data processing is based on [ICC](https://github.com/dipika-singhania/ICC-Semi-Supervised-TAS). 
Thanks to the authors for their work!

### References

If you find this work helpful, please consider citing our paper
```
@article{zhou2023smc,
  title={SMC-NCA: Semantic-guided Multi-level Contrast for Semi-supervised Action Segmentation},
  author={Zhou, Feixiang and Jiang, Zheheng and Zhou, Huiyu and Li, Xuelong},
  journal={arXiv preprint arXiv:2312.12347},
  year={2023}
}
```

