 ActivityNet SRL (ASRL) is from ActivityNet Captions (AC) and ActivityNet Entities (AE) datasets.




## Preparing Data

 download the data through   download_data.sh:

Optional: set the data folder.
```
cd $ROOT/data
bash download_data.sh all [data_folder]
```

After everything is downloaded successfully, the folder structure should look like:

```
data
|-- anet (530gb)
    |-- anet_detection_vg_fc6_feat_100rois.h5
    |-- anet_detection_vg_fc6_feat_100rois_resized.h5
    |-- anet_detection_vg_fc6_feat_gt5_rois.h5
    |-- fc6_feat_100rois
    |-- fc6_feat_5rois
    |-- rgb_motion_1d
|-- anet_cap_ent_files (31M)
    |-- anet_captions_all_splits.json
    |-- anet_ent_cls_bbox_trainval.json
    |-- csv_dir
	 |-- train.csv
	 |-- train_postproc.csv
	 |-- val.csv
	 |-- val_postproc.csv
    |-- dic_anet.json
|-- anet_srl_files (112M)
    |-- arg_vocab.pkl
    |-- trn_asrl_annots.csv
    |-- trn_srl_obj_to_index_dict.json
    |-- val_asrl_annots.csv
    |-- val_srl_obj_to_index_dict.json
```

It should ~530 gb of data !!

NOTE: Highly advisable to have the features in SSD; otherwise massive drop in speed!

