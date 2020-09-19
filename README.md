# CED : Credible Early Detection of Social Media Rumors

The experiment code of [CED](http://114.215.64.60:8094/~chm/publications/tkde2019_CED.pdf), implemented in Python 2.7 and Tensorflow 1.3.0. 
Due to the large INPUT gap between the proposed model and baseline models, we organize each model into a separate .py file. Later we will modularize the shared code between these models and reconstruct the whole network.

## Models:
  - 1_CNN_OM: CNN just to deal with original microblogs.
  - 2_TF_IDF: SVM classifier using TF-IDF representation vector from 10_parted_posts_seqvec.txt, which has already batched N=10 consecutive reposts together.
  - 3_GRU_2: 2-layer GRU to deal with repost sequences.
  - 4_CAMI: Our self-implementation model of [CAMI](http://ir.ia.ac.cn/bitstream/173211/19743/1/ijcai17.pdf).
  - 5_(1/2/3)_CED: Our proposed model CED,CED-OM and CED-CNN.

## Input Files:
  - class_8050.json: Class label and repost feature length of file_name. All 8050 samples are from [Rumdect](http://alt.qcri.org/⇠wgao/data/rumdect.zip) and our published dataset [Chinese_Rumor_Dataset](https://github.com/thunlp/Chinese_Rumor_Dataset).
    ```sh
    {"file_name1":{"class":[0,1], "len":5}, "file_name2":{"class":[1, 0], "len":16}, ......}
    ```
  - msg_id.json/txt: Padded word embedding ID of original message.
    ```sh
    {"file_name1":[15029,4890,2332,3380,382,6019,320,8524,671,0], "file_name2":[2003,60,1390,0,0,0,0,0,0,0], ......}
    ```
  - post_id.json: Padded word embedding ID of repost message. 
    ```sh
    {"file_name1":[[22,31,1866,468,1170,469,220,5285, ...], [1102,1712,1304,930,127,1712,193,22, ...], ...], "file_name2":[[...], [...], ...].shape = [Padded length of N reposts' words, Corresponding "len" in class_8050.json], ......}
    ```
  - 10_parted_posts_seqvec.txt: Padded TF-IDF features only for CAMI. Still N=10 to compare with other models.

## Train
```sh
$ # python model_name_to_train.py , for example:
$ python 5_3_CED_CNN.py
```

We also show our experiment exvironment in requorements.txt file.

## Citation

If you use this code for research, please cite our paper as follows:

```
@article{song2019ced,
  title={CED: credible early detection of social media rumors},
  author={Song, Changhe and Yang, Cheng and Chen, Huimin and Tu, Cunchao and Liu, Zhiyuan and Sun, Maosong},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2019},
  publisher={IEEE}
}
```
