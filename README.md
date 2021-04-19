# TreeDecoder

The source codes has been released, will make it clear for those who are not familiar with deep learning and encoder-decoder models:<br>

The data will be released after it is prepared:<br>

* **Tree Decoder**: A Tree-Structured Decoder for Image-to-Markup Generation<br>


## Quick Start
### 1. Prepocess

Preprocessing of training set. (.pkl)
<br/>
`python data/gen_pkl.py --dataset_type CROHME --op_mode train`

Preprocessing of test set. (.pkl)
<br/>
`python data/gen_pkl.py --dataset_type CROHME --op_mode test`

### 2. Generate ME vocabulary
`python data/gen_voc.py --dataset_type CROHME`

### 3. Generate GTD files
`python codes/latex2gtd --dataset_type CROHME`

### 4. Generate GTD label & align file (.pkl)
`python codes/prepare_label.py --dataset_type CROHME`

### 5. Training model
`python codes/train_wap.py --dataset_type CROHME`

### 6. Testing model
`python codes/translate.py --dataset_type CROHME --batch_size 8 --K 112 --k 3 --model_path ../train/models/210418/WAP_params_last.pkl --dictionary_target ../data/CROHME/dictionary.txt --dictionary_retarget ../data/CROHME/relation_dictionary.txt --fea ../data/CROHME/image/offline-test.pkl --output_path ../test/`
