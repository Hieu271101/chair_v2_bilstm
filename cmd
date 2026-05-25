
sample
 python -m venv .venv                                                                                                    
. .\.venv\Scripts\Activate.ps1                                                                     
 python -m pip install --upgrade pip                                                              
 python render_all_sketches.py                                                                     
python prepare_base_csv.py                                                                        
python prepare_seq_csv.py                                                                         

 python train_base.py --triplets base_triplets.csv --epochs 40 --batch-size 20 --save chair_base.pth  

python train_bilstm.py --triplets seq_triplets.csv --base-model chair_base.pth --epochs 1 --batch-size 20 --save bilstm_test.pth            

python evaluate.py --base-model chair_base.pth --bilstm-model bilstm_test.pth --seq-csv seq_triplets.csv --gallery-dir ChairV2/ChairV2/photo

 python train_bilstm.py --triplets seq_triplets.csv --base-model chair_base.pth --epochs 500 --batch-size 1000 --save bilstm_chair.pth              