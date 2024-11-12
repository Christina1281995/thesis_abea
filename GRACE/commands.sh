echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/same_split_as_absa/ --data_name=abea_w_none_clean --output_dir=data/same_split_as_absa/out_ABEA_0.05decay_0.25DO_5freeze     --train_file=abea_w_none_clean_train.txt --valid_file=abea_w_none_clean_trial.txt --test_file=abea_w_none_clean_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 5 --learning_rate 3e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='ABEA_0.05decay_0.25DO_5freeze' --classification_head='1x linear layer' && \

echo 'Step 1 done...'
sleep 60

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/same_split_as_absa/ --data_name=abea_w_none_clean --output_dir=data/same_split_as_absa/out_ABEA_0.05decay_0.25DO_5freeze     --train_file=abea_w_none_clean_train.txt --valid_file=abea_w_none_clean_trial.txt --test_file=abea_w_none_clean_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 3 --learning_rate 1e-05     --use_ghl --use_vat --init_model data/same_split_as_absa/out_ABEA_0.05decay_0.25DO_5freeze/pytorch_model.bin.4 --training_step=2 --training_name='ABEA_0.05decay_0.25DO_5freeze' --classification_head='1x linear layer' && \

echo 'Step 2 done...'
sleep 60

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/same_split_as_absa/ --data_name=abea_w_none_clean --output_dir=data/same_split_as_absa/out_ABEA_0.05decay_0.25DO_5freeze_ateacs         --train_file=abea_w_none_clean_train.txt --valid_file=abea_w_none_clean_trial.txt --test_file=abea_w_none_clean_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 10 --learning_rate 3e-06         --use_ghl --init_model  data/same_split_as_absa/out_ABEA_0.05decay_0.25DO_5freeze/pytorch_model.bin.2         --decoder_shared_layer=9 --num_decoder_layer=2 --training_step=3 --training_name='ABEA_0.05decay_0.25DO_5freeze' --classification_head='1x linear layer'
sleep 100

