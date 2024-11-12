echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_0 --output_dir=data/abea_10_foldout_refined_nouns_0     --train_file=refined_nouns_0_train.txt --valid_file=refined_nouns_0_trial.txt --test_file=refined_nouns_0_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_0' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_0 --output_dir=data/abea_10_foldout_refined_nouns_0     --train_file=refined_nouns_0_train.txt --valid_file=refined_nouns_0_trial.txt --test_file=refined_nouns_0_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_0/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_0' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_0 --output_dir=data/abea_10_foldout_refined_nouns_0_ateacs         --train_file=refined_nouns_0_train.txt --valid_file=refined_nouns_0_trial.txt --test_file=refined_nouns_0_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_0/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_0' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_1 --output_dir=data/abea_10_foldout_refined_nouns_1     --train_file=refined_nouns_1_train.txt --valid_file=refined_nouns_1_trial.txt --test_file=refined_nouns_1_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_1' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_1 --output_dir=data/abea_10_foldout_refined_nouns_1     --train_file=refined_nouns_1_train.txt --valid_file=refined_nouns_1_trial.txt --test_file=refined_nouns_1_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_1/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_1' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_1 --output_dir=data/abea_10_foldout_refined_nouns_1_ateacs         --train_file=refined_nouns_1_train.txt --valid_file=refined_nouns_1_trial.txt --test_file=refined_nouns_1_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_1/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_1' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_2 --output_dir=data/abea_10_foldout_refined_nouns_2     --train_file=refined_nouns_2_train.txt --valid_file=refined_nouns_2_trial.txt --test_file=refined_nouns_2_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_2' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_2 --output_dir=data/abea_10_foldout_refined_nouns_2     --train_file=refined_nouns_2_train.txt --valid_file=refined_nouns_2_trial.txt --test_file=refined_nouns_2_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_2/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_2' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_2 --output_dir=data/abea_10_foldout_refined_nouns_2_ateacs         --train_file=refined_nouns_2_train.txt --valid_file=refined_nouns_2_trial.txt --test_file=refined_nouns_2_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_2/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_2' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_3 --output_dir=data/abea_10_foldout_refined_nouns_3     --train_file=refined_nouns_3_train.txt --valid_file=refined_nouns_3_trial.txt --test_file=refined_nouns_3_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_3' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_3 --output_dir=data/abea_10_foldout_refined_nouns_3     --train_file=refined_nouns_3_train.txt --valid_file=refined_nouns_3_trial.txt --test_file=refined_nouns_3_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_3/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_3' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_3 --output_dir=data/abea_10_foldout_refined_nouns_3_ateacs         --train_file=refined_nouns_3_train.txt --valid_file=refined_nouns_3_trial.txt --test_file=refined_nouns_3_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_3/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_3' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_4 --output_dir=data/abea_10_foldout_refined_nouns_4     --train_file=refined_nouns_4_train.txt --valid_file=refined_nouns_4_trial.txt --test_file=refined_nouns_4_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_4' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_4 --output_dir=data/abea_10_foldout_refined_nouns_4     --train_file=refined_nouns_4_train.txt --valid_file=refined_nouns_4_trial.txt --test_file=refined_nouns_4_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_4/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_4' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_4 --output_dir=data/abea_10_foldout_refined_nouns_4_ateacs         --train_file=refined_nouns_4_train.txt --valid_file=refined_nouns_4_trial.txt --test_file=refined_nouns_4_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_4/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_4' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_5 --output_dir=data/abea_10_foldout_refined_nouns_5     --train_file=refined_nouns_5_train.txt --valid_file=refined_nouns_5_trial.txt --test_file=refined_nouns_5_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_5' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_5 --output_dir=data/abea_10_foldout_refined_nouns_5     --train_file=refined_nouns_5_train.txt --valid_file=refined_nouns_5_trial.txt --test_file=refined_nouns_5_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_5/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_5' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_5 --output_dir=data/abea_10_foldout_refined_nouns_5_ateacs         --train_file=refined_nouns_5_train.txt --valid_file=refined_nouns_5_trial.txt --test_file=refined_nouns_5_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_5/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_5' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_6 --output_dir=data/abea_10_foldout_refined_nouns_6     --train_file=refined_nouns_6_train.txt --valid_file=refined_nouns_6_trial.txt --test_file=refined_nouns_6_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_6' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_6 --output_dir=data/abea_10_foldout_refined_nouns_6     --train_file=refined_nouns_6_train.txt --valid_file=refined_nouns_6_trial.txt --test_file=refined_nouns_6_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_6/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_6' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_6 --output_dir=data/abea_10_foldout_refined_nouns_6_ateacs         --train_file=refined_nouns_6_train.txt --valid_file=refined_nouns_6_trial.txt --test_file=refined_nouns_6_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_6/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_6' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_7 --output_dir=data/abea_10_foldout_refined_nouns_7     --train_file=refined_nouns_7_train.txt --valid_file=refined_nouns_7_trial.txt --test_file=refined_nouns_7_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_7' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_7 --output_dir=data/abea_10_foldout_refined_nouns_7     --train_file=refined_nouns_7_train.txt --valid_file=refined_nouns_7_trial.txt --test_file=refined_nouns_7_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_7/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_7' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_7 --output_dir=data/abea_10_foldout_refined_nouns_7_ateacs         --train_file=refined_nouns_7_train.txt --valid_file=refined_nouns_7_trial.txt --test_file=refined_nouns_7_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_7/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_7' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_8 --output_dir=data/abea_10_foldout_refined_nouns_8     --train_file=refined_nouns_8_train.txt --valid_file=refined_nouns_8_trial.txt --test_file=refined_nouns_8_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_8' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_8 --output_dir=data/abea_10_foldout_refined_nouns_8     --train_file=refined_nouns_8_train.txt --valid_file=refined_nouns_8_trial.txt --test_file=refined_nouns_8_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_8/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_8' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_8 --output_dir=data/abea_10_foldout_refined_nouns_8_ateacs         --train_file=refined_nouns_8_train.txt --valid_file=refined_nouns_8_trial.txt --test_file=refined_nouns_8_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_8/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_8' --classification_head='nn.linear()' && \

sleep 600

echo 'Starting new training iteration...'
CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_9 --output_dir=data/abea_10_foldout_refined_nouns_9     --train_file=refined_nouns_9_train.txt --valid_file=refined_nouns_9_trial.txt --test_file=refined_nouns_9_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 5e-05     --use_ghl --init_model pretrained_weight/pytorch_model.bin  --training_step=1 --training_name='best_model_fold_9' --classification_head='nn.linear()' && \

echo 'Step 1 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_run.py --do_train --do_eval     --data_dir=data/abea_10_fold --data_name=refined_nouns_9 --output_dir=data/abea_10_foldout_refined_nouns_9     --train_file=refined_nouns_9_train.txt --valid_file=refined_nouns_9_trial.txt --test_file=refined_nouns_9_test.gold.txt     --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2     --max_seq_length=128 --warmup_proportion=0.1     --train_batch_size 32 --num_train_epochs 10 --learning_rate 2e-05     --use_ghl --use_vat --init_model data/abea_10_foldout_refined_nouns_9/pytorch_model.bin.9 --training_step=2 --training_name='best_model_fold_9' --classification_head='nn.linear()' && \

echo 'Step 2 done...'
sleep 300

CUDA_VISIBLE_DEVICES=0 python ate_asc_run.py --do_train --do_eval         --data_dir=data/abea_10_fold --data_name=refined_nouns_9 --output_dir=data/abea_10_foldout_refined_nouns_9_ateacs         --train_file=refined_nouns_9_train.txt --valid_file=refined_nouns_9_trial.txt --test_file=refined_nouns_9_test.gold.txt         --bert_model=bert-base-uncased --do_lower_case --gradient_accumulation_steps=2         --max_seq_length=128 --warmup_proportion=0.1         --train_batch_size 32 --num_train_epochs 15 --learning_rate 5e-06         --use_ghl --init_model  data/abea_10_foldout_refined_nouns_9/pytorch_model.bin.9         --decoder_shared_layer=5 --num_decoder_layer=4 --training_step=3 --training_name='best_model_fold_9' --classification_head='nn.linear()'
sleep 600

