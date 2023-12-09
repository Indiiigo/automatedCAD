## to get pvi values on the train-like test sets

We reuse the following command from https://github.com/kawine/dataset_difficulty, but change the datasets:

```
v_info(
  f"./local_data/snli_test_std.csv",
  f"models/finetuned/roberta-base_snli_std2",
  f"./local_data/snli_test_null.csv", 
  f"models/finetuned/roberta-base_snli_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_std2_test.csv"
)
```

### for sexism

First, we train a model on mixed CAD data and test on mixed CAD data
```
v_info(
  f"./local_data/sexism_CAD_mixed_2_dev.csv",
  f"models/finetuned/roberta-base_sexism_CAD_mixed_2",
  f"./local_data/sexism_CAD_mixed_2_dev_null.csv", 
  f"models/finetuned/roberta-base_sexism_CAD_mixed_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_sexism_CAD_mixed_2_dev_T_sexism_CAD_mixed.csv"
)
```
Second, we train on OG data and test on OG data

```
v_info(
  f"./local_data/sexism_OG_2_dev.csv",
  f"models/finetuned/roberta-base_sexism_OG_2",
  f"./local_data/sexism_CAD_mixed_2_dev_null.csv", 
  f"models/finetuned/roberta-base_sexism_OG_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_sexism_OG_2_dev_T_sexism_OG.csv"
)
```

Cross-dataset  is also possible, but it may violate V-information's IID assumption:
```
v_info(
  f"./local_data/sexism_OG_2_dev.csv",
  f"models/finetuned/roberta-base_sexism_CAD_mixed_2",
  f"./local_data/sexism_OG_2_dev_null.csv", 
  f"models/finetuned/roberta-base_sexism_CAD_mixed_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_sexism_OG_2_dev_T_sexism_CAD_mixed.csv"
)
```

### for HS


```
v_info(
  f"./local_data/hatespeech_CAD_mixed_2_dev.csv",
  f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2",
  f"./local_data/hatespeech_CAD_mixed_2_dev_null.csv", 
  f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_hatespeech_CAD_mixed_2_dev_T_hatespeech_CAD_mixed.csv"
)
```

```
v_info(
  f"./local_data/hatespeech_OG_2_dev.csv",
  f"models/finetuned/roberta-base_hatespeech_OG_2",
  f"./local_data/hatespeech_CAD_mixed_2_dev_null.csv", 
  f"models/finetuned/roberta-base_hatespeech_OG_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_hatespeech_OG_2_dev_T_hatespeech_OG.csv"
)
```

```
v_info(
  f"./local_data/hatespeech_OG_2_dev.csv",
  f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2",
  f"./local_data/hatespeech_OG_2_dev_null.csv", 
  f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_hatespeech_OG_2_dev_T_hatespeech_CAD_mixed.csv"
)
```

--------------------------------------------------------------------------------------------

for ood_test sets:

```
v_info(
  	f"./local_data/sexism_2_ood_test.csv",
  	f"models/finetuned/roberta-base_sexism_OG_2",
  	f"./local_data/sexism_2_ood_test_null.csv", 
  	f"models/finetuned/roberta-base_sexism_OG_2_null",
  	'roberta-base',
  	out_fn=f"PVI/roberta-base_sexism_2_ood_test_T_sexism_OG.csv"
	)
```

```
v_info(
  	f"./local_data/sexism_2_ood_test.csv",
  	f"models/finetuned/roberta-base_sexism_CAD_mixed_2",
  	f"./local_data/sexism_2_ood_test_null.csv",
  	f"models/finetuned/roberta-base_sexism_CAD_mixed_2_null",
  	'roberta-base',
  	out_fn=f"PVI/roberta-base_sexism_2_ood_test_T_sexism_CAD_mixed.csv"
	)
```

```
v_info(
  f"./local_data/hatespeech_2_ood_test.csv",
  f"models/finetuned/roberta-base_hatespeech_OG_2",
  f"./local_data/hatespeech_2_ood_test_null.csv", 
  f"models/finetuned/roberta-base_hatespeech_OG_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_hatespeech_2_ood_test_T_hatespeech_OG.csv"
)
```

```
v_info(
  f"./local_data/hatespeech_2_ood_test.csv",
  f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2",
  f"./local_data/hatespeech_2_ood_test_null.csv",
  f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2_null",
  'roberta-base',
  out_fn=f"PVI/roberta-base_hatespeech_2_ood_test_T_hatespeech_CAD_mixed.csv"
)
```

--------------------------------------------------------------------------------------------


#### for ranking training instances:

```
v_info(
  	f"./local_data/sexism_CAD_mixed_2_train.csv",
  	f"models/finetuned/roberta-base_sexism_CAD_mixed_2",
  	f"./local_data/sexism_CAD_mixed_2_train_null.csv", 
  	f"models/finetuned/roberta-base_sexism_CAD_mixed_2_null",
  	'roberta-base',
  	out_fn=f"PVI/roberta-base_sexism_2_CAD_train_T_sexism_CAD.csv"
	)
```

```
v_info(
  	f"./local_data/hatespeech_CAD_mixed_2_train.csv",
  	f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2",
  	f"./local_data/hatespeech_CAD_mixed_2_train_null.csv", 
  	f"models/finetuned/roberta-base_hatespeech_CAD_mixed_2_null",
  	'roberta-base',
  	out_fn=f"PVI/roberta-base_hatespeech_2_CAD_train_T_hatespeech_CAD.csv"
	)
```
