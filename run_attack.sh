echo ----------------------------Осуществление бэкдор-атак------------------------------

for((integer = 0; integer <= 0; integer ++))
do
  foo1="python3 attacks_crafting.py --out_dir attack$integer"
  $foo1
  foo2="python3 train_models_contam.py --attack_dir attack$integer --model_dir model$integer"
  $foo2
done

