echo ----------------------------Устранение бэкдоров ------------------------------

for((integer = 0; integer <= 0; integer ++))
do
  foo1="python3 univ_bm.py --model_dir model$integer --attack_dir attack$integer"
  $foo1

done


