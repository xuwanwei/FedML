# IC
#python ./main_fed3.py \
#--gpu 0 \
#--dataset mnist \
#--data_dir ./../../../data/mnist \
#--model lr \
#--partition_method hetero \
#--client_num_in_total 100 \
#--client_num_per_round 6 \
#--comm_round 100 \
#--epochs 5 \
#--batch_size 10 \
#--client_optimizer sgd \
#--ci 0 \
#--budget_per_round 200 \
#--frequency_of_the_test 1 \
#--seed 1103 \
#--draw True

# test with rounds
python ./main_fedbf.py \
--gpu 0 \
--dataset mnist \
--data_dir ./../../../data/mnist \
--model lr \
--partition_method hetero \
--client_num_in_total 20 \
--client_num_per_round 6 \
--comm_round 5 \
--epochs 5 \
--batch_size 10 \
--client_optimizer sgd \
--ci 0 \
--budget_per_round 20 \
--frequency_of_the_test 1 \
--seed 1103 \
--draw True
