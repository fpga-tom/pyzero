#!/bin/bash

for i in `seq 60`; do
  echo 'EPOCH: ' $i
  ansible-playbook -i inventory.yaml train.yaml
done;