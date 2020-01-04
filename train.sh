#!/bin/bash

for i in `seq 3`; do
  echo 'EPOCH: ' $i
  ansible-playbook -i inventory.yaml train.yaml
done;