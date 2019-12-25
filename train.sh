#!/bin/bash

for i in `seq 5`; do
  echo 'Round: ' $i
  ansible-playbook -i inventory.yaml train.yaml
done;