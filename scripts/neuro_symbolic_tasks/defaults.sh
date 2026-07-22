#!/usr/bin/env bash

# Sourced by task.sh. This file defines task-specific defaults and is not a
# standalone command-line entrypoint.

neuro_symbolic_task_init_defaults() {
  declare -gA DEFAULT_TRAIN
  declare -gA DEFAULT_EVAL
  declare -gA DEFAULT_LIMIT
  declare -gA DEFAULT_PROG
  declare -gA DEFAULT_ITERS
  declare -gA DEFAULT_EPOCHS
  declare -gA DEFAULT_TNORM
  declare -gA DEFAULT_HIDDEN
  declare -gA DEFAULT_LOGICAL
  declare -gA DEFAULT_BATCH
  declare -gA DEFAULT_LR
  declare -gA USE_MNIST

  DEFAULT_TRAIN[addition]="${PROJECT_ROOT}/indexes/mnistAdd_train.txt"
  DEFAULT_EVAL[addition]="${PROJECT_ROOT}/indexes/mnistAdd_test.txt"
  DEFAULT_LIMIT[addition]=-1
  DEFAULT_PROG[addition]="${PROJECT_ROOT}/src/examples/addition/programs/addition_simple.aspif"
  DEFAULT_ITERS[addition]=1
  DEFAULT_EPOCHS[addition]=10
  DEFAULT_TNORM[addition]="product"
  DEFAULT_HIDDEN[addition]=8
  DEFAULT_LOGICAL[addition]="${LOGICAL_DIM}"
  DEFAULT_BATCH[addition]=64
  DEFAULT_LR[addition]="${LR}"
  USE_MNIST[addition]=true

  DEFAULT_TRAIN[addition2]="${PROJECT_ROOT}/indexes/mnistAdd2_train.txt"
  DEFAULT_EVAL[addition2]="${PROJECT_ROOT}/indexes/mnistAdd2_test.txt"
  DEFAULT_LIMIT[addition2]=-1
  DEFAULT_PROG[addition2]="${PROJECT_ROOT}/src/examples/addition2/programs/addition2_simple.aspif"
  DEFAULT_ITERS[addition2]=1
  DEFAULT_EPOCHS[addition2]=10
  DEFAULT_TNORM[addition2]="product"
  DEFAULT_HIDDEN[addition2]=8
  DEFAULT_LOGICAL[addition2]="${LOGICAL_DIM}"
  DEFAULT_BATCH[addition2]=64
  DEFAULT_LR[addition2]="${LR}"
  USE_MNIST[addition2]=true

  DEFAULT_TRAIN[add2x2]="${PROJECT_ROOT}/indexes/add2x2_train.txt"
  DEFAULT_EVAL[add2x2]="${PROJECT_ROOT}/indexes/add2x2_test.txt"
  DEFAULT_LIMIT[add2x2]=-1
  DEFAULT_PROG[add2x2]="${PROJECT_ROOT}/src/examples/add2x2/programs/add2x2_simple.aspif"
  DEFAULT_ITERS[add2x2]=1
  DEFAULT_EPOCHS[add2x2]=10
  DEFAULT_TNORM[add2x2]="product"
  DEFAULT_HIDDEN[add2x2]=8
  DEFAULT_LOGICAL[add2x2]="${LOGICAL_DIM}"
  DEFAULT_BATCH[add2x2]=64
  DEFAULT_LR[add2x2]="${LR}"
  USE_MNIST[add2x2]=true

  DEFAULT_TRAIN[apply2x2]="${PROJECT_ROOT}/indexes/apply2x2_train_10000.txt"
  DEFAULT_EVAL[apply2x2]="${PROJECT_ROOT}/indexes/apply2x2_test.txt"
  DEFAULT_LIMIT[apply2x2]=-1
  DEFAULT_PROG[apply2x2]="${PROJECT_ROOT}/src/examples/apply2x2/programs/apply2x2.aspif"
  DEFAULT_ITERS[apply2x2]=1
  DEFAULT_EPOCHS[apply2x2]=10
  DEFAULT_TNORM[apply2x2]="product"
  DEFAULT_HIDDEN[apply2x2]=8
  DEFAULT_LOGICAL[apply2x2]="${LOGICAL_DIM}"
  DEFAULT_BATCH[apply2x2]=64
  DEFAULT_LR[apply2x2]="${LR}"
  USE_MNIST[apply2x2]=false

  DEFAULT_TRAIN[member3]="${PROJECT_ROOT}/indexes/member3_train_10000.txt"
  DEFAULT_EVAL[member3]="${PROJECT_ROOT}/indexes/member3_test.txt"
  DEFAULT_LIMIT[member3]=-1
  DEFAULT_PROG[member3]="${PROJECT_ROOT}/src/examples/member3/programs/member3_simple.aspif"
  DEFAULT_ITERS[member3]=1
  DEFAULT_EPOCHS[member3]=10
  DEFAULT_TNORM[member3]="product"
  DEFAULT_HIDDEN[member3]=8
  DEFAULT_LOGICAL[member3]="${LOGICAL_DIM}"
  DEFAULT_BATCH[member3]=64
  DEFAULT_LR[member3]="${LR}"
  USE_MNIST[member3]=true

  DEFAULT_TRAIN[member5]="${PROJECT_ROOT}/indexes/member5_train_10000.txt"
  DEFAULT_EVAL[member5]="${PROJECT_ROOT}/indexes/member5_test.txt"
  DEFAULT_LIMIT[member5]=-1
  DEFAULT_PROG[member5]="${PROJECT_ROOT}/src/examples/member5/programs/member5_simple.aspif"
  DEFAULT_ITERS[member5]=1
  DEFAULT_EPOCHS[member5]=10
  DEFAULT_TNORM[member5]="product"
  DEFAULT_HIDDEN[member5]=8
  DEFAULT_LOGICAL[member5]="${LOGICAL_DIM}"
  DEFAULT_BATCH[member5]=64
  DEFAULT_LR[member5]="${LR}"
  USE_MNIST[member5]=true

}
