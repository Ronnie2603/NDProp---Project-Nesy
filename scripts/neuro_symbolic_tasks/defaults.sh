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

  local task
  for task in addition addition2 add2x2 apply2x2 member3 member5; do
    DEFAULT_LIMIT["${task}"]=-1
    DEFAULT_ITERS["${task}"]=1
    DEFAULT_EPOCHS["${task}"]=10
    DEFAULT_TNORM["${task}"]="product"
    DEFAULT_HIDDEN["${task}"]=8
    DEFAULT_LOGICAL["${task}"]=1
    DEFAULT_BATCH["${task}"]=64
    DEFAULT_LR["${task}"]=1e-3
    USE_MNIST["${task}"]=true
  done

  DEFAULT_TRAIN[addition]="${PROJECT_ROOT}/indexes/mnistAdd_train.txt"
  DEFAULT_EVAL[addition]="${PROJECT_ROOT}/indexes/mnistAdd_test.txt"
  DEFAULT_PROG[addition]="${PROJECT_ROOT}/src/examples/addition/programs/addition_simple.aspif"

  DEFAULT_TRAIN[addition2]="${PROJECT_ROOT}/indexes/mnistAdd2_train.txt"
  DEFAULT_EVAL[addition2]="${PROJECT_ROOT}/indexes/mnistAdd2_test.txt"
  DEFAULT_PROG[addition2]="${PROJECT_ROOT}/src/examples/addition2/programs/addition2_simple.aspif"

  DEFAULT_TRAIN[add2x2]="${PROJECT_ROOT}/indexes/add2x2_train.txt"
  DEFAULT_EVAL[add2x2]="${PROJECT_ROOT}/indexes/add2x2_test.txt"
  DEFAULT_PROG[add2x2]="${PROJECT_ROOT}/src/examples/add2x2/programs/add2x2_simple.aspif"

  DEFAULT_TRAIN[apply2x2]="${PROJECT_ROOT}/indexes/apply2x2_train_10000.txt"
  DEFAULT_EVAL[apply2x2]="${PROJECT_ROOT}/indexes/apply2x2_test.txt"
  DEFAULT_PROG[apply2x2]="${PROJECT_ROOT}/src/examples/apply2x2/programs/apply2x2.aspif"
  USE_MNIST[apply2x2]=false

  DEFAULT_TRAIN[member3]="${PROJECT_ROOT}/indexes/member3_train_10000.txt"
  DEFAULT_EVAL[member3]="${PROJECT_ROOT}/indexes/member3_test.txt"
  DEFAULT_PROG[member3]="${PROJECT_ROOT}/src/examples/member3/programs/member3_simple.aspif"

  DEFAULT_TRAIN[member5]="${PROJECT_ROOT}/indexes/member5_train_10000.txt"
  DEFAULT_EVAL[member5]="${PROJECT_ROOT}/indexes/member5_test.txt"
  DEFAULT_PROG[member5]="${PROJECT_ROOT}/src/examples/member5/programs/member5_simple.aspif"

}
