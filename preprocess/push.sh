#!/bin/bash

######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

#Login
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 999701187340.dkr.ecr.us-west-2.amazonaws.com

# Push Docker image
docker push 999701187340.dkr.ecr.us-west-2.amazonaws.com/wandb
