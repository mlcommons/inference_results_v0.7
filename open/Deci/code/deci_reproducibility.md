# Usage of Deci's code and models
- Clone the official MLPerf inference repo `git clone https://github.com/mlperf/inference.git`
- Checkout to v0.7 branch `git checkout r0.7`
- Copy and overwrite the `.py` files from the `code` folder in this submission `cp deci/code/* PATH_TO_PROJECT/code/python/`
- Download Deci's models and copy them into `PATH_TO_PROJECT/code/python/models/`:
`wget https://deci-model-repository-research.s3.amazonaws.com/mlperf_models/MODEL_NAME`
- The `MODEL_NAME` must be one of the following:
	- *deci-model-macbookpro-batch-1.zip*
	- *deci-model-macbookpro-batch-64.zip*
	- *deci-model-cascadelake-batch-1.zip*
	- *deci-model-cascadelake-batch-64.zip*
- Follow the MLPerf instruction with `--profile deci_model#-openvino` (`#` = 1 or 64):