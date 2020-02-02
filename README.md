# CIFAR10 classification API using bazel

**prerequisites**

User must have cuda 9.0 installed in current system.

Run command to install requirements in local system basically to  create enviornment to run ***bazel run*** command.
```
pip install -r requirements.txt
```

## To run training and classificartion API

Run command to train model:
```
bazel run :CIFAR10_serving_trainer
```

Run command to run prediction API:
```
bazel run :CIFAR10_serving_prediction
```

After running API command as per above one can run following commmand template from new terminal:
```
curl -F "file=@<file path>" http://<localhost:port>/predict_image
```
