# QRNG
An AI suite used to find the pattern in random numbers, especially for Quantum Random Number Generators

# Install environment

create a new conda environment with python=, then run:
```
pip install -r requirements.txt
```

# Running
A typical running scripts looks like this:
```
python run_lightning.py --gpus=3 --epochs=1000 --learning_rate=1e-4 --batch_size=32768 --data_dir=/home/hh/bell_test_9994.dat --seqlen=256 --num_class=16 --model=FC --predictor=normal
```
- --gpus is the number of gpus you want to use
- --learning_rate is the learning rate for the model, you can leave it at default.
- --epoch is the number of epochs you want to run, usually 40 is enough
- --batch_size determines the batch_size,  depends on the size of your gpu memory. Usually the bigger, the better, but it is limited by the gpu's memory. You can begin with a large number, then decrease it half at a time until there is no OOM CUDA ERROR.
- --data_dir determines where your data is in. The data should be stored as pure form of data
- --seqlen the length for the input for the history. The larger, the more gpu memories are needed.
- --num_class The number of classes network outputs. Depends on the type of problem you are in, for Bell Test data, usually it is 16, for normal QRNG, it is 256
- --model is the type of model you want to use, you can choose from FC,CNN(normal data) or BellFC,BellCNN(Bell Test data)
- --predictor is type of predictor. Set to normal for normal data and Set to bell for Bell test data