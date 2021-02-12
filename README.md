## I-BERT Tutorial

### Requirements

#### 1. Software requirement

- Python >3.6
- Numpy >1.18
- PyTorch >1.7
- CUDA
- pytorch-dnc[https://github.com/ixaxaar/pytorch-dnc]

Using NVIDIA-Docker is not mandatory, but we strongly recommend using NVIDIA-Docker pytorch version download from https://ngc.nvidia.com/catalog/containers/nvidia:pytorch

#### 2. Hardware requirement

- Nvidia GPU with CUDA Capability<br>

#### 3. Installation

Clone our repository, then run the codes below for installation. Our repository will be available to the public once the paper review process completes.

```
# Create necessary directories
mkdir log
mkdir output_dir
```

```
# Generate listops dataset
python listops.py
```

Or, you can run the all-in-one script file

```bash
./run.sh
```

### Running I-BERT

-----

I-BERT can be simply run from Bash. Below is the most core command line to run I-BERT. 

```bash
python3 AutoEncode.py --net ibert
```

Then, you should be able to see a message `Executing Autoencoder model with IBERT's Architecture.` After then, if you see the message as in the below screen (`Epoch #1 ... Perplexity : ...`), it is highly likely that I-BERT will be successfully running on your computer.

```
Epoch #1:
train seq acc:	0.016640625
train loss:	0.36817466298118234
Current LR:2.91e-05
Training Perplexity :	4.0623637819561855
Train sequences per second : 122.06420314033554
val accuracy at 13 digits = 0.0
val accuracy at 14 digits = 0.0
val accuracy at 15 digits = 0.0
val accuracy at 16 digits = 0.0
validation loss:	0.4672691101829211
Perplexity :	3.2456890745652585
```

The program will be automatically terminated after training/validating 50 epochs in the default setting.

For `listops` dataset you need to run a different code: 

```bash
python3 Classifier.py --net ibert --model_size small
```

### Options

-----

Our program supports multiple command-line options to provide a better user experience. The below table shows major options that can be simply appended when running the program.

#### Number Sequence Dataset, Penn Tree Bank Dataset (AutoEncode.py)

| Options      | Default | Description                                                  |
| ------------ | ------- | ------------------------------------------------------------ |
| --net        | tf      | Model for your task <br>ibert: I-BERT <br>xlnet: XLNet<br>lstm: LSTM seq2seq <br>tf: BERT <br>ibertpos: I-BERT with positional encoding <br>ibert2: I-BERT2<br>dnc: Differentiable Neural Computer<br>ut: Universal Transformer |
| --seq_type   | fib     | task for prediction <br>fib: addition task (NSP Dataset)<br>copy: copy task (NSP Dataset)<br>palin: reverse task (NSP Dataset)<br>ptbc: Penn Tree Bank Character<br>ptbw: Penn Tree Bank Word |
| --digits     | 12      | Max number of training digits <br>(Only supports for algorithmic tests) |
| --model_size | 512     | Tiny: L=2 H=2 D=128 <br>Mini: L=4 H=4 D=256<br/>Small: L=4 H=4 D=512<br/>Medium: L=8 H=8 D=512 <br/>Base: L=12 H=12 D=768 |
| --batch_size | 32      | Number of epochs                                             |
| --epochs     | 50      | Number of epochs                                             |
| --lr         | 3e-5    | Learning rate                                                |
| --log        | false   | Log training/validation results                              |
| --exp        | 0       | Assign log file identifier when --log is true                |

For example, if we want to run a `Penn Tree Bank Word` dataset with 100 epochs with I-BERT, we can try the following: 

```bash
python3 AutoEncode.py --net ibert --seq_type ptbw --epochs 100
```



#### Listops Dataset (Classifier.py)

| Options      | Default | Description                                                  |
| ------------ | ------- | ------------------------------------------------------------ |
| --net        | tf      | Model for your task <br>ibert: I-BERT <br>xlnet: XLNet<br>lstm: LSTM seq2seq <br>tf: BERT <br>ibertpos: I-BERT with positional encoding <br>ibert2: I-BERT2<br>dnc: Differentiable Neural Computer<br>ut: Universal Transformer |
| --model_size | 512     |Tiny: L=2 H=2 D=128 <br>Mini: L=4 H=4 D=256<br/>Small: L=4 H=4 D=512<br/>Medium: L=8 H=8 D=512 <br/>Base: L=12 H=12 D=768 |
| --batch_size | 32      | Number of epochs                                             |
| --epochs     | 50      | Number of epochs                                             |
| --lr         | 3e-5    | Learning rate                                                |


### Sample result

-----

Just as in paper, we provide multiple kinds of metrics including training sequence accuracy, training/validation accuracy, and perplexity. Below shows an example output at epoch 50 after executing `python3 AutoEncode.py --net ibert`.

```
Epoch #50:
train seq acc:	0.9978515625
train loss:	0.0002572423391580969
Current LR:6.541961260422218e-06
Training Perplexity :	1.0010830311149899
Train sequences per second : 120.4091252320059
val accuracy at 13 digits = 0.9166666666666666
val accuracy at 14 digits = 0.8125
val accuracy at 15 digits = 0.6614583333333334
val accuracy at 16 digits = 0.359375
validation loss:	0.055553702671507686
Perplexity :	1.15058179827412
```


### Result Analysis

-----

If you choose to log the experiment results, they will be saved in the directory `/log/`. The name of the log file follows the format below. 

```
1 2020-05-23 04/25/16 fib ibert.log
```

- 1 here represents the identifier number of each experiment produced by `--exp <N>  ` where `<N>` is an integer
- 2020-05-23 refers to `year-month-date`
- 04/25/16 shows the `hour/min/sec` when the program is executed for the first time. 
- `fib` is the dataset used for the experiment produced by `--seq_type <dataset> ` where `<dataset>` can be among `fib, copy, palin, ptbc, ptbw` 
- `ibert` is the model used for the experiment produced by `--net <model>` where `<model>` can be among `ibert, xlnet, lstm, tf, ibertpos, ibert2`.
