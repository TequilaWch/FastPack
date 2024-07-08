# FastPack
Implementation of "FastPack: Rapid Network Traffic Classification with Limited Packet Learning".

If you find this method helpful for your research, please cite this paper:
> Our paper is currently under review and has not been published yet.

## Requirement
- python == 3.7.16
- numpy == 1.21.5
- tensorflow == 2.6.0
- pandas == 1.3.5
- scikit-learn == 1.0.2
- pyshark == 0.6
- seaborn
- matplotlib
- tqdm

## Dataset Format
The raw input data consists of multiple files, and are stored in the Data directory as *ApplicationName_time.csv*.

The files in *Data* need further processing. The processed data is then saved into another CSV file and stored in *Processed Data*.

The processed CSV file contains the following fields:
> frame.len, frame.cap_len, ip.hdr_len, ip.dsfield.ecn, ip.len, ip.frag_offset, ip.ttl, tcp.hdr_len, tcp.len,tcp.flags.ns, tcp.flags.fin, tcp.window_size_value, tcp.urgent_pointer, udp.length, protocal, srcport, dstport,timestamp, ip.src, ip.dst, protocol, app

A data packet is stored in the CSV file as follows:
> 1306, 68, 20, 0, 1290, 0, 88, 32, 1238, 0, 0, 260, 0, TCP, 443, 49639, 1554037442312280000, 192.124.220.225, 10.145.102.237, TCP, instagram

## How to use
### Step 1. Pre-Process The Dataset
To process the data into a format acceptable for the model, first extract the raw data packets from the PCAP files and store them as *ApplicationName.csv* in *Data*. Then, execute the following code:
> python main.py --mode=prepro

The dataset will saved in *Trainer_Input* folder.
### Step 2. Train The Model
We can train our model by:
> python main.py --mode=train

The trained model will be saved in *Model* folder.
### Step 3. Evaluation
We can conduct the evaluation with:
> python main.py --mode=eva

This will output the various metrics of the model on the validation set.