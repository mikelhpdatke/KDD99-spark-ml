#!/bin/bash
file_pcap=$1
./pcap2kdd.sh $file_pcap
python3 IDS_iot_run_v1.py

