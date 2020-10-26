#!/bin/bash
file_pcap=$1
./pcap2kdd.sh $file_pcap
python IDS_iot_run_v1.py

