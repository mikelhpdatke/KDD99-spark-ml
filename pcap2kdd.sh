#!/bin/bash
bro -C -r $1 darpa2gurekddcup.bro > conn.list
sort -n conn.list > conn_sort.list
./trafAld.out conn_sort.list
python kdd.py trafAld.list
