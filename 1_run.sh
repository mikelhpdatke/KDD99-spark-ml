inotifywait -m ./pcap -e close_write |
    while read path action file; do
        ./1_script_run.sh ./pcap/$file
  	# rm -rf ./pcap/$file
    done
