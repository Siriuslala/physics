#!/bin/bash

RETRY_DELAY=10

while true; do

    huggingface-cli download \
        --repo-type dataset \
        --resume-download \
        qihoo360/WISA-80K \
        --revision dddbd5683581c2ebf0b463e2b1c3342b2094bfb3 \
        --local-dir /datacache/huggingface/hub/datasets--qihoo360--WISA-80K \
        --include 'data/videos/*'
    
    if [ $? -eq 0 ]; then
        echo "===== ✅ Download Finished ====="
        break
    else
        echo "===== ❌ Download Failed, ${RETRY_DELAY} seconds later ====="
        sleep $RETRY_DELAY
    fi
done