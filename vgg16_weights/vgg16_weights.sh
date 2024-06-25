# Inspired by ../dataset/voc07.sh and ../dataset/voc12.sh

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    echo "Using current directory for download ..."
  else
    # check if is valid directory
    if [ ! -d $1 ]; then
        echo $1 "is not a valid directory"
        exit 0
    fi
    echo "navigating to" $1 "..."
    cd $1
fi

echo "Downloading VGG16 weights ..."
# Download the data.
curl -L "https://drive.usercontent.google.com/u/0/uc?id=1sBmajn6vOE7qJ8GnxUJt4fGPuffVUZox&export=download" -o VGG_ILSVRC_16_layers_fc_reduced.h5
echo "Done downloading."

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"