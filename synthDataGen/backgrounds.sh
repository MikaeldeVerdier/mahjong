# Inspired by .../ssd/dataset/voc07.sh and .../ssd/dataset/voc12.sh

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

echo "Downloading backgrounds ..."
# Download the data.
curl -LO "https://drive.google.com/uc?export=download&id=1lmRdNj-pjnp3HsvmlEF6CSXtUqSNK9Cp"
echo "Done downloading."

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"