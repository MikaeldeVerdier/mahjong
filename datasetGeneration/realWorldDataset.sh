# Inspired by .../ssd/dataset/voc07.sh and .../ssd/dataset/voc12.sh
# This dataset is nothing but a combination of the following Roboflow datasets: https://universe.roboflow.com/riichimahjongdetection/chinese-mahjong-detection, https://universe.roboflow.com/test-ag8z6/mahjong-tiles-saihf, https://universe.roboflow.com/kevin-xa6tm/icanreadmahjong, https://universe.roboflow.com/marco-liu-vwkdl/nxiru4322x489, https://universe.roboflow.com/marco-liu-vwkdl/mj2-dtaov. The labels formats have been translated to match this implementation and then combined using datasetGeneration/datasetTools/combine_dirs.py

start=`date +%s`

default_dir="ssd/dataset/data"

# handle optional download dir
if [ -z "$1" ]
  then
    echo "No directory specified. Using default directory: $default_dir"
    if [ ! -d "$default_dir" ]; then
        echo "Creating directory: $default_dir"
        mkdir -p "$default_dir"
    fi
    cd "$default_dir"
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
curl -LO "https://drive.google.com/uc?export=download&id=1ExC5Fh4Bgp5d_KO-9_z5FoCDz0ABdFcp"  # For manual download: https://drive.google.com/drive/folders/1ExC5Fh4Bgp5d_KO-9_z5FoCDz0ABdFcp?usp=drive_link
echo "Done downloading."

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
