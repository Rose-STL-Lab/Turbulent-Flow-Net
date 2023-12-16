# usage: source split.sh path_absolute/path relative to terminal 
# eg. source scripts_south/bayesian_opt/split.sh scripts_south/bayesian_opt/tfnet_rbc_data_gmm/
cd $1
echo $1
split -l 5000 -d --additional-suffix=.txt log.txt log
if [ $? -eq 0 ]; then
    echo OK, deleting log file
    rm log.txt
else
    echo split command FAILED
fi
cd ../../../