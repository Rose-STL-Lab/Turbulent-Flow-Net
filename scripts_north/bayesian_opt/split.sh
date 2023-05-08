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