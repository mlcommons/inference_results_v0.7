
cat /dev/shm/log_mlperf | grep "RUNTIME" > Run.log
cat /dev/shm/log_mlperf | grep "POST" > post.log
#cat  | grep "Sort" > sort.log



echo "runtime API"
./find.sh Run.log

echo "post"
./find.sh post.log


#echo "box"
#./2find.sh box.log


