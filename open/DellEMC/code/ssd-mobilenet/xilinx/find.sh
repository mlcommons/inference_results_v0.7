cat ${1} |awk '{print $3}' > log2
k=0
sum=0
for i in `cat log2`
do
temp=${i%u*}
k=$[`expr $k+1`]
sum=$[`expr $sum+$temp`]
done
echo test rounds:  $k 
#echo $sum
avg=$[`expr $sum/$k`]
#echo $k
#for i in `cat log2`
#do
#temp=${i%u*}
#if [ $k -gt $temp ]; then
# k=$temp
#fi
#done
echo average: $avg
