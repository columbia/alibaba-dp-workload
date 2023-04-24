mkdir -p cluster-trace-gpu-v2020/data
cd cluster-trace-gpu-v2020/data

curl -O https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_group_tag_table.header
curl -O https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_instance_table.header
curl -O https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_job_table.header
curl -O https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_machine_metric.header
curl -O https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_machine_spec.header
curl -O https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_sensor_table.header
curl -O https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020/data/pai_task_table.header

curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_group_tag_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_instance_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_job_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_metric.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_machine_spec.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_sensor_table.tar.gz
curl -O https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/pai_task_table.tar.gz

for file in `ls *.tar.gz`; do tar -xzf $file; done