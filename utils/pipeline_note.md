# Pipeline for LKML Mining
转换LKML邮件数据为结构化的事件数据
python /Users/luyaobai/Workspace/lkml-mining/utils/dataset_to_event_json_v3.py \
  --db /Users/luyaobai/Workspace/lkml-mining/data/data-20180101-20181231.db \
  --out /Users/luyaobai/Workspace/lkml-mining/data/2018/jsondata

把邮件内容区分为代码和非代码部分
python /Users/luyaobai/Workspace/lkml-mining/utils/split_code_from_content.py \
  --input /Users/luyaobai/Workspace/lkml-mining/data/2018/jsondata \
  --output /Users/luyaobai/Workspace/lkml-mining/data/2018/depart_jsondata

合并同一事件的多封邮件
python /Users/luyaobai/Workspace/lkml-mining/utils/merge_event_messages_by_code.py \
  --input /Users/luyaobai/Workspace/lkml-mining/data/2018/depart_jsondata \
  --output /Users/luyaobai/Workspace/lkml-mining/data/2018/mergedata

对同一个事件的不同版本 进行聚类分组
python /Users/luyaobai/Workspace/lkml-mining/utils/event_series_group.py \
  --input /Users/luyaobai/Workspace/lkml-mining/data/2018/mergedata \
  --output /Users/luyaobai/Workspace/lkml-mining/data/2018/event_series

# LLM processing
打标签
python /Users/luyaobai/Workspace/lkml-mining/utils/tag_mails_v2.py \
  --input /Users/luyaobai/Workspace/lkml-mining/data/2018/mergedata \
  --output /Users/luyaobai/Workspace/lkml-mining/data/2018/events_tagged

python utils/filtered_mails_security.py \
  --input /Users/luyaobai/Workspace/lkml-mining/data/2018/jsondata \
  --output /Users/luyaobai/Workspace/lkml-mining/data/2018/security_related \
  --model qwen3 \
  --concurrency 10


python /Users/luyaobai/Workspace/lkml-mining/utils/security_filter_series.py \
  --input /Users/luyaobai/Workspace/lkml-mining/data/2018/event_series \
  --output_true /Users/luyaobai/Workspace/lkml-mining/data/2018/classed_event_series/security \
  --output_false /Users/luyaobai/Workspace/lkml-mining/data/2018/classed_event_series/non_security
