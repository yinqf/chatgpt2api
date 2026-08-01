docker buildx build \
  --platform linux/amd64 \
  -t registry.cn-hongkong.aliyuncs.com/yinqf/chatgpt2api:1.0.0 \
  . \
  --push