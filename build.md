# 构建并推送 fork 镜像

在仓库根目录运行，镜像标签取自 `VERSION`。

```sh
docker buildx build \
  --platform linux/amd64 \
  --tag "registry.cn-hongkong.aliyuncs.com/yinqf/chatgpt2api:$(cat VERSION)" \
  . \
  --push
```
