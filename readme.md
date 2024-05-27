## 创建 docker container

## 各种环境报错。
1. `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

```bash
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y
```

TODO: 在 dockerfile 中增加下面部分
```bash
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

