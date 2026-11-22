# 外部示例数据

实验使用 BUILTNYU/ridepooling 的 Sioux Falls 公开示例前 240 个请求，保留原到达时刻。源版本为 `fe3af6015fe651d52f911b6ab96f30cbb0b9de4f`，读取脚本逐文件核验源数据。

阅读并确认上游许可后，通过以下命令获取数据：

```bash
python scripts/fetch_example_data.py --accept-source-license --prefix 240 --out data/private_example
```

源文件下载至 `data/private_example/`；派生决策矩阵、模型参数和实验结果已随工程提供。数据获取规则和来源校验值见 `scripts/fetch_example_data.py`。
