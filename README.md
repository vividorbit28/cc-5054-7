# 分层视频子流提取与预览

针对 x265 生成的 **真实时间层** HEVC 片段，工具读取每个 NAL 的层号（`TemporalId`）和
帧间引用（RPS），输出指定层集合的**合法子码流**，再用**独立解码会话**验证切换前后的
时间对齐、图像质量与真实字节数，最后在浏览器窗口里拖动切换点即时重建并播放。

```
tools/make_sample.py  →  data/sample_4layer.h265  (4 个时间层, 96 帧, 3 GOP)
layered_video/bitstream.py   纯解析：层号、POC、实际参考帧（不解码）
layered_video/extractor.py   按策略裁剪 + 依赖检查 + 序列化 Annex-B
layered_video/decode.py      严格解码、逐帧 PSNR、恢复判定
layered_video/locator.py     按访问单元跟踪的断链定位（故障归因/缩减/修复）
layered_video/analysis.py    plan → sub-stream → 校验（CLI 与服务器共用）
layered_video/cli.py         命令行
layered_video/server.py      HTTP + H.264 预览代理
layered_video/static/        可拖动切换点的界面
```

## 快速开始

```bash
python tools/make_sample.py                 # 生成 4 层样本（需已安装 PyAV/libx265）
python -m layered_video.cli info data/sample_4layer.h265
python -m layered_video.server --stream data/sample_4layer.h265
# 打开 http://127.0.0.1:8765/
```

依赖：Python 3.11、PyAV（FFmpeg 8）、numpy。浏览器端只用到原生 `<video>`。

## 两个目标时刻（同层号，一次缺增强参考）

保留层集合固定为 `{0,1,2}`。片段在显示序 6 处，tid-2 的帧引用了显示序 4（tid-1）和
显示序 2（tid-2）的增强帧，而这两个帧在“切换前只放基础层”的裁剪里被丢掉了；
显示序 10 处的同层集合请求则无需任何额外参考。

```
cli verify --at 6  --layers 0,1,2 --policy naive     # 拒绝：缺显示 4(tid1)、2(tid2)
cli verify --at 6  --layers 0,1,2 --policy preroll   # 切换@6，预取 [4,2]，最早恢复 6
cli verify --at 6  --layers 0,1,2 --policy defer     # 切换@10，最早恢复 10
cli verify --at 10 --layers 0,1,2 --policy defer     # 切换@10，无需预取
```

三种策略的取舍在界面同一标尺上并列可见：**延后**牺牲切换时刻，**预取**牺牲字节
（多带两个前置增强帧），**朴素截断**被提取器直接拒绝。

## 校验口径

* `bitstream.py` 不经过解码器，直接解析每个图像的首个 slice 头，取 `TemporalId`、
  `slice_pic_order_cnt_lsb` 与 RPS，再按 GOP 解析出**实际引用的帧**。
* 子码流交给**全新的** `av.CodecContext`，并开启
  `err_detect=…+explode` 与 `ec=0`：缺少参考帧会抛异常，而不是让解码器做错误隐藏。
* 逐帧判定只在以下条件全部成立时记作“恢复”：
  该帧存在于子流、干净解码、**不与前一帧重复**、与整段干净解码逐像素一致（PSNR 阈值）、
  且层号属于目标增强层。被强制截断的非法流一律不计。
* 字节数取自子流本身：`图像字节 + 参数集字节 = 子流字节`，并核对访问单元数。

## 界面

* 标尺上灰色是低带宽段（层 0），蓝色是升层段，竖线处是切换点，可拖动重建。
* 绿点标出达到目标质量的帧，黄点标出为精确切换而预取的参考帧。
* 右侧同时给出请求时刻、实际切换、延后帧数、预取帧、子流字节、错误隐藏/重复计数
  和最早恢复帧。
* 浏览器无法解码 HEVC，因此预览是**仅供观看**的 H.264 代理；所有数字都来自对原始
  HEVC 子码流的严格解码。页面底部另有“下载 HEVC 子流”。

## 断链定位（locator）

对一个**解码冻结**的子流，`locator.py` 按访问单元归因：先把子流切成 AU
（`first_slice_segment_flag`，无需参数集），按 VCL 内容摘要对齐回源流，于是源流
的 RPS 依赖关系直接成为子流的依赖视图。两个单变量假设实验区分根因——**只替换
参数集**或**只恢复一条引用**——哪个能让独立解码恢复，根因就是哪个。随后：

* 缩减器在保持故障签名（同一源 AU、同一错误类、配置状态与缺引用状态不变）的
  前提下删到最短子流；
* 修复只应用被证明的那一处改动（补配置 = 修正提取器；补引用闭包 = 修正能力
  判定），绝不把整条原流贴回去，并从实际解码画面标出首帧恢复位置；
* 对能完整解码的合法片段，`required_segments` 给出逐段必需性：删掉仍能解码的
  冗余参数集**不**会被报作必需条件；
* 从故障 AU 向前找最近可独立进入的 IRAP（实际解码验证），给出可跳转的恢复位置。

```bash
python -m layered_video.cli diagnose sub.h265 --source data/sample_4layer.h265 [--json]
```

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：真实层号（直接从 NAL 头复核）、同一层集合的两处时刻、提取器拒绝破损裁剪、
严格解码拒绝缺参考、子流逐像素一致、延后切换不提前报恢复、强制截断不计恢复、
字节账目与访问单元数。
