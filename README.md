## Installation

It is recommended to create the environment from the provided file first:

```bash
conda env create -f environment.base.yml
conda activate AIMS4PT
```

Then install the package from the project root:

```bash
pip install .
```


For development mode:

```bash
pip install -e ".[dev]"
```


提醒用户不要轻易跑paper\notebooks里面的notebook，除非你知道你在干什么！
需要安装 pip install -e ".[dev]" 模式

因为里面有些notebook是用来复现AIMS4PT 的training的 (01_build_AIMS4PT_cpx_pressure_workflow.ipynb and 02_build_AIMS4PT_cpx_temperature_workflow.ipynb)，如果直接运行可能会覆盖掉之前的训练好的模型（可能会因为版本不同造成结果小幅波动？）。而且跑完所有的notebook可能需要很长时间，如果中断可能会导致数据不完整。

所以请务必先了解notebook的内容和目的，再决定是否运行。

