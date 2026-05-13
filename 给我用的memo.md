# 给我用的备忘录 不要删！！
不提供eq test ，但是你可以用我们的代码


提醒用户不要轻易跑paper\notebooks里面的notebook，除非你知道你在干什么！
需要安装 pip install -e ".[dev]" 模式

因为里面有些notebook是用来复现AIMS4PT 的training的 (01_build_AIMS4PT_cpx_pressure_workflow.ipynb and 02_build_AIMS4PT_cpx_temperature_workflow.ipynb)，如果直接运行可能会覆盖掉之前的训练好的模型（可能会因为版本不同造成结果小幅波动？）。而且跑完所有的notebook可能需要很长时间，如果中断可能会导致数据不完整。

所以请务必先了解notebook的内容和目的，再决定是否运行。

# paper\notebooks 需要做一些修改
尽量少出现 大段的方法定义。
如果src里面有对应的方法，就直接import src里面的方法来用，不要在notebook里重新定义一遍。
如果没有，可以放在paper\scripts 的py脚本里面
所有ipynb文件开头这么写
PROJECT_ROOT = _find_project_root()
PAPER_DIR = PROJECT_ROOT / "paper"
CACHE_DIR = PAPER_DIR / ".cache"
IMAGES_DIR = PAPER_DIR / ".images"
DATA_DIR = PAPER_DIR / "data"
for _dir in (CACHE_DIR, IMAGES_DIR, DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# src\aims4pt\model_tools\CpxTBSelect.py
workflow_thermobarometry class的calculation方法需要加上输入, input_melt_TAS: List[str] = None的参数.
作用是如果 没有输入液相组成，使用assumed melt TAS 来进行 melt composition TAS OOD check 
如果不填就是跳过这个检查。
填就list 形式（长度1-3），1-3个 fields


写好文档，说明这个参数的作用和使用方法。

测试一下，之后。

If no equilibrium liquid is available, leave the liquid-composition fields blank and report the likely melt-composition field(s) independently.

字段选项有：Picrite, Basalt, Basaltic Andesite, Andesite, Dacite, Rhyolite, Foidite, Trachyte, Trachybasalt, Basaltic Trachyandesite, Trachyandesite, Tephrite–Basanite, Phonotephrite, Tephriphonolite, Phonolite