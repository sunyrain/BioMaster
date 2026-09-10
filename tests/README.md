# 测试分层

当前测试按职责分为三类，不以文件名中的内部版本号判断是否“最新”：

1. `test_current_project_contract.py`：最上层口径门禁，防止384/450/745、86,674/437,248、KIRHub功能抑制/亲和力等定义再次漂移。
2. 当前主线回归：`test_bidirectional_v6.py`、`test_comprehensive_balanced.py`、`test_drug_centric_ranker_v1.py`、`test_target_discovery_scope_ch37_v3.py`。
3. 专项或历史复现：结构流程、旧ODTI数据合同、KIRHub功能适配及交付脚本测试。它们不自动代表当前生产模型，但在对应实现仍保留时继续作为回归保护。

删除测试前必须先删除或替换其对应实现与复现承诺。仓库清理不通过删测试制造“更简洁”的假象；默认只从README和CI入口隔离历史测试。
