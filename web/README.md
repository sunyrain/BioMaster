# BioMaster Discovery Atlas

React + TypeScript 研究可视化界面，数据由仓库内 Python 服务提供。完整使用说明见 [平台文档](../docs/BIOMASTER_EXPLORER_ZH.md)。

```bash
# 项目根目录
cd web
npm ci
npm run build
cd ..
python scripts/run_explorer.py
```

访问 http://127.0.0.1:18765。远程 IDE 在 Ports（端口）面板中将远程 18765 转发至本机 18765；如果 IDE 分配了其他本地端口，请使用该本地端口访问。

开发时运行后端，再在 `web/` 执行 `npm run dev`。Vite 将 `/api` 转发至 `127.0.0.1:18765`。

界面只展示已有实体与实际结果。新分子／靶点需要在项目数据流程中先完成登记、特征与评分；缺失结果保留为空。

`public/fonts/` 是随页面本地提供的 DM Sans 与 Manrope 字体及其 OFL 许可证。运行界面不需要远程字体或 3D 库 CDN。
