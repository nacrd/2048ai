# 上游与二开来源

- 上游：https://github.com/gabrielecirulli/2048
- 固定提交：`478b6ec346e3787f589e4af751378d06ded4cbbc`
- Fork：https://github.com/nacrd/2048ai
- 原版权：Copyright (c) 2014 Gabriele Cirulli；根目录 `LICENSE.txt` 保留原 MIT 许可证。

原版 `js/`、字体、样式和 meta 资产保留。生产页面使用新的 TypeScript 纯引擎、Worker 求解和动态布局；规则与视觉继承原版，动画适配为动态尺寸。未修改的 `js/game_manager.js`、`js/grid.js` 和 `js/tile.js` 是规则对照基准，不由生产页面加载。

同步上游前先检查规则差异：`git fetch upstream`，不要直接覆盖二开页面。测试绑定上述提交，不把上游任意新版本的结果默认为本版规则。
