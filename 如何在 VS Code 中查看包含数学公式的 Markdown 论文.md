# 在 VS Code 中查看包含数学公式的 Markdown 论文

## 前提说明

学术论文通常包含大量复杂的 LaTeX 数学公式（如 `\mathfrak`、`\underset`、`\mathcal{L}`、`\sum` 等），VS Code 的**原生 Markdown 预览**不支持这些公式的渲染。需要安装专用插件并配置快捷键。

---

## 步骤一：安装必要插件

安装以下插件以支持复杂数学公式渲染：

| 插件名称 | 扩展 ID | 作用 |
|:---|:---|:---|
| Markdown All in One | `yzhang.markdown-all-in-one` | 提供 Markdown 增强功能 |
| Markdown Preview Enhanced | `shd101wyy.markdown-preview-enhanced` | 提供数学公式渲染（核心） |

**安装方法：**

1. 按 `Ctrl+Shift+X` 打开扩展市场
2. 搜索上述插件名称或扩展 ID
3. 点击 `Install` 安装

---

## 步骤二：配置快捷键

VS Code 默认的 `Ctrl+Shift+V` 绑定的是原生预览，需要将其覆盖为 Markdown Preview Enhanced 的预览命令。

### 2.1 打开用户级快捷键配置文件

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Preferences: Open Keyboard Shortcuts (JSON)`
3. 选择该命令，打开**用户级**的 `keybindings.json` 文件

> 📌 该文件路径示例：
> - Windows：`%APPDATA%\Code\User\keybindings.json`
> - Linux/macOS：`~/.config/Code/User/keybindings.json`

### 2.2 添加快捷键覆盖规则

在打开的 `keybindings.json` 文件中，添加以下内容：

```json
[
    {
        "key": "ctrl+shift+v",
        "command": "markdown-preview-enhanced.openPreview",
        "when": "editorLangId == markdown && !terminalFocus && !notebookEditorFocused"
    }
]