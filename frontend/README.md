# 知应 AI｜多 Agent 智能客服平台前端

这是知应 AI 的 Vue 3 + Vite 管理与调试界面，已作为 `frontend/` 子目录并入主项目。

## 功能

- 智能客服多轮对话
- 健康检查与运行状态查看
- RAG 知识库检索
- 知识文档导入和文件上传
- 对话意图、Agent 类型、知识库使用及人工升级标记展示
- 本地 Vite 开发和 Docker + Nginx 部署

## 本地开发

先启动父目录中的 Python 后端，默认地址为 `http://localhost:8000`。

安装依赖并运行前端：

```bash
npm ci
npm run dev
```

访问 `http://localhost:5173`。

开发服务器会把 `/api/zhiying/*` 代理到 `http://localhost:8000/*`。如后端地址不同，可以覆盖：

```bash
VITE_ZHIYING_API_URL=http://localhost:8000 npm run dev
```

## 随主项目部署

在父目录执行：

```bash
docker compose up --build
```

访问地址：

- 完整应用入口：`http://localhost`
- 前端容器直连：`http://localhost:5174`
- 后端 Swagger：`http://localhost:8000/docs`

整套 Compose 中，前端 Nginx 通过 Docker 网络将 `/api/zhiying/*` 代理到 `zhiying:8000`。

## 单独运行前端容器

后端先在宿主机 `localhost:8000` 启动，然后在本目录执行：

```bash
docker compose up --build
```

访问 `http://localhost:5174`。独立 Compose 会将请求代理到 `host.docker.internal:8000`。

## 配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VITE_ZHIYING_API_URL` | `/api/zhiying` | Vite 构建或开发时的 API 基础地址 |
| `ZHIYING_API_UPSTREAM` | 由 Compose 提供 | Nginx 运行时的后端 upstream，不包含协议 |
| `NGINX_ENVSUBST_FILTER` | `ZHIYING_API_UPSTREAM` | 限制 Nginx 模板只替换知应 AI upstream |

## 构建验证

```bash
npm ci
npm run build
```

构建产物输出到 `dist/`。