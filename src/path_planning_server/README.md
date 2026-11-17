# Path Planning Server

该 ROS 2 包包含双 Piper 的 MoveIt 2 规划服务、任务状态机、静态场景配置和测试。工作空间说明与构建步骤见[项目首页](../../README.md)。

## 路径与阶段

```mermaid
flowchart LR
    Start[当前位姿] -->|OMPL| Pre[预抓取]
    Pre -->|笛卡尔路径| Grasp[抓取]
    Grasp -->|笛卡尔路径| Retreat[回退]
    Retreat -->|OMPL 途径点| Place[放置]
    Place -->|笛卡尔路径| Exit[离开放置区]
```

`config/ompl_waypoints/` 保存分场景和任务阶段的路线；`config/scenes/` 描述静态碰撞环境；`config/grasp_bridge_state_machine.right.yaml` 配置高层任务参数。`src/` 包含规划节点与插件实现，`scripts/` 包含任务状态机与相机数据同步逻辑。

运行前需要按实际机器人重新检查坐标系、相机标定、途径点和碰撞模型。饮料包装模型与现场标定数据由使用者单独配置。
