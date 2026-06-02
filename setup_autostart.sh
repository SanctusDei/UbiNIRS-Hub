#!/bin/bash

echo "======================================"
echo "  NIRScanner 自动部署与自启动配置脚本"
echo "======================================"

# 1. 自动获取绝对路径
APP_DIR=$(pwd)
USER_NAME=$(whoami)
# 强制让 Python 路径指向主目录 (~) 下的 my_env
PYTHON_PATH="/home/$USER_NAME/my_env/bin/python3"
SCRIPT_PATH="$APP_DIR/gui_app.py"

echo "[1/5] 正在校验路径..."
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ 错误：找不到 Python 解释器 ($PYTHON_PATH)"
    echo "请确保你的虚拟环境 my_env 真的在 /home/$USER_NAME/ 下！"
    exit 1
fi
echo "✔️ Python 路径: $PYTHON_PATH"
echo "✔️ 脚本 路径: $SCRIPT_PATH"

# 2. 生成中转包装脚本 (解决环境变量与屏幕权限)
echo "[2/5] 正在创建启动包装脚本 (start_gui_root.sh)..."
WRAPPER_PATH="/home/$USER_NAME/start_gui_root.sh"
cat <<INNER_EOF > "$WRAPPER_PATH"
#!/bin/bash
export DISPLAY=:0
export XAUTHORITY=/home/$USER_NAME/.Xauthority
# 清除 SSH 转发干扰
unset SSH_AUTH_SOCK
# 以 Root 身份启动，并原封不动地继承显示环境
sudo -E $PYTHON_PATH $SCRIPT_PATH
INNER_EOF
chmod +x "$WRAPPER_PATH"

# 3. 自动配置 Sudo 免密 (使用更安全的 sudoers.d 方法)
echo "[3/5] 正在配置 Root 免密权限..."
SUDOERS_FILE="/tmp/nirscanner_sudo"
# 写入特定命令的免密规则
echo "$USER_NAME ALL=(ALL) NOPASSWD: $PYTHON_PATH $SCRIPT_PATH" > "$SUDOERS_FILE"
# 修正权限并移动到系统目录
sudo chown root:root "$SUDOERS_FILE"
sudo chmod 0440 "$SUDOERS_FILE"
sudo mv "$SUDOERS_FILE" /etc/sudoers.d/nirscanner_sudo

# 4. 自动写入桌面自启动配置文件
echo "[4/5] 正在配置桌面 Autostart 自动启动..."
AUTOSTART_DIR="/home/$USER_NAME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
DESKTOP_PATH="$AUTOSTART_DIR/nirscanner.desktop"

cat <<INNER_EOF > "$DESKTOP_PATH"
[Desktop Entry]
Type=Application
Name=NIRScanner GUI
Exec=$WRAPPER_PATH
Terminal=false
X-GNOME-Autostart-enabled=true
INNER_EOF

echo "[5/5] 部署完成！"
echo "======================================"
echo "所有的路径、权限和启动项已全部自动配置完毕。"
echo "请直接在终端输入: sudo reboot"
echo "重启后你的 GUI 将自动出现在外接屏幕上！"
