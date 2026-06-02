"""Task management mixin for SpectrometerApp.

Provides: DB init, task CRUD, task page (card-based browser),
new-task page (scrollable form), tag management, swipe navigation.
"""

import os, json, shutil, sqlite3, tkinter as tk
import numpy as np
from tkinter import ttk, messagebox
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
import joblib


class TaskManagerMixin:
    """Task DB + UI mixed into SpectrometerApp."""

    # ── Name sanitization ─────────────────────────────────────────────
    @staticmethod
    def _sanitize_name(name):
        """Sanitize task name for use in filenames."""
        safe = name.replace(" ", "_")
        safe = safe.replace("/", "_")
        safe = safe.replace("\\", "_")
        safe = safe.replace(":", "_")
        safe = safe.replace(".", "_")
        return safe

    # ── DB ──────────────────────────────────────────────────────────
    def _init_db(self):
        """初始化 SQLite 数据库结构"""
        # db_path
        self.db_path = "spectral_tasks.db"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # task 表格

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                task_type TEXT,
                algorithm TEXT,
                model_path TEXT,
                is_trained BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()


    def _load_tasks_from_db(self):
        """从 SQLite 数据库读取所有任务并刷新到界面的内存列表中"""
        self.tasks = [] # 每次读取前先清空旧列表，防止重复
        try:

            conn = sqlite3.connect(self.db_path)
            # 这一行是魔法！它让 SQLite 返回字典 (dict) 而不是难懂的元组 (tuple)
            # 这样查出来的每一行数据，都会自动变成 {'name': 'Task1', 'algorithm': 'SVM'...} 的格式
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 查询所有任务
            cursor.execute("SELECT * FROM tasks")
            rows = cursor.fetchall()

            # 将数据库的数据转换回你 UI 需要的字典列表
            for row in rows:
                self.tasks.append(dict(row))

            conn.close()
            print(f"[SYS] 界面刷新成功！当前数据库共有 {len(self.tasks)} 个任务。")
            self.available_predict_tasks = [t for t in self.tasks if t.get("is_trained") == 1]

        except Exception as e:
            print(f"[DB ERROR] 从数据库读取任务列表失败: {e}")

    # ── Task List page (card browser) ────────────────────────────────
    def build_task_panel(self):

        if hasattr(self, "_load_tasks_from_db"):
          self._load_tasks_from_db()
        else:
          self.tasks = []

        self.current_task_idx = 0
        self.is_animating = False

        tk.Label(self.task_page, text="[ TASK LIST ]", font=self.FONT_TITLE, bg=self.SCREEN_BG, fg=self.SCREEN_FG).pack(pady=(15, 10))

        self.task_card = tk.Canvas(self.task_page, bg="#b0b0b0", bd=4, relief=tk.SUNKEN, highlightthickness=0)
        self.task_card.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 15))

        # Initialize text at coordinates (0,0), will be centered by _resize_task_card
        self.task_name_txt = self.task_card.create_text(0, 0, text="", font=("Courier New", 16, "bold"), fill="#111111", anchor="center")
        self.task_desc_txt = self.task_card.create_text(0, 0, text="", font=("Courier New", 12), fill="#333333", anchor="center", justify="center")
        self.task_num_txt = self.task_card.create_text(0, 0, text="", font=("Courier New", 11, "bold"), fill="#555555", anchor="center")

        # Static directional arrows
        self.left_arrow = self.task_card.create_text(0, 0, text="<", font=("Courier New", 26, "bold"), fill="#7a7a7a")
        self.right_arrow = self.task_card.create_text(0, 0, text=">", font=("Courier New", 26, "bold"), fill="#7a7a7a")

        # Bind dragging and resizing events
        self.task_card.bind("<Configure>", self._resize_task_card)
        self.task_card.bind("<ButtonPress-1>", self._on_swipe_start)
        self.task_card.bind("<B1-Motion>", self._on_swipe_drag)
        self.task_card.bind("<ButtonRelease-1>", self._on_swipe_end)

        # Clickable arrows
        self.task_card.tag_bind(self.left_arrow, "<ButtonRelease-1>", lambda e: self._prev_task())
        self.task_card.tag_bind(self.right_arrow, "<ButtonRelease-1>", lambda e: self._next_task())

        self.swipe_start_x = 0

        btn_frame = tk.Frame(self.task_page, bg=self.SCREEN_BG)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=40, pady=(0, 20))
        btn_frame.columnconfigure(0, weight=1); btn_frame.columnconfigure(1, weight=1); btn_frame.columnconfigure(2, weight=1)

        tk.Button(btn_frame, text="[ SELECT ]", font=self.FONT_SYS, command=self._select_task, bg="#4caf50", fg="#111", relief=tk.RAISED, bd=3).grid(row=0, column=0, padx=5, sticky="ew")
        tk.Button(btn_frame, text="[ DELETE ]", font=self.FONT_SYS, command=self._delete_task, bg="#f44336", fg="#111", relief=tk.RAISED, bd=3).grid(row=0, column=1, padx=5, sticky="ew")
        tk.Button(btn_frame, text="[ NEW ]", font=self.FONT_SYS, command=self._new_task, bg="#a3a3a3", fg="#111", relief=tk.RAISED, bd=3).grid(row=0, column=2, padx=5, sticky="ew")

        self._update_task_card_text()

    # ── Task card animations ─────────────────────────────────────────
    def _resize_task_card(self, event):
        """Triggers when UI is loaded or resized. Keeps layout perfectly symmetrical."""
        if getattr(self, 'is_animating', False): return
        w, h = event.width, event.height
        if w < 10 or h < 10: return

        self.task_card.coords(self.left_arrow, 30, h/2)
        self.task_card.coords(self.right_arrow, w - 30, h/2)
        self.task_card.itemconfig(self.task_desc_txt, width=w - 100)
        self._set_text_x_offset(0) # Reset to absolute center

    def _set_text_x_offset(self, offset_x):
        """Moves only the text elements left or right by the offset amount."""
        w, h = self.task_card.winfo_width(), self.task_card.winfo_height()
        if w < 10: return
        base_x = w / 2

        self.task_card.coords(self.task_name_txt, base_x + offset_x, h * 0.2)
        self.task_card.coords(self.task_desc_txt, base_x + offset_x, h * 0.5)
        self.task_card.coords(self.task_num_txt, base_x + offset_x, h * 0.85)

    def _on_swipe_start(self, event):
        if getattr(self, 'is_animating', False) or not self.tasks: return
        self.swipe_start_x = event.x

    def _on_swipe_drag(self, event):
        """Live dragging visual effect. Text physically follows your mouse/finger."""
        if getattr(self, 'is_animating', False) or not self.tasks: return
        offset_x = event.x - self.swipe_start_x
        self._set_text_x_offset(offset_x)

    def _on_swipe_end(self, event):
        """Evaluates whether to snap back or switch to the next task."""
        if getattr(self, 'is_animating', False) or not self.tasks: return
        offset_x = event.x - self.swipe_start_x

        if offset_x > 80:
            self._animate_out_and_switch(1)  # Swiped right -> Previous Task
        elif offset_x < -80:
            self._animate_out_and_switch(-1) # Swiped left -> Next Task
        else:
            self._animate_restore(offset_x)  # Swiped too little -> Snap Back

    def _animate_out_and_switch(self, direction):
        """Smooth slide out, switch content, and slide back in."""
        self.is_animating = True
        w = self.task_card.winfo_width()
        current_offset = self.task_card.coords(self.task_name_txt)[0] - (w/2)

        def step_out(offset):
            next_offset = offset + direction * 40 # Speed of slide out
            if abs(next_offset) > w:
                # Arrived out of bounds, switch the underlying data
                if direction == 1:
                    self.current_task_idx = (self.current_task_idx - 1) % len(self.tasks)
                else:
                    self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)

                self._update_task_card_text()
                # Teleport text to the opposite edge to prepare for slide-in
                step_in(-direction * w)
            else:
                self._set_text_x_offset(next_offset)
                self.root.after(15, lambda: step_out(next_offset)) # 60fps refresh rate

        def step_in(offset):
            move_step = 40 if offset < 0 else -40 # Speed of slide in
            next_offset = offset + move_step

            # Snap to center when crossing the 0 point
            if (move_step > 0 and next_offset >= 0) or (move_step < 0 and next_offset <= 0):
                self._set_text_x_offset(0)
                self.is_animating = False
            else:
                self._set_text_x_offset(next_offset)
                self.root.after(15, lambda: step_in(next_offset))

        step_out(current_offset)

    def _animate_restore(self, current_offset):
        """Elastic snap-back animation if the user doesn't swipe far enough."""
        self.is_animating = True

        def step_restore(offset):
            if abs(offset) <= 25:
                self._set_text_x_offset(0)
                self.is_animating = False
                return

            move_step = -25 if offset > 0 else 25
            next_offset = offset + move_step
            self._set_text_x_offset(next_offset)
            self.root.after(15, lambda: step_restore(next_offset))

        step_restore(current_offset)

    def _prev_task(self):
        if getattr(self, 'is_animating', False) or not self.tasks: return
        self._animate_out_and_switch(1)

    def _next_task(self):
        if getattr(self, 'is_animating', False) or not self.tasks: return
        self._animate_out_and_switch(-1)

    def _update_task_card_text(self):
        """智能解析 JSON 数据并美化显示"""
        if not self.tasks:
            self.task_card.itemconfig(self.task_name_txt, text="[ NO TASKS AVAILABLE ]")
            self.task_card.itemconfig(self.task_desc_txt, text="Please create a new task.")
            self.task_card.itemconfig(self.task_num_txt, text="[ Task 0 / 0 ]")
            return

        task = self.tasks[self.current_task_idx]
        self.task_card.itemconfig(self.task_name_txt, text=task["name"])

        # 🚨 修复乱码：把 JSON 字符串还原成好看的说明文字
        try:
            meta = json.loads(task["description"])
            clean_desc = meta.get("raw_desc", "No description provided.")
            type_info = meta.get("task_type", task.get("task_type", "N/A"))
            algo_info = meta.get("algorithm", task.get("algorithm", "N/A"))

            # 拼装出带有科技感的展示文本
            display_text = f"{clean_desc}\n\n[ {type_info} | {algo_info} ]"
        except:
            # 如果是旧版本创建的非 JSON 任务，直接显示
            display_text = task["description"]

        self.task_card.itemconfig(self.task_desc_txt, text=display_text)
        self.task_card.itemconfig(self.task_num_txt, text=f"[ Task {self.current_task_idx + 1} / {len(self.tasks)} ]")

    # ── Task actions ─────────────────────────────────────────────────
    def _select_task(self):
        if getattr(self, 'is_animating', False): return

        if not self.tasks:
            self._lock_task_controls()
            self.show_execute_page()
            return

        task = self.tasks[self.current_task_idx]
        selected_name = task["name"]
        algorithm = task.get("algorithm", "UNKNOWN")
        is_trained = bool(task.get("is_trained", 0))

        self.current_task_var.set(f"CURRENT TASK: {selected_name}")

        if hasattr(self, 'status_display_var'):
            if is_trained:
                self.status_display_var.set(f"[ MODEL READY ]\n\nALGO: {algorithm}\nSTATUS: READY FOR INFERENCE")
                # 解锁全部按钮，并赋予色彩
                if hasattr(self, 'predict_btn'): self.predict_btn.config(state=tk.NORMAL, bg="#4682B4")
                if hasattr(self, 'train_btn'): self.train_btn.config(state=tk.NORMAL, bg="#2E8B57")
            else:
                self.status_display_var.set(f"[ WARN: MODEL UNTRAINED ]\n\nALGO: {algorithm}\nPLEASE TRAIN THE MODEL FIRST.")
                # 预测禁用，但允许训练
                if hasattr(self, 'predict_btn'): self.predict_btn.config(state=tk.DISABLED, bg="#5a5a5a")
                if hasattr(self, 'train_btn'): self.train_btn.config(state=tk.NORMAL, bg="#2E8B57")

        self.show_execute_page()

    def _lock_task_controls(self):
        """统一锁定所有任务相关按钮"""
        if hasattr(self, 'predict_btn'):
            self.predict_btn.config(state=tk.DISABLED, bg="#5a5a5a")
        # �修复：无任务时，强行锁死训练按钮
        if hasattr(self, 'train_btn'):
            self.train_btn.config(state=tk.DISABLED, bg="#5a5a5a")

        self.current_task_var.set("CURRENT TASK: NONE")
        if hasattr(self, 'status_display_var'):
            self.status_display_var.set("[ NO TASK SELECTED ]\n\nPlease select or create a task.")

    def _delete_task(self):
        if getattr(self, 'is_animating', False) or not self.tasks: return

        # Get the task dictionary and its name
        task = self.tasks[self.current_task_idx]
        task_name = task["name"]
        task_id = task.get("id")

        # Pop up a confirmation dialog
        confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete:\n\n[{task_name}]?\n(Model and Data will be archived to 'dash/')")

        # Proceed only if user clicks 'Yes'
        if confirm:
            try:
                # 1. 从 SQLite 数据库中删除记录
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id, ))
                conn.commit()
                conn.close()
                print(f"[SYS] Task '{task_name}' (ID: {task_id}) deleted from DB.")

                # 确保 dash 文件夹存在
                dash_dir = "dash"
                if not os.path.exists(dash_dir): os.makedirs(dash_dir)

                # 2. 归档模型文件 (.pkl)
                model_path = task.get("model_path")
                if model_path and os.path.exists(model_path):
                    file_name = os.path.basename(model_path)
                    target_path = os.path.join(dash_dir, file_name)
                    shutil.move(model_path, target_path)
                    print(f"[SYS] Model archived: {target_path}")

                # 3. 🚨 归档数据集文件 (.csv) - 新增逻辑
                data_file_name = f"task_{task_id}_dataset.csv"
                data_source_path = os.path.join("data", data_file_name)
                data_target_path = os.path.join(dash_dir, data_file_name)

                if os.path.exists(data_source_path):
                    shutil.move(data_source_path, data_target_path)
                    print(f"[SYS] Dataset archived: {data_target_path}")
                else:
                    print(f"[INFO] No dataset file found for archiving: {data_source_path}")

                # 3.5 归档 NPY 特征缓存和 JSONL 性能日志 (name-based + ID-based)
                safe_name = self._sanitize_name(task_name)
                for prefix in [f"task_{safe_name}", f"task_{task_id}"]:
                    for suffix in ["_X.npy", "_y.npy", "_perf.jsonl", "_dataset.csv"]:
                        fpath = os.path.join("data", f"{prefix}{suffix}")
                        if os.path.exists(fpath):
                            target = os.path.join(dash_dir, f"{prefix}{suffix}")
                            shutil.move(fpath, target)
                            print(f"[SYS] Archived: {target}")

            except Exception as e:
                print(f"[DB ERROR] Delete failed !: {e}")
                messagebox.showerror("Error", f"Failed to delete/archive task: {e}")

            # Reload data from DB and update UI
            self._load_tasks_from_db()
            self.current_task_idx = self.current_task_idx % len(self.tasks) if self.tasks else 0
            self._set_text_x_offset(0)
            self._update_task_card_text()

    def _new_task(self):
        """Switches to the new task creation page and resets fields."""
        if getattr(self, 'is_animating', False): return
        self.new_task_name_var.set("")
        self.new_task_desc_var.set("")
        self.new_task_type_combo.current(0)

        self.current_cls_tags = ["Class A", "Class B"]
        if hasattr(self, "cls_listbox"):
          self._refresh_cls_listbox()

        self._update_algo_options()
        self.new_task_page.tkraise()

    # ── New Task page (scrollable form) ──────────────────────────────
    def build_new_task_panel(self):
        """Builds the scrollable UI for creating a new custom task (Dynamic Parameters & Robust Touch)"""

        # 1. 构建底层滚动框架 (Canvas + Scrollbar)
        self.task_scrollbar = tk.Scrollbar(self.new_task_page, orient="vertical", width=22)
        self.task_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.task_scroll_canvas = tk.Canvas(self.new_task_page, bg=self.SCREEN_BG, highlightthickness=0)
        self.task_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.task_scroll_canvas.configure(yscrollcommand=self.task_scrollbar.set)
        self.task_scrollbar.configure(command=self.task_scroll_canvas.yview)

        self.new_task_content = tk.Frame(self.task_scroll_canvas, bg=self.SCREEN_BG)
        self.canvas_window = self.task_scroll_canvas.create_window((0, 0), window=self.new_task_content, anchor="n")

        def _configure_content(event):
            self.task_scroll_canvas.configure(scrollregion=self.task_scroll_canvas.bbox("all"))
        def _configure_canvas(event):
            self.task_scroll_canvas.itemconfig(self.canvas_window, width=event.width)
            self.task_scroll_canvas.coords(self.canvas_window, event.width / 2, 0)

        self.new_task_content.bind("<Configure>", _configure_content)
        self.task_scroll_canvas.bind("<Configure>", _configure_canvas)

        # 2. 核心触屏滑动引擎 & 完美的防误触逻辑
        self.touch_start_y = 0

        def _on_swipe_start(event):
            self.touch_start_y = event.y_root
            self.task_scroll_canvas.scan_mark(event.x_root, event.y_root)

        def _on_swipe_drag(event):
            self.task_scroll_canvas.scan_dragto(event.x_root, event.y_root, gain=1)

        # 专为输入框设计的绑定连招
        def _record_touch(event):
            self.touch_start_y = event.y_root

        def _safe_open(target_var, title, event):
            if abs(event.y_root - self.touch_start_y) > 15: return
            self.open_keyboard(target_var, title)

        def bind_touch_keyboard(widget, target_var, title):
            """统一绑定：按下时记录坐标，抬起时安全弹键盘"""
            widget.bind("<ButtonPress-1>", _record_touch, add="+")
            widget.bind("<ButtonRelease-1>", lambda e, v=target_var, t=title: _safe_open(v, t, e), add="+")

        # 3. 在内容 Frame 中放置表单控件
        center_frame = tk.Frame(self.new_task_content, bg=self.SCREEN_BG)
        center_frame.pack(pady=(20, 40))

        tk.Label(center_frame, text="[ CREATE NEW TASK ]", font=self.FONT_TITLE, bg=self.SCREEN_BG, fg=self.SCREEN_FG).grid(row=0, column=0, columnspan=2, pady=(0, 30))
        def screen_label(text, parent=center_frame): return tk.Label(parent, text=text, bg=self.SCREEN_BG, fg=self.SCREEN_FG, font=self.FONT_SYS)

        # (1) Task Name
        screen_label("TASK NAME :").grid(row=1, column=0, padx=10, pady=10, sticky=tk.E)
        self.new_task_name_var = tk.StringVar()
        self.entry_new_name = tk.Entry(center_frame, textvariable=self.new_task_name_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=24)
        self.entry_new_name.config(state="readonly")
        bind_touch_keyboard(self.entry_new_name, self.new_task_name_var, "TASK NAME") # 使用新方法绑定
        self.entry_new_name.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)

        # (2) Task Description
        screen_label("TASK DESC :").grid(row=2, column=0, padx=10, pady=10, sticky=tk.NE)
        self.new_task_desc_var = tk.StringVar()
        self.entry_new_desc = tk.Label(center_frame, textvariable=self.new_task_desc_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=24, height=4, anchor="nw", justify="left", wraplength=260)
        bind_touch_keyboard(self.entry_new_desc, self.new_task_desc_var, "DESCRIPTION") # 使用新方法绑定
        self.entry_new_desc.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)

        # (3) Task Type Selection
        screen_label("TASK TYPE :").grid(row=3, column=0, padx=10, pady=10, sticky=tk.E)
        self.new_task_type_combo = ttk.Combobox(center_frame, font=self.FONT_SYS, state="readonly", width=22)
        self.new_task_type_combo["values"] = ("Classification", "Regression")
        self.new_task_type_combo.current(0)
        self.new_task_type_combo.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
        self.new_task_type_combo.bind("<<ComboboxSelected>>", self._update_algo_options)

        # (4) Algorithm Selection
        screen_label("ALGORITHM :").grid(row=4, column=0, padx=10, pady=10, sticky=tk.E)
        self.new_task_algo_combo = ttk.Combobox(center_frame, font=self.FONT_SYS, state="readonly", width=22)
        self.new_task_algo_combo.grid(row=4, column=1, padx=10, pady=10, sticky=tk.W)


        # �动态预埋参数组件
        # --- 分类参数组件 ---
        self.current_cls_tags = ["ClassA", "ClassB"] # 局部临时标签数组

        self.lbl_cls_tag = screen_label("CLASS TAG :")
        self.dynamic_cls_tag_var = tk.StringVar()
        self.entry_cls_tag = tk.Entry(center_frame, textvariable=self.dynamic_cls_tag_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=24)
        self.entry_cls_tag.config(state="readonly")
        bind_touch_keyboard(self.entry_cls_tag, self.dynamic_cls_tag_var, "CLASS TAG")
        bind_touch_keyboard(self.lbl_cls_tag, self.dynamic_cls_tag_var, "CLASS TAG")

        # 标签操作按钮栏 (新建、修改、删除)
        self.cls_btn_frame = tk.Frame(center_frame, bg=self.SCREEN_BG)
        tk.Button(self.cls_btn_frame, text="[ADD]", font=("Courier New", 11, "bold"), bg="#4caf50", fg="#111", command=self._add_cls_tag, width=6).pack(side=tk.LEFT, padx=4)
        tk.Button(self.cls_btn_frame, text="[EDIT]", font=("Courier New", 11, "bold"), bg="#2196f3", fg="#fff", command=self._edit_cls_tag, width=6).pack(side=tk.LEFT, padx=4)
        tk.Button(self.cls_btn_frame, text="[DEL]", font=("Courier New", 11, "bold"), bg="#f44336", fg="#fff", command=self._del_cls_tag, width=6).pack(side=tk.LEFT, padx=4)

        # 标签展示列表 (调大字体适配触屏)
        self.cls_listbox = tk.Listbox(center_frame, font=("Courier New", 12, "bold"), bg="#404040", fg="#39ff14",
                                      selectbackground="#2b2b2b", selectforeground="#39ff14", height=3, width=24, highlightthickness=1, relief=tk.SUNKEN,exportselection=False  )
        self.cls_listbox.bind("<<ListboxSelect>>", self._on_listbox_select)
        self._refresh_cls_listbox() # 初始化列表内容

        # --- 回归参数组件 ---
        self.lbl_reg_tgt = screen_label("TARGET NAME:")
        self.dynamic_reg_target_var = tk.StringVar(value="Concentration")
        self.entry_reg_tgt = tk.Entry(center_frame, textvariable=self.dynamic_reg_target_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=24)
        self.entry_reg_tgt.config(state="readonly")
        bind_touch_keyboard(self.entry_reg_tgt, self.dynamic_reg_target_var, "TARGET NAME")
        bind_touch_keyboard(self.lbl_reg_tgt, self.dynamic_reg_target_var, "TARGET NAME")

        self.lbl_reg_rng = screen_label("EXPECT RANGE:")
        self.dynamic_reg_range_var = tk.StringVar(value="0-100")
        self.entry_reg_rng = tk.Entry(center_frame, textvariable=self.dynamic_reg_range_var, font=self.FONT_SYS, bg="#b0b0b0", fg=self.SCREEN_FG, relief=tk.SUNKEN, bd=3, width=24)
        self.entry_reg_rng.config(state="readonly")
        bind_touch_keyboard(self.entry_reg_rng, self.dynamic_reg_range_var, "TARGET RANGE")
        bind_touch_keyboard(self.lbl_reg_rng, self.dynamic_reg_range_var, "TARGET RANGE")

        self._update_algo_options() # 初始化显示

        # 底部按键
        btn_frame = tk.Frame(center_frame, bg=self.SCREEN_BG)
        btn_frame.grid(row=8, column=0, columnspan=2, pady=40)
        tk.Button(btn_frame, text="< CANCEL", command=self.show_task_page, font=self.FONT_SYS, bg="#cccccc", fg=self.SCREEN_FG, relief=tk.RAISED, bd=3, width=12).pack(side=tk.LEFT, padx=20)
        tk.Button(btn_frame, text="CONFIRM >", command=self._save_new_task, bg="#4caf50", fg="#111111", font=self.FONT_SYS, relief=tk.RAISED, bd=3, width=12).pack(side=tk.LEFT, padx=20)

        # 4. 全局滑动穿透
        def _bind_mouse_scroll(widget):
            widget.bind("<ButtonPress-1>", _on_swipe_start, add="+")
            widget.bind("<B1-Motion>", _on_swipe_drag, add="+")
            for child in widget.winfo_children():
                if child.winfo_class() in ('Frame', 'Label'):
                    _bind_mouse_scroll(child)

        _bind_mouse_scroll(self.task_scroll_canvas)
        _bind_mouse_scroll(self.new_task_content)

    # ── Tag management ───────────────────────────────────────────────
    def _add_cls_tag(self):
        """新建标签到列表"""
        val = self.dynamic_cls_tag_var.get().strip()
        if val and val not in self.current_cls_tags:
            self.current_cls_tags.append(val)
            self._refresh_cls_listbox()
            self.dynamic_cls_tag_var.set("") # 清空输入框

    def _edit_cls_tag(self):
        """修改选中的标签 (绝对防弹版)"""
        # 1. 检查我们手动记录的索引是否存在
        if not hasattr(self, 'selected_cls_idx') or self.selected_cls_idx is None:
            messagebox.showwarning("Warning", "Please tap a tag from the list first to edit it.")
            return

        # 2. 获取新输入的值
        val = self.dynamic_cls_tag_var.get().strip()
        if val:
            try:
                # 直接使用我们记住的索引进行修改
                self.current_cls_tags[self.selected_cls_idx] = val
                self._refresh_cls_listbox()
                self.dynamic_cls_tag_var.set("") # 清空输入框

                # 修改完成后，重置记录的索引并清空列表高亮
                self.selected_cls_idx = None
                self.cls_listbox.selection_clear(0, tk.END)
            except IndexError:
                pass
        else:
            messagebox.showwarning("Warning", "Tag name cannot be empty!")

    def _del_cls_tag(self):
        """删除选中的标签 (绝对防弹版)"""
        # 1. 检查我们手动记录的索引是否存在
        if not hasattr(self, 'selected_cls_idx') or self.selected_cls_idx is None:
            messagebox.showwarning("Warning", "Please tap a tag from the list first to delete it.")
            return

        try:
            # 直接使用我们记住的索引进行删除
            del self.current_cls_tags[self.selected_cls_idx]
            self._refresh_cls_listbox()
            self.dynamic_cls_tag_var.set("") # 清空输入框

            # 删除完成后，重置记录的索引
            self.selected_cls_idx = None
        except IndexError:
            pass

    def _refresh_cls_listbox(self):
        """刷新展示列表内容"""
        self.cls_listbox.delete(0, tk.END)
        for tag in self.current_cls_tags:
            self.cls_listbox.insert(tk.END, f" • {tag}")
        # 联动刷新外层滚动画布的尺寸
        self.root.after(50, lambda: self.task_scroll_canvas.configure(scrollregion=self.task_scroll_canvas.bbox("all")))

    def _on_listbox_select(self, event):
        """点击列表条目时，手动记录选中的索引，并回填内容"""
        try:
            selections = self.cls_listbox.curselection()
            if selections:
                # 核心魔法：手动把行号存起来，死死咬住，绝不丢失！
                self.selected_cls_idx = selections[0]

                # 去掉显示用的 " • " 前缀还原文本
                raw_tag = self.current_cls_tags[self.selected_cls_idx]
                self.dynamic_cls_tag_var.set(raw_tag)
        except Exception:
            pass

    def _update_algo_options(self, event=None):
        """动态切换分类/回归的算法列表，并重新编排复杂的参数行"""
        task_type = self.new_task_type_combo.get()

        if task_type == "Classification":
            self.new_task_algo_combo["values"] = ("SVM (Support Vector)", "Random Forest", "KNN (K-Nearest)", "LDA (Linear Discriminant)")
            # 隐藏回归
            self.lbl_reg_tgt.grid_remove(); self.entry_reg_tgt.grid_remove()
            self.lbl_reg_rng.grid_remove(); self.entry_reg_rng.grid_remove()
            # 展开全新的分类标签三行套件
            self.lbl_cls_tag.grid(row=5, column=0, padx=10, pady=5, sticky=tk.E)
            self.entry_cls_tag.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
            self.cls_btn_frame.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)
            self.cls_listbox.grid(row=7, column=1, padx=10, pady=5, sticky=tk.W)
        else:
            self.new_task_algo_combo["values"] = ("PLSR (Partial Least Sq)", "PCR (Principal Comp)", "SVR (Support Vector)")
            # 隐藏分类套件
            self.lbl_cls_tag.grid_remove(); self.entry_cls_tag.grid_remove()
            self.cls_btn_frame.grid_remove(); self.cls_listbox.grid_remove()
            # 展开回归两行套件
            self.lbl_reg_tgt.grid(row=5, column=0, padx=10, pady=10, sticky=tk.E)
            self.entry_reg_tgt.grid(row=5, column=1, padx=10, pady=10, sticky=tk.W)
            self.lbl_reg_rng.grid(row=6, column=0, padx=10, pady=10, sticky=tk.E)
            self.entry_reg_rng.grid(row=6, column=1, padx=10, pady=10, sticky=tk.W)

        self.new_task_algo_combo.current(0)
        # 刷新滚动区域
        self.root.after(50, lambda: self.task_scroll_canvas.configure(scrollregion=self.task_scroll_canvas.bbox("all")))

    # ── Save new task ─────────────────────────────────────────────────
    def _save_new_task(self):
        """完全基于 JSON 格式的数据持久化"""

        name = self.new_task_name_var.get().strip()
        desc = self.new_task_desc_var.get().strip()
        if not name: name = "Unnamed Task"

        task_type = self.new_task_type_combo.get()
        algo = self.new_task_algo_combo.get()

        # 1. 获取标签，校验数据
        tags = []
        if task_type == "Classification":
            tags = getattr(self, 'current_cls_tags', [])
            if len(tags) < 2:
                messagebox.showwarning("Validation Error", "Classification requires at least 2 tags!\nPlease add more class tags.")
                return

        dup_check = sqlite3.connect(self.db_path)
        dup_cursor = dup_check.cursor()
        dup_cursor.execute("SELECT COUNT(*) FROM tasks WHERE name = ?", (name,))
        if dup_cursor.fetchone()[0] > 0:
            dup_check.close()
            messagebox.showwarning("Duplicate Name", f"Task '{name}' already exists!")
            return
        dup_check.close()

        # 2. 🚨 核心改动：把所有参数打包成字典，然后转成标准的 JSON 字符串
        meta_payload = {
            "raw_desc": desc,
            "task_type": task_type,
            "algorithm": algo,
            "classes": tags if task_type == "Classification" else [],
            "reg_target": self.dynamic_reg_target_var.get() if hasattr(self, 'dynamic_reg_target_var') and task_type != "Classification" else "",
            "reg_range": self.dynamic_reg_range_var.get() if hasattr(self, 'dynamic_reg_range_var') and task_type != "Classification" else ""
        }

        # 把字典变成 JSON 字符串 (这就绝不会有换行符错乱的问题了)
        formatted_desc = json.dumps(meta_payload, ensure_ascii=False)

        # 3. 初始化空模型文件
        model_dir = "models"
        if not os.path.exists(model_dir): os.makedirs(model_dir)

        if algo == "SVM (Support Vector)": empty_model = SVC(probability=True) if task_type == "Classification" else SVR()
        elif algo == "Random Forest": empty_model = RandomForestClassifier()
        elif algo == "KNN (K-Nearest)": empty_model = KNeighborsClassifier()
        elif algo == "LDA (Linear Discriminant)": empty_model = LinearDiscriminantAnalysis()
        elif algo == "PLSR (Partial Least Sq)": empty_model = PLSRegression()
        elif algo == "PCR (Principal Comp)": empty_model = PLSRegression()  # 用 PLS 作为 PCR 占位基底，训练时 Pipeline 会插入 PCA
        elif algo == "SVR (Support Vector)": empty_model = SVR()
        else: empty_model = SVC(probability=True)

        safe_filename = name.replace(" ", "_").replace("/", "_") + ".pkl"
        file_path = os.path.join(model_dir, safe_filename)

        try: joblib.dump(empty_model, file_path)
        except Exception as e: print(f"[ERROR] 模型保存失败: {str(e)}")

        # 4. 写入 SQLite 数据库
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks (name, description, task_type, algorithm, model_path, is_trained)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, formatted_desc, task_type, algo, file_path, False))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DB ERROR] 写入数据库失败: {e}")
            messagebox.showerror("Database Error", f"Failed to save task to database:\n{e}")
            return

        # 4.5 生成任务数据文件 (NPY + JSONL)
        try:
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            safe_name = self._sanitize_name(name)

            # 空的特征缓存文件 (X: 光谱, y: 标签)
            x_path = os.path.join(data_dir, f"task_{safe_name}_X.npy")
            y_path = os.path.join(data_dir, f"task_{safe_name}_y.npy")
            if not os.path.exists(x_path):
                np.save(x_path, np.array([], dtype=np.float64).reshape(0, 1))
            if not os.path.exists(y_path):
                np.save(y_path, np.array([], dtype=object))

            # 空的性能日志文件
            perf_path = os.path.join(data_dir, f"task_{safe_name}_perf.jsonl")
            if not os.path.exists(perf_path):
                with open(perf_path, 'w', encoding='utf-8') as f:
                    pass  # 创建空文件

            print(f"[SYS] Data files created for task '{name}' -> task_{safe_name}_*")
        except Exception as e:
            print(f"[WARN] Failed to create data files for task '{name}': {e}")

        # 5. 刷新界面
        self._load_tasks_from_db()
        self.current_task_idx = len(self.tasks) - 1
        if hasattr(self, '_set_text_x_offset'): self._set_text_x_offset(0)
        if hasattr(self, '_update_task_card_text'): self._update_task_card_text()
        self.show_task_page()
