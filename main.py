# main.py
import tkinter as tk
from data.database import DatabaseManager
from core.hardware_mgr import HardwareManager
from core.ml_engine import MLEngine
from ui.app_window import MainWindow

def main():
    root = tk.Tk()
    
    # 1. 启动数据引擎
    db_mgr = DatabaseManager()
    
    # 2. 启动硬件与算法引擎
    hw_mgr = HardwareManager()
    ml_engine = MLEngine()
    
    # 3. 启动 UI，并将引擎的控制权交接给它
    app = MainWindow(root, db=db_mgr, hw=hw_mgr, ml=ml_engine)
    
    # 4. 进入系统循环
    root.mainloop()

if __name__ == "__main__":
    main()