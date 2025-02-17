# ui.py
from ui.main_application import MainApplication
import tkinter as tk

if __name__ == '__main__':
    root = tk.Tk()
    app = MainApplication(root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
