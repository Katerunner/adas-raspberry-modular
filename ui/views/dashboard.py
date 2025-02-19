# ui/views/dashboard.py
import ttkbootstrap as tk
from ui.base.updatable_frame import UpdatableFrame
from ui.views.rapidview import RapidView
from ui.views.fullview import FullView


class DashboardView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.btn_rapid = tk.Button(btn_frame, text="Rapid View", command=self.show_rapid)
        self.btn_rapid.pack(side=tk.LEFT, padx=5)
        self.btn_full = tk.Button(btn_frame, text="Full View", command=self.show_full)
        self.btn_full.pack(side=tk.LEFT, padx=5)
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        self.current_subview = None
        self.show_rapid()

    def show_rapid(self):
        if self.current_subview:
            self.current_subview.destroy()
        self.current_subview = RapidView(self.content_frame, self.ps)
        self.current_subview.pack(fill=tk.BOTH, expand=True)

    def show_full(self):
        if self.current_subview:
            self.current_subview.destroy()
        self.current_subview = FullView(self.content_frame, self.ps)
        self.current_subview.pack(fill=tk.BOTH, expand=True)
