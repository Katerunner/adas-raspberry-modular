# ui/base/updatable_frame.py
import tkinter as tk


class UpdatableFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.after_ids = []

    def add_after(self, delay, func):
        aid = self.after(delay, func)
        self.after_ids.append(aid)
        return aid

    def cancel_updates(self):
        for aid in self.after_ids:
            try:
                self.after_cancel(aid)
            except Exception as e:
                print("Error cancelling after id:", aid, e)
        self.after_ids = []

    def destroy(self):
        self.cancel_updates()
        super().destroy()
