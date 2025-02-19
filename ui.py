#!/usr/bin/env python3
import ttkbootstrap as ttk
from ui.main_application import MainApplication

if __name__ == '__main__':
    # Create the main window with a dark theme (e.g., "darkly")
    root = ttk.Window(themename="darkly", title="ADAS System v0.032_alpha", iconphoto="assets/icon.PNG")
    root.geometry("1024x768")

    # Expand (maximize) the window at start.
    root.state('zoomed')

    # Available theme names include:
    # "default", "flatly", "darkly", "cyborg", "superhero", "vapor",
    # "litera", "journal", "minty", "pulse", "sandstone", "simplex",
    # "spacelab", "united", "yeti"

    app = MainApplication(root)
    app.pack(fill='both', expand=True)
    root.mainloop()
