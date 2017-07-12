from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib import gridspec
import numpy as np
import threading
import queue
import os
import sys
import ctypes

plt.rc('font', **{'size': 14})

q1 = queue.Queue()


def make_fourier_trans(t, sig, normalize=False, subtract_mean=False, add_zeros=0):
    if subtract_mean:
        sig = sig - sig.mean()
    for i in range(add_zeros):
        if i == 5:
            break
        t = np.append(t, t + t[-1] + t[1])
        sig = np.append(0 * sig, sig)
    n = len(sig)
    fourier_trans = np.fft.fftshift(np.fft.fft(sig)) / n * (t.max() - t.min())
    fourier_abs = np.abs(fourier_trans) ** 2
    if normalize:
        fourier_abs = fourier_abs / fourier_abs.max()

    diffs = (t - np.roll(t, 1))[1:]
    n_data = max(fourier_trans.shape)
    freq = np.fft.fftshift(np.fft.fftfreq(n_data, d=diffs.mean()))
    return freq, fourier_trans, fourier_abs


class NewCBox(ttk.Combobox):
    def __init__(self, master, dictionary, current=0, *args, **kw):
        ttk.Combobox.__init__(self, master, values=list(dictionary.keys()), state='readonly', *args, **kw)
        self.dictionary = dictionary
        self.set(current)

    def value(self):
        return self.dictionary[self.get()]


class LabelWithEntry(tk.Frame):
    def __init__(self, master, text, width_label=10, width_entry=20, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.label = ttk.Label(self, text=text, width=width_label)
        self.label.grid(row=0, column=0)

        self.entry = ttk.Entry(self, width=width_entry)
        self.entry.grid(row=0, column=1)

    def get(self):
        return self.entry.get()

    def set(self, value, fmt="{0:6.4f}"):
        self.entry.delete(0, 'end')
        self.entry.insert(0, fmt.format(value))


class EmbeddedFigure(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master

        self.options_dict = {'logarithmic': [False, False]}

        self.f = plt.Figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        self.subplot1 = self.f.add_subplot(gs[0])
        self.subplot1.format_coord = lambda x, y: "x={0:1.2f}, y={1:1.1f}".format(x, y)

        self.subplot2 = self.f.add_subplot(gs[1])
        self.subplot2.format_coord = lambda x, y: "x={0:1.2f}, y={1:1.1f}".format(x, y)

        colortuple = master.winfo_rgb(self.master.cget('bg'))
        color_rgb = [x / 16 ** 4 for x in colortuple]
        self.f.patch.set_facecolor(color_rgb)
        self.subplot1.set_ylim(0, 1)
        self.subplot2.set_xlabel('')
        self.subplot2.set_ylim(0, 0.01)
        self.subplot2.set_ylabel('Spectrum')

        self.f.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.f, self)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.last_data = None

    def update_figure(self, data=None):

        if data is None:
            if self.last_data is None:
                return
            else:
                data = self.last_data
        else:
            self.last_data = data

        x = data[:, 0]
        y = data[:, 1]

        self.subplot1.cla()
        self.subplot2.cla()

        self.subplot1.plot(x, y)
        self.subplot1.set_xlim(x.min(), x.max())

        freqs, fft_sig, fft_sig_abs = make_fourier_trans(x, y)
        self.subplot2.plot(freqs, fft_sig_abs)

        self.subplot2.set_xlim(0, freqs.max())
        self.subplot2.set_ylabel('Spectrum')

        if self.options_dict['logarithmic'][0]:
            self.subplot2.set_xlim(left=(freqs.max() - freqs.min()) / 1000)
            self.subplot2.set_xscale('log')

        else:
            self.subplot2.set_xscale('linear')

        if self.options_dict['logarithmic'][1]:
            self.subplot2.set_yscale('log')
        else:
            self.subplot2.set_yscale('linear')

        plt.pause(0.001)
        self.canvas.draw()


class OptionWindow(tk.Toplevel):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.master = master

        self.time_column_entry = LabelWithEntry(self, 'Column time:', width_label=20, width_entry=5)
        self.time_column_entry.set(0, fmt="{0:2d}")
        self.time_column_entry.pack()

        self.thz_column_entry = LabelWithEntry(self, 'Column Signal:', width_label=20, width_entry=5)
        self.thz_column_entry.set(0, fmt="{0:2d}")
        self.thz_column_entry.pack()

        self.ok_button = ttk.Button(self, text="OK", command=self.set_values, takefocus=False)
        self.ok_button.pack()

    def set_values(self):
        print(int(self.time_column_entry.get()))
        print(int(self.thz_column_entry.get()))


class ButtonFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master

        def random_plot():
            data = np.zeros((1000, 2))
            x = np.linspace(0, 1, 1000)
            data[:, 0] = x
            data[:, 1] = 0.2 * np.random.randn(1000) + np.sin(2 * np.pi * 10 * x) + 0.5 * np.sin(
                2 * np.pi * 20 * x) + 0.25 * np.sin(2 * np.pi * 30 * x) + np.sin(2 * np.pi * 40 * x) / 8 + np.sin(
                2 * np.pi * 50 * x) / 16 + np.sin(2 * np.pi * 60 * x) / 32

            master.embedded_figure.update_figure(data=data)

        self.plot_button = ttk.Button(self, text='Plot', command=random_plot, takefocus=False)
        self.plot_button.pack(ipady=15, side=tk.LEFT)


class FigureOptionFrame(tk.Frame):
    def __init__(self, master, option_dict, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.option_dict = option_dict

        self.padx = 2

        self.log_var_x = tk.IntVar()
        self.log_var_x.set(self.option_dict['logarithmic'][0])
        self.log_checkbutton_x = ttk.Checkbutton(self, text="x-Log", variable=self.log_var_x, takefocus=False,
                                                 command=self.set_values_for_figure)
        self.log_checkbutton_x.pack(side=tk.LEFT, padx=self.padx)

        self.log_var_y = tk.IntVar()
        self.log_var_y.set(self.option_dict['logarithmic'][1])
        self.log_checkbutton_y = ttk.Checkbutton(self, text="y-Log", variable=self.log_var_y, takefocus=False,
                                                 command=self.set_values_for_figure)
        self.log_checkbutton_y.pack(side=tk.LEFT, padx=self.padx)

    def set_values_for_figure(self):
        if self.log_var_x.get():
            self.option_dict['logarithmic'][0] = True
        else:
            self.option_dict['logarithmic'][0] = False

        if self.log_var_y.get():
            self.option_dict['logarithmic'][1] = True
        else:
            self.option_dict['logarithmic'][1] = False

        q1.put('update figure')


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.master = master

        master.resizable(width=False, height=False)
        master.geometry('{}x{}'.format(1000, 920))

        # master.wm_attributes("-topmost", 1)

        self.button_frame = ButtonFrame(self)
        self.button_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.embedded_figure = EmbeddedFigure(self)
        self.embedded_figure.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.figure_option_frame = FigureOptionFrame(self, self.embedded_figure.options_dict)
        self.figure_option_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.option_window = None

        self.create_menubar()
        self.bind_global_commands()

        master.title('Example Gui')
        master.config(menu=self.menubar)
        master.protocol('WM_DELETE_WINDOW', self.quit_program)
        master.after(100, self.update_status)

    def update_status(self):
        root.after(100, self.update_status)
        if not q1.empty():
            q_el = q1.get()
            if q_el == 'update figure':
                self.embedded_figure.update_figure()

    def create_menubar(self):
        self.menubar = tk.Menu(self, tearoff=0)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.quit_program, accelerator="Strq+q")

        self.optionmenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Options", menu=self.optionmenu)

        self.optionmenu.add_command(label='Example', command=self.open_option_window)

    def quit_program(self):
        self.master.destroy()

    def bind_global_commands(self):
        self.bind_all('<Control-q>', lambda event: self.quit_program())

    def open_option_window(self):
        if self.option_window is None or not self.option_window.winfo_exists():
            self.option_window = OptionWindow(self)


root = tk.Tk()
# root.iconbitmap(r'icon.ico')
app = Application(master=root)
root.mainloop()
